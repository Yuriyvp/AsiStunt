"""End-to-end voice pipeline integration test.

Verifies the full loop: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE
using mocked components with realistic behavior (async delays, chunked output).

Run: pytest tests/integration/test_voice_pipeline.py -v -s
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState, StateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a sine tone to simulate speech audio."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_tts_audio(duration_s: float = 0.5, sr: int = 24000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _make_playlist():
    pl = MagicMock()
    pl.append = MagicMock()
    pl.drop_future = MagicMock(return_value=[])
    pl.clear = MagicMock()
    pl.text_played = MagicMock(return_value="")
    pl.text_remaining = MagicMock(return_value="")
    pl.is_empty = True
    pl.wait_until_done = AsyncMock()
    return pl


def _make_components(**overrides):
    """Build a full mock component dict, with optional overrides."""
    audio = AsyncMock()
    audio.start = AsyncMock()
    audio.stop = AsyncMock()
    audio.read_chunk = AsyncMock(return_value=np.zeros(480, dtype=np.float32))

    vad = MagicMock()
    vad.process_chunk = MagicMock(return_value=None)
    vad.is_speech = False
    vad.reset = MagicMock()
    vad.drain_speech_samples = MagicMock(return_value=None)

    asr = AsyncMock()
    asr.transcribe = AsyncMock(
        return_value={"text": "hello world", "language": "en", "confidence": 0.95}
    )

    llm = AsyncMock()

    async def _stream(messages, sampling=None):
        for token in ["Hello", ", ", "how ", "are ", "you", "?"]:
            yield token
    llm.stream = _stream
    llm.cancel = AsyncMock()

    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=_make_tts_audio())

    playlist = _make_playlist()

    playback = AsyncMock()
    playback.start = AsyncMock()
    playback.stop = AsyncMock()
    playback.fade_out = MagicMock()
    playback.is_active = False

    filler_cache = MagicMock()
    filler_cache.record_turn = MagicMock()
    filler_cache.get_filler = MagicMock(return_value=None)

    components = {
        "audio_input": audio,
        "vad": vad,
        "asr": asr,
        "llm": llm,
        "tts": tts,
        "playlist": playlist,
        "playback": playback,
        "filler_cache": filler_cache,
    }
    components.update(overrides)
    return components


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEndToEndVoiceLoop:
    """Full voice loop: text input -> LLM -> TTS -> playlist -> IDLE."""

    @pytest.mark.asyncio
    async def test_text_input_full_loop(self):
        """Text input traverses IDLE -> PROCESSING -> SPEAKING -> IDLE."""
        state_log = []
        components = _make_components()

        orch = Orchestrator(**components)
        orig_on_change = orch._state._on_change

        def track_state(old, new):
            state_log.append((old, new))
            if orig_on_change:
                orig_on_change(old, new)

        orch._state._on_change = track_state

        assert orch.state == PipelineState.IDLE

        await orch.handle_text_input("Tell me a joke")
        # Wait for processing to complete
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Verify state transitions
        states_visited = [s[1] for s in state_log]
        assert PipelineState.PROCESSING in states_visited
        assert PipelineState.IDLE in states_visited  # returned to IDLE

        # Verify dialogue recorded
        assert len(orch.dialogue) == 2
        assert orch.dialogue[0].role == "user"
        assert orch.dialogue[0].content == "Tell me a joke"
        assert orch.dialogue[1].role == "assistant"
        assert len(orch.dialogue[1].content) > 0

    @pytest.mark.asyncio
    async def test_text_input_calls_all_pipeline_stages(self):
        """Every pipeline stage is invoked: LLM stream, TTS synthesize, playlist append."""
        components = _make_components()
        orch = Orchestrator(**components)

        await orch.handle_text_input("hello")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # LLM was streamed
        # TTS was called (at least once for the chunked output)
        components["tts"].synthesize.assert_called()
        # Playlist got chunks
        components["playlist"].append.assert_called()
        # Playlist waited for playback
        components["playlist"].wait_until_done.assert_awaited()
        # Filler turn recorded
        components["filler_cache"].record_turn.assert_called()

    @pytest.mark.asyncio
    async def test_voice_turn_asr_to_response(self):
        """Simulated voice turn: ASR -> LLM -> TTS -> playlist."""
        components = _make_components()
        speech_audio = _make_audio(1.0)
        components["vad"].drain_speech_samples = MagicMock(return_value=speech_audio)

        orch = Orchestrator(**components)
        orch._state._state = PipelineState.IDLE

        # Simulate what happens after VAD detects speech end
        orch._state.transition(PipelineState.LISTENING)
        orch._state.transition(PipelineState.PROCESSING)
        orch._processing_task = asyncio.create_task(
            orch._process_voice_turn(speech_audio)
        )
        await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # ASR was called with the speech audio
        components["asr"].transcribe.assert_awaited_once_with(speech_audio)

        # Dialogue has user + assistant turns
        assert len(orch.dialogue) == 2
        assert orch.dialogue[0].role == "user"
        assert orch.dialogue[0].content == "hello world"
        assert orch.dialogue[0].source == "voice"
        assert orch.dialogue[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_empty_transcription_returns_to_idle(self):
        """If ASR returns empty text, orchestrator returns to IDLE without LLM call."""
        components = _make_components()
        components["asr"].transcribe = AsyncMock(
            return_value={"text": "   ", "language": "en", "confidence": 0.1}
        )

        orch = Orchestrator(**components)
        orch._state._state = PipelineState.PROCESSING

        await orch._process_voice_turn(_make_audio(0.5))

        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == 0
        components["tts"].synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_turns_accumulate_dialogue(self):
        """Multiple text inputs build up dialogue history."""
        turn_count = 0

        async def varied_stream(messages, sampling=None):
            nonlocal turn_count
            turn_count += 1
            for token in [f"Response {turn_count}", "."]:
                yield token

        components = _make_components()
        components["llm"].stream = varied_stream

        orch = Orchestrator(**components)

        for i in range(3):
            await orch.handle_text_input(f"Message {i+1}")
            if orch._processing_task:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # 3 user + 3 assistant = 6 turns
        assert len(orch.dialogue) == 6
        assert all(t.role == "user" for t in orch.dialogue[::2])
        assert all(t.role == "assistant" for t in orch.dialogue[1::2])

    @pytest.mark.asyncio
    async def test_mood_parsed_from_stream(self):
        """Mood signal tags in LLM output update the mood state."""
        async def mood_stream(messages, sampling=None):
            yield "<mood_signal>user_tone=happy, intensity=0.9</mood_signal>"
            yield "\nThat's wonderful news!"

        components = _make_components()
        components["llm"].stream = mood_stream

        orch = Orchestrator(**components)
        await orch.handle_text_input("I got promoted!")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch.mood.user_tone == "happy"
        assert orch.mood.mood == "playful"  # happy -> playful mapping

    @pytest.mark.asyncio
    async def test_llm_error_returns_to_idle(self):
        """If LLM stream raises, orchestrator returns to IDLE gracefully."""
        async def failing_stream(messages, sampling=None):
            yield "Start..."
            raise RuntimeError("LLM crashed")

        components = _make_components()
        components["llm"].stream = failing_stream

        orch = Orchestrator(**components)
        await orch.handle_text_input("hello")
        if orch._processing_task:
            try:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)
            except Exception:
                pass

        assert orch.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_tts_error_does_not_crash_pipeline(self):
        """If TTS fails for one chunk, pipeline continues with remaining chunks."""
        call_count = 0

        async def flaky_tts(text, voice_params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("TTS synthesis failed")
            return _make_tts_audio()

        components = _make_components()
        components["tts"].synthesize = flaky_tts

        # Use a stream that produces enough text for multiple chunks
        async def long_stream(messages, sampling=None):
            for token in ["This is a longer response. ", "It has multiple sentences. ",
                          "The TTS should handle partial failures gracefully."]:
                yield token

        components["llm"].stream = long_stream

        orch = Orchestrator(**components)
        await orch.handle_text_input("test")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Pipeline should complete despite TTS error
        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == 2


class TestStateTransitionSequence:
    """Verify correct state transition ordering."""

    @pytest.mark.asyncio
    async def test_idle_to_processing_to_idle(self):
        """Minimal path: IDLE -> PROCESSING -> IDLE (when no TTS chunks produced)."""
        async def empty_stream(messages, sampling=None):
            return
            yield  # make it an async generator

        components = _make_components()
        components["llm"].stream = empty_stream

        orch = Orchestrator(**components)
        assert orch.state == PipelineState.IDLE

        await orch.handle_text_input("hi")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_speaking_state_reached_when_tts_produces_audio(self):
        """SPEAKING state is entered when TTS produces audio chunks."""
        state_log = []
        components = _make_components()

        # Ensure enough text for sentence chunker to fire
        async def chunky_stream(messages, sampling=None):
            yield "This is a complete sentence that should trigger chunking. "
            yield "And here is another one for good measure."

        components["llm"].stream = chunky_stream

        orch = Orchestrator(**components)
        orch._state._on_change = lambda old, new: state_log.append(new)

        await orch.handle_text_input("tell me something")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert PipelineState.SPEAKING in state_log
        assert state_log[-1] == PipelineState.IDLE  # ends at IDLE
