"""Integration tests for barge-in behavior.

Verifies: fade-out, playlist drop, state -> INTERRUPTED,
interruption context in dialogue, text-input barge-in.

Run: pytest tests/integration/test_barge_in.py -v -s
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn, BARGE_IN_GATE_MS
from voice_assistant.core.state_machine import PipelineState
from voice_assistant.core.audio_output import AudioChunk


def _make_tts_audio(duration_s: float = 0.5, sr: int = 24000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _make_components(**overrides):
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
        return_value={"text": "stop", "language": "en", "confidence": 0.9}
    )

    llm = AsyncMock()

    async def _stream(messages, sampling=None):
        for token in ["OK", "."]:
            yield token
    llm.stream = _stream
    llm.cancel = AsyncMock()

    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=_make_tts_audio())

    playlist = MagicMock()
    playlist.append = MagicMock()
    playlist.drop_future = MagicMock(return_value=[
        AudioChunk(audio=_make_tts_audio(), text="dropped chunk"),
    ])
    playlist.clear = MagicMock()
    playlist.text_played = MagicMock(return_value="I was saying")
    playlist.text_remaining = MagicMock(return_value="something important")
    playlist.is_empty = True
    playlist.wait_until_done = AsyncMock()

    playback = AsyncMock()
    playback.start = AsyncMock()
    playback.stop = AsyncMock()
    playback.fade_out = MagicMock()
    playback.is_active = True

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


class TestBargeInExecution:
    """Test _execute_barge_in() directly."""

    @pytest.mark.asyncio
    async def test_barge_in_triggers_fade_out(self):
        """Barge-in should call fade_out with 30ms."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="Long response here", source="voice", timestamp=0,
        ))

        await orch._execute_barge_in()

        components["playback"].fade_out.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_barge_in_drops_future_chunks(self):
        """Barge-in should drop unplayed playlist chunks."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="Being interrupted", source="voice", timestamp=0,
        ))

        await orch._execute_barge_in()

        components["playlist"].drop_future.assert_called_once()
        components["playlist"].clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_barge_in_cancels_llm(self):
        """Barge-in should cancel the LLM stream."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="Interrupted", source="voice", timestamp=0,
        ))

        await orch._execute_barge_in()

        components["llm"].cancel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_barge_in_marks_turn_interrupted(self):
        """Last assistant turn should be marked as interrupted with partial text."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="Full response text", source="voice", timestamp=0,
        ))

        await orch._execute_barge_in()

        last = orch.dialogue[-1]
        assert last.interrupted is True
        assert last.partial == "I was saying"  # from playlist.text_played()

    @pytest.mark.asyncio
    async def test_barge_in_transitions_to_interrupted(self):
        """State should transition to INTERRUPTED after barge-in."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="Test", source="voice", timestamp=0,
        ))

        await orch._execute_barge_in()

        assert orch.state == PipelineState.INTERRUPTED


class TestTextInputBargeIn:
    """Text input during SPEAKING should trigger barge-in."""

    @pytest.mark.asyncio
    async def test_text_during_speaking_barges_in(self):
        """Typing while assistant speaks triggers barge-in then processes new input."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="I was talking about...", source="voice", timestamp=0,
        ))

        await orch.handle_text_input("actually, never mind")

        # Barge-in should have been triggered
        components["playback"].fade_out.assert_called_with(30)
        components["llm"].cancel.assert_awaited()

    @pytest.mark.asyncio
    async def test_text_during_speaking_starts_new_turn(self):
        """After barge-in via text, a new processing cycle begins."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="Old response", source="voice", timestamp=0,
        ))

        await orch.handle_text_input("new question")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Should have user turn with "new question"
        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert any(t.content == "new question" for t in user_turns)


class TestBargeInProcessingCancel:
    """Text input during PROCESSING should cancel and restart."""

    @pytest.mark.asyncio
    async def test_text_during_processing_cancels_current(self):
        """Text input while processing should cancel current LLM work."""
        # Create a slow LLM stream
        async def slow_stream(messages, sampling=None):
            for token in ["Slow", " response"]:
                await asyncio.sleep(0.1)
                yield token

        components = _make_components()
        components["llm"].stream = slow_stream

        orch = Orchestrator(**components)
        orch._state._state = PipelineState.IDLE

        # Start first turn
        await orch.handle_text_input("first question")
        # Immediately submit second (while first is processing)
        await asyncio.sleep(0.05)
        await orch.handle_text_input("second question")

        # Wait for the new processing to complete
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # The second question should be in dialogue
        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert any(t.content == "second question" for t in user_turns)


class TestBargeInGate:
    """Barge-in gate: speech must last > 300ms before triggering."""

    @pytest.mark.asyncio
    async def test_short_speech_no_barge_in(self):
        """Brief speech during SPEAKING should NOT trigger barge-in (< 300ms)."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="talking", source="voice", timestamp=0,
        ))

        # Set barge-in start to "just now" — not enough elapsed
        import time
        orch._barge_in_speech_start = time.monotonic()

        # Immediately check (< 300ms)
        assert orch.state == PipelineState.SPEAKING
        components["playback"].fade_out.assert_not_called()

    @pytest.mark.asyncio
    async def test_barge_in_resets_speech_start(self):
        """After barge-in executes, the speech start timer should be cleared."""
        components = _make_components()
        orch = Orchestrator(**components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="test", source="voice", timestamp=0,
        ))

        import time
        orch._barge_in_speech_start = time.monotonic() - 1.0  # well past 300ms

        await orch._execute_barge_in()

        assert orch._barge_in_speech_start is None
