"""Integration tests for Orchestrator — all components mocked.

These tests verify the wiring and state transitions without requiring
real audio devices, models, or GPU. Full live testing is in Stage 20.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn, BARGE_IN_GATE_MS
from voice_assistant.core.state_machine import PipelineState


def _make_mock_playlist():
    """Mock playlist that auto-completes wait_until_done."""
    pl = MagicMock()
    pl.append = MagicMock()
    pl.drop_future = MagicMock(return_value=[])
    pl.clear = MagicMock()
    pl.text_played = MagicMock(return_value="")
    pl.text_remaining = MagicMock(return_value="")
    pl.is_empty = True

    pl.wait_until_done = AsyncMock()
    return pl


@pytest.fixture
def mock_components():
    """Create mocked versions of all orchestrator dependencies."""
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
    asr.transcribe = AsyncMock(return_value={"text": "hello", "language": "en", "confidence": 0.95})

    llm = AsyncMock()

    async def mock_stream(messages, sampling=None, **kwargs):
        for token in ["Hi", " there", "!"]:
            yield token
    llm.stream = mock_stream
    llm.cancel = AsyncMock()

    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=np.ones(2400, dtype=np.float32) * 0.3)

    playlist = _make_mock_playlist()

    playback = AsyncMock()
    playback.start = AsyncMock()
    playback.stop = AsyncMock()
    playback.fade_out = MagicMock()
    playback.is_active = False

    filler_cache = MagicMock()
    filler_cache.record_turn = MagicMock()
    filler_cache.get_filler = MagicMock(return_value=None)

    return {
        "audio_input": audio,
        "vad": vad,
        "asr": asr,
        "llm": llm,
        "tts": tts,
        "playlist": playlist,
        "playback": playback,
        "filler_cache": filler_cache,
    }


class TestOrchestratorConstruction:
    def test_instantiate(self, mock_components):
        orch = Orchestrator(**mock_components)
        assert orch.state == PipelineState.IDLE

    def test_dialogue_starts_empty(self, mock_components):
        orch = Orchestrator(**mock_components)
        assert orch.dialogue == []


class TestOrchestratorTextInput:
    @pytest.mark.asyncio
    async def test_text_input_processes_turn(self, mock_components):
        """Text input should go through LLM → TTS → playlist."""
        orch = Orchestrator(**mock_components)
        orch._state._state = PipelineState.IDLE

        await orch.handle_text_input("hello")
        # Give processing task time to complete
        await asyncio.sleep(0.05)
        if orch._processing_task and not orch._processing_task.done():
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert len(orch.dialogue) == 2
        assert orch.dialogue[0].role == "user"
        assert orch.dialogue[0].content == "hello"
        assert orch.dialogue[0].source == "text"
        assert orch.dialogue[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_text_input_calls_tts(self, mock_components):
        orch = Orchestrator(**mock_components)
        orch._state._state = PipelineState.IDLE

        await orch.handle_text_input("test message")
        await asyncio.sleep(0.05)
        if orch._processing_task and not orch._processing_task.done():
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        mock_components["tts"].synthesize.assert_called()

    @pytest.mark.asyncio
    async def test_text_input_during_speaking_barges_in(self, mock_components):
        orch = Orchestrator(**mock_components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="I was saying...", source="voice", timestamp=0,
        ))

        await orch.handle_text_input("interrupt!")
        mock_components["playback"].fade_out.assert_called_with(15)
        mock_components["llm"].cancel.assert_called()

    @pytest.mark.asyncio
    async def test_text_input_queues_to_playlist(self, mock_components):
        orch = Orchestrator(**mock_components)
        orch._state._state = PipelineState.IDLE

        await orch.handle_text_input("say something")
        await asyncio.sleep(0.05)
        if orch._processing_task and not orch._processing_task.done():
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        mock_components["playlist"].append.assert_called()


class TestOrchestratorTurn:
    def test_turn_dataclass(self):
        t = Turn(role="user", content="Hello", source="voice", timestamp=0.0)
        assert t.role == "user"
        assert t.content == "Hello"
        assert t.source == "voice"
        assert not t.interrupted
        assert t.partial is None

    @pytest.mark.asyncio
    async def test_filler_recorded_each_turn(self, mock_components):
        orch = Orchestrator(**mock_components)
        orch._state._state = PipelineState.IDLE

        await orch.handle_text_input("hello")
        await asyncio.sleep(0.05)
        if orch._processing_task and not orch._processing_task.done():
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        mock_components["filler_cache"].record_turn.assert_called()

    @pytest.mark.asyncio
    async def test_mood_updated_from_stream(self, mock_components):
        """LLM stream with mood tag should update mood state."""
        async def stream_with_mood(messages, sampling=None, **kwargs):
            yield "<mood_signal>user_tone=happy, intensity=0.8</mood_signal>"
            yield "\nGreat to hear!"
        mock_components["llm"].stream = stream_with_mood

        orch = Orchestrator(**mock_components)
        orch._state._state = PipelineState.IDLE

        await orch.handle_text_input("I got a promotion!")
        await asyncio.sleep(0.05)
        if orch._processing_task and not orch._processing_task.done():
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch.mood.user_tone == "happy"

    @pytest.mark.asyncio
    async def test_barge_in_marks_interrupted(self, mock_components):
        orch = Orchestrator(**mock_components)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="Long response...", source="voice", timestamp=0,
        ))

        await orch._execute_barge_in()
        assert orch.dialogue[-1].interrupted
        assert orch.state == PipelineState.INTERRUPTED


class TestOrchestratorImport:
    def test_import(self):
        from voice_assistant.core.orchestrator import Orchestrator
        assert Orchestrator is not None

    def test_state_machine_import(self):
        from voice_assistant.core.state_machine import StateMachine, PipelineState
        assert StateMachine is not None
        assert PipelineState.IDLE is not None
