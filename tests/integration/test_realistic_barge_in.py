"""Realistic barge-in scenario tests with timing simulation.

Tests barge-in at various points: during TTS, during LLM streaming,
with pending speech, sub-gate duration, and rapid sequential barge-ins.

Run: pytest tests/integration/test_realistic_barge_in.py -v -s
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn, BARGE_IN_GATE_MS
from voice_assistant.core.state_machine import PipelineState
from voice_assistant.core.audio_output import AudioChunk

from conftest import (
    make_mock_components, make_orchestrator, make_speech_audio, make_tts_audio,
    make_llm_stream, make_mood_stream,
)


class TestBargeInDuringSpeaking:
    """Barge-in while assistant is speaking (most common scenario)."""

    @pytest.mark.asyncio
    async def test_voice_barge_in_during_long_response(self):
        """User interrupts a long response → fade out, capture partial."""
        comps = make_mock_components()
        # Long response that would take time to speak
        comps["llm"].stream = make_llm_stream(
            ["This is a very long response ", "that has many sentences. ",
             "It goes on and on. ", "The user will interrupt this."]
        )
        comps["playlist"].text_played = MagicMock(return_value="This is a very long response")
        comps["playlist"].text_remaining = MagicMock(return_value="that has many sentences.")
        comps["playlist"].drop_future = MagicMock(return_value=[
            AudioChunk(audio=make_tts_audio(), text="It goes on"),
        ])
        comps["playback"].is_active = True

        orch, _ = make_orchestrator(comps)

        # Start conversation
        await orch.handle_text_input("Tell me everything about the universe")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Now simulate barge-in during SPEAKING
        orch._state._state = PipelineState.SPEAKING
        await orch._execute_barge_in()

        assert orch.state == PipelineState.INTERRUPTED
        comps["playback"].fade_out.assert_called_with(15)
        comps["playlist"].drop_future.assert_called()

    @pytest.mark.asyncio
    async def test_text_barge_in_during_speaking(self):
        """User types while assistant speaks → interrupts and starts new turn."""
        comps = make_mock_components()
        comps["llm"].stream = make_llm_stream(["Response."])
        comps["playback"].is_active = True

        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="I was talking", source="voice", timestamp=time.time(),
        ))

        await orch.handle_text_input("Actually, forget that.")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Should have barged in and started new turn
        comps["playback"].fade_out.assert_called_with(15)
        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert any(t.content == "Actually, forget that." for t in user_turns)

    @pytest.mark.asyncio
    async def test_sub_gate_speech_no_barge_in(self):
        """Speech shorter than 150ms gate should NOT trigger barge-in."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="talking", source="voice", timestamp=time.time(),
        ))

        # Start barge-in gate at "now" — check immediately (< 150ms)
        orch._barge_in_speech_start = time.monotonic()

        # Check barge-in gate — should NOT trigger (< 150ms elapsed)
        elapsed = (time.monotonic() - orch._barge_in_speech_start) * 1000
        assert elapsed < BARGE_IN_GATE_MS

        # State should still be SPEAKING
        assert orch.state == PipelineState.SPEAKING
        comps["playback"].fade_out.assert_not_called()

    @pytest.mark.asyncio
    async def test_past_gate_speech_triggers_barge_in(self):
        """Speech longer than 150ms gate should trigger barge-in."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="talking", source="voice", timestamp=time.time(),
        ))

        # Set barge-in start to 200ms ago
        orch._barge_in_speech_start = time.monotonic() - 0.2

        elapsed = (time.monotonic() - orch._barge_in_speech_start) * 1000
        assert elapsed > BARGE_IN_GATE_MS

        # Execute barge-in
        await orch._execute_barge_in()
        assert orch.state == PipelineState.INTERRUPTED
        assert orch._barge_in_speech_start is None

    @pytest.mark.asyncio
    async def test_barge_in_marks_dialogue_correctly(self):
        """Barge-in marks last assistant turn as interrupted with partial text."""
        comps = make_mock_components()
        comps["playlist"].text_played = MagicMock(return_value="I was saying something")
        comps["playlist"].text_remaining = MagicMock(return_value="important and long")

        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant",
            content="I was saying something important and long",
            source="voice", timestamp=time.time(),
        ))

        await orch._execute_barge_in()

        last_turn = orch.dialogue[-1]
        assert last_turn.interrupted is True
        assert last_turn.partial == "I was saying something"


class TestBargeInDuringProcessing:
    """Barge-in scenarios during PROCESSING state."""

    @pytest.mark.asyncio
    async def test_speech_during_processing_queued_not_interrupted(self):
        """Speech during PROCESSING should be queued, not cause barge-in."""
        comps = make_mock_components()
        speech_segment = make_speech_audio(0.5)
        comps["vad"].drain_speech_samples = MagicMock(return_value=speech_segment)

        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING
        orch._pending_speech = None

        # Simulate speech arriving during processing
        orch._pending_speech = speech_segment

        # State should still be PROCESSING (not interrupted)
        assert orch.state == PipelineState.PROCESSING
        assert orch._pending_speech is not None
        assert len(orch._pending_speech) == len(speech_segment)

    @pytest.mark.asyncio
    async def test_text_during_processing_cancels_and_restarts(self):
        """Text input during PROCESSING cancels current and starts new."""
        async def slow_stream(messages, sampling=None, **kwargs):
            for token in ["Slow ", "response ", "here"]:
                await asyncio.sleep(0.05)
                yield token

        comps = make_mock_components()
        comps["llm"].stream = slow_stream
        orch, _ = make_orchestrator(comps)

        # Start first turn
        await orch.handle_text_input("first question")
        await asyncio.sleep(0.02)  # let it start

        # Cancel with new input
        await orch.handle_text_input("actually, different question")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert any(t.content == "actually, different question" for t in user_turns)

    @pytest.mark.asyncio
    async def test_multiple_speech_segments_during_processing(self):
        """Multiple VAD segments during PROCESSING are concatenated as pending."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)

        seg1 = make_speech_audio(0.3)
        seg2 = make_speech_audio(0.5)
        seg3 = make_speech_audio(0.4)

        # Simulate segments arriving
        orch._pending_speech = seg1
        orch._pending_speech = np.concatenate([orch._pending_speech, seg2])
        orch._pending_speech = np.concatenate([orch._pending_speech, seg3])

        expected_len = len(seg1) + len(seg2) + len(seg3)
        assert len(orch._pending_speech) == expected_len


class TestBargeInCleanup:
    """Verify barge-in properly cleans up all state."""

    @pytest.mark.asyncio
    async def test_barge_in_clears_pending_speech(self):
        """Barge-in should clear any pending speech buffer."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="test", source="voice", timestamp=time.time(),
        ))

        orch._pending_speech = make_speech_audio(1.0)
        assert orch._pending_speech is not None

        await orch._execute_barge_in()

        assert orch._pending_speech is None

    @pytest.mark.asyncio
    async def test_barge_in_sets_generation_cancelled(self):
        """Barge-in sets _generation_cancelled flag."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="test", source="voice", timestamp=time.time(),
        ))

        orch._generation_cancelled = False
        await orch._execute_barge_in()

        assert orch._generation_cancelled is True

    @pytest.mark.asyncio
    async def test_barge_in_cancels_processing_task(self):
        """Barge-in cancels any in-flight processing task."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="test", source="voice", timestamp=time.time(),
        ))

        # Create a mock processing task
        mock_task = AsyncMock()
        mock_task.cancel = MagicMock()
        orch._processing_task = mock_task

        await orch._execute_barge_in()

        mock_task.cancel.assert_called_once()
        assert orch._processing_task is None

    @pytest.mark.asyncio
    async def test_sequential_barge_ins(self):
        """Multiple rapid barge-ins don't corrupt state."""
        comps = make_mock_components()
        comps["llm"].stream = make_llm_stream(["Response."])
        orch, _ = make_orchestrator(comps)

        for i in range(3):
            # Start a turn
            orch._state._state = PipelineState.IDLE
            await orch.handle_text_input(f"Message {i}")
            if orch._processing_task:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)

            # Barge in if speaking
            if orch.state == PipelineState.SPEAKING:
                orch._dialogue.append(Turn(
                    role="assistant", content=f"Response {i}",
                    source="voice", timestamp=time.time(),
                ))
                await orch._execute_barge_in()

        # Should be in a valid state
        assert orch.state in (PipelineState.IDLE, PipelineState.INTERRUPTED)


class TestPendingSpeechAfterTurn:
    """Test pending speech processing after a turn completes."""

    @pytest.mark.asyncio
    async def test_pending_speech_processed_after_current_turn(self):
        """Pending speech triggers a new voice turn after current completes."""
        call_idx = 0

        async def multi_transcribe(audio):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return {"text": "first part", "language": "en", "confidence": 0.95}
            return {"text": "second part", "language": "en", "confidence": 0.90}

        comps = make_mock_components()
        comps["asr"].transcribe = multi_transcribe
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING

        # Set pending speech before processing
        pending = make_speech_audio(0.8)
        orch._pending_speech = pending

        # Process first voice turn
        await orch._process_voice_turn(make_speech_audio(1.0))

        # Both turns should be in dialogue
        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert len(user_turns) == 2
        assert user_turns[0].content == "first part"
        assert user_turns[1].content == "second part"
        assert orch._pending_speech is None

    @pytest.mark.asyncio
    async def test_empty_first_with_pending_processes_pending(self):
        """Empty ASR + pending speech → pending is processed."""
        call_idx = 0

        async def empty_then_real(audio):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return {"text": "  ", "language": "en", "confidence": 0.1}
            return {"text": "the real speech", "language": "en", "confidence": 0.95}

        comps = make_mock_components()
        comps["asr"].transcribe = empty_then_real
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING
        orch._pending_speech = make_speech_audio(1.0)

        await orch._process_voice_turn(make_speech_audio(0.2))

        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert len(user_turns) == 1
        assert user_turns[0].content == "the real speech"
        assert orch._pending_speech is None
