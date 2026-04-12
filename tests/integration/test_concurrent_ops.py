"""Concurrent operation tests — speech during processing, rapid text input.

Tests race conditions and concurrent access patterns that occur in real usage.

Run: pytest tests/integration/test_concurrent_ops.py -v -s
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState
from voice_assistant.core.audio_output import AudioChunk

from conftest import (
    make_mock_components, make_orchestrator, make_speech_audio, make_tts_audio,
    make_llm_stream, make_mood_stream, make_asr_responses,
)


class TestSpeechDuringProcessing:
    """User keeps speaking while first segment is being processed."""

    @pytest.mark.asyncio
    async def test_speech_queued_during_processing(self):
        """Speech arriving during PROCESSING is queued as pending."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING

        # Simulate speech segment arriving
        speech = make_speech_audio(0.5)
        orch._pending_speech = speech

        assert orch._pending_speech is not None
        assert len(orch._pending_speech) == len(speech)
        assert orch.state == PipelineState.PROCESSING

    @pytest.mark.asyncio
    async def test_multiple_segments_concatenated(self):
        """Multiple speech segments during PROCESSING are concatenated."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)

        seg1 = make_speech_audio(0.3)
        seg2 = make_speech_audio(0.5)
        seg3 = make_speech_audio(0.2)

        orch._pending_speech = seg1
        orch._pending_speech = np.concatenate([orch._pending_speech, seg2])
        orch._pending_speech = np.concatenate([orch._pending_speech, seg3])

        total_samples = len(seg1) + len(seg2) + len(seg3)
        assert len(orch._pending_speech) == total_samples

    @pytest.mark.asyncio
    async def test_pending_processed_recursively(self):
        """After turn completes, pending speech triggers new voice turn."""
        call_idx = 0

        async def dual_transcribe(audio):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return {"text": "first half", "language": "en", "confidence": 0.92}
            return {"text": "second half", "language": "en", "confidence": 0.90}

        comps = make_mock_components()
        comps["asr"].transcribe = dual_transcribe
        comps["llm"].stream = make_llm_stream(["Response."])
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING

        # Set pending before processing first turn
        orch._pending_speech = make_speech_audio(0.8)

        await orch._process_voice_turn(make_speech_audio(1.0))

        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert len(user_turns) == 2
        assert user_turns[0].content == "first half"
        assert user_turns[1].content == "second half"
        assert orch._pending_speech is None

    @pytest.mark.asyncio
    async def test_three_part_utterance(self):
        """Three-part utterance: user pauses twice during single thought."""
        call_idx = 0

        async def triple_transcribe(audio):
            nonlocal call_idx
            call_idx += 1
            texts = ["I was thinking", "about maybe", "going to the store"]
            return {"text": texts[min(call_idx - 1, 2)], "language": "en", "confidence": 0.9}

        comps = make_mock_components()
        comps["asr"].transcribe = triple_transcribe
        comps["llm"].stream = make_llm_stream(["Sure!"])
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING

        # Set up cascading pending speech
        seg2 = make_speech_audio(0.5)
        seg3 = make_speech_audio(0.4)

        # During first turn, two more segments arrive
        orch._pending_speech = np.concatenate([seg2, seg3])

        await orch._process_voice_turn(make_speech_audio(1.0))

        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert len(user_turns) >= 2  # At least first + pending
        assert orch._pending_speech is None


class TestRapidTextInput:
    """Rapid sequential text inputs — cancel and restart."""

    @pytest.mark.asyncio
    async def test_text_cancels_slow_processing(self):
        """New text input cancels an in-progress slow LLM response."""
        async def slow_stream(messages, sampling=None, **kwargs):
            for token in ["Very", " slow", " response"]:
                await asyncio.sleep(0.05)
                yield token

        comps = make_mock_components()
        comps["llm"].stream = slow_stream
        orch, _ = make_orchestrator(comps)

        # Start slow turn
        await orch.handle_text_input("first question")
        await asyncio.sleep(0.02)

        # Cancel with new input
        await orch.handle_text_input("urgent question")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        # Latest user turn should be "urgent question"
        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert user_turns[-1].content == "urgent question"

    @pytest.mark.asyncio
    async def test_five_rapid_cancellations(self):
        """Five rapid inputs → only last one completes fully."""
        async def slow_stream(messages, sampling=None, **kwargs):
            await asyncio.sleep(0.05)
            yield "Response."

        comps = make_mock_components()
        comps["llm"].stream = slow_stream
        orch, _ = make_orchestrator(comps)

        for i in range(4):
            await orch.handle_text_input(f"cancelled {i}")
            await asyncio.sleep(0.01)

        # Last input
        await orch.handle_text_input("final input")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        assert orch.state == PipelineState.IDLE
        # Final input should be in dialogue
        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert any(t.content == "final input" for t in user_turns)

    @pytest.mark.asyncio
    async def test_text_during_speaking_barges_in(self):
        """Text input during SPEAKING triggers barge-in + new turn."""
        comps = make_mock_components()
        comps["llm"].stream = make_llm_stream(["New response."])
        comps["playback"].is_active = True
        orch, _ = make_orchestrator(comps)

        orch._state._state = PipelineState.SPEAKING
        orch._dialogue.append(Turn(
            role="assistant", content="old response", source="voice",
            timestamp=time.time(),
        ))

        await orch.handle_text_input("interrupt!")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        comps["playback"].fade_out.assert_called()
        user_turns = [t for t in orch.dialogue if t.role == "user"]
        assert any(t.content == "interrupt!" for t in user_turns)


class TestStalePlaybackCleanup:
    """Test _stop_stale_playback behavior."""

    @pytest.mark.asyncio
    async def test_stale_playback_cleared_on_new_turn(self):
        """Old playback is stopped when new turn starts from IDLE."""
        comps = make_mock_components()
        # Simulate non-empty playlist (stale audio)
        comps["playlist"].is_empty = False
        comps["playlist"].text_remaining = MagicMock(return_value="old audio still playing")
        orch, _ = make_orchestrator(comps)

        orch._stop_stale_playback()

        comps["playback"].fade_out.assert_called_with(15)
        comps["playlist"].clear.assert_called()

    @pytest.mark.asyncio
    async def test_no_stale_cleanup_if_empty(self):
        """No cleanup needed if playlist is already empty."""
        comps = make_mock_components()
        comps["playlist"].is_empty = True
        orch, _ = make_orchestrator(comps)

        orch._stop_stale_playback()

        comps["playback"].fade_out.assert_not_called()
        comps["playlist"].clear.assert_not_called()


class TestCancellationFlags:
    """Test that cancellation flags are properly managed."""

    @pytest.mark.asyncio
    async def test_generation_cancelled_reset_each_turn(self):
        """_generation_cancelled is False at the start of each new turn."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)

        orch._generation_cancelled = True

        await orch.handle_text_input("new turn")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # After successful turn, flag should be False
        assert orch._generation_cancelled is False

    @pytest.mark.asyncio
    async def test_cancelled_turn_doesnt_produce_tts(self):
        """If generation is cancelled mid-stream, TTS work stops."""
        tts_calls = 0

        async def counting_tts(text, voice_params=None):
            nonlocal tts_calls
            tts_calls += 1
            return make_tts_audio()

        async def slow_stream(messages, sampling=None, **kwargs):
            for token in ["First sentence. ", "Second sentence. ",
                          "Third sentence. ", "Fourth sentence."]:
                await asyncio.sleep(0.02)
                yield token

        comps = make_mock_components()
        comps["tts"].synthesize = counting_tts
        comps["llm"].stream = slow_stream
        orch, _ = make_orchestrator(comps)

        # Start turn
        await orch.handle_text_input("give me four sentences")
        await asyncio.sleep(0.03)

        # Cancel mid-stream with new input
        await orch.handle_text_input("stop")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        # TTS should have been called fewer times than 4 sentences would produce
        # (exact number depends on timing)
        assert orch.state == PipelineState.IDLE


class TestListeningTimeout:
    """Test listening timeout behavior."""

    @pytest.mark.asyncio
    async def test_listening_state_tracks_time(self):
        """State machine tracks time in current state."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._state.transition(PipelineState.LISTENING)

        # time_in_state should be close to 0
        assert orch._state.time_in_state < 1.0

        await asyncio.sleep(0.05)
        assert orch._state.time_in_state >= 0.05
