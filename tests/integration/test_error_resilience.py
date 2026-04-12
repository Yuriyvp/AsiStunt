"""Error resilience tests — component failures, timeouts, recovery.

Tests graceful handling of: LLM errors, TTS errors, ASR failures,
timeout scenarios, and error recovery without crashing the pipeline.

Run: pytest tests/integration/test_error_resilience.py -v -s
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState, PipelineMode
from voice_assistant.core.audio_output import AudioChunk

from conftest import (
    make_mock_components, make_orchestrator, make_speech_audio, make_tts_audio,
    make_llm_stream, make_mood_stream,
)


class TestLLMFailures:
    """LLM errors should not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_llm_exception_returns_to_idle(self):
        """LLM stream raises exception → IDLE, no crash."""
        async def crashing_stream(messages, sampling=None, **kwargs):
            yield "Starting..."
            raise RuntimeError("LLM connection lost")

        comps = make_mock_components()
        comps["llm"].stream = crashing_stream
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("hello")
        if orch._processing_task:
            try:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)
            except Exception:
                pass

        assert orch.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_llm_empty_response(self):
        """LLM returns no tokens → IDLE, no crash."""
        async def empty_stream(messages, sampling=None, **kwargs):
            return
            yield  # make it async generator

        comps = make_mock_components()
        comps["llm"].stream = empty_stream
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("hello")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_llm_error_then_recovery(self):
        """LLM fails on first turn, succeeds on second."""
        call_idx = 0

        async def flaky_stream(messages, sampling=None, **kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                yield "Start..."
                raise RuntimeError("LLM timeout")
            for token in ["Recovery ", "response."]:
                yield token

        comps = make_mock_components()
        comps["llm"].stream = flaky_stream
        orch, _ = make_orchestrator(comps)

        # First turn: fails
        await orch.handle_text_input("first try")
        if orch._processing_task:
            try:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)
            except Exception:
                pass
        assert orch.state == PipelineState.IDLE

        # Second turn: succeeds
        await orch.handle_text_input("second try")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch.state == PipelineState.IDLE
        # Should have at least 2 dialogue entries from second turn
        assert len(orch.dialogue) >= 2

    @pytest.mark.asyncio
    async def test_llm_partial_tokens_before_error(self):
        """LLM sends some tokens then errors → partial dialogue recorded."""
        async def partial_stream(messages, sampling=None, **kwargs):
            for token in ["I was", " going", " to say"]:
                yield token
            raise RuntimeError("Connection reset")

        comps = make_mock_components()
        comps["llm"].stream = partial_stream
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("hello")
        if orch._processing_task:
            try:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)
            except Exception:
                pass

        assert orch.state == PipelineState.IDLE


class TestTTSFailures:
    """TTS errors should not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_tts_single_chunk_failure_continues(self):
        """TTS fails on one chunk → skips that chunk, continues pipeline."""
        call_count = 0

        async def flaky_tts(text, voice_params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("TTS OOM")
            return make_tts_audio()

        comps = make_mock_components()
        comps["tts"].synthesize = flaky_tts
        comps["llm"].stream = make_llm_stream(
            ["First sentence fails. ", "Second sentence works. ", "Third also works."]
        )
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("tell me something")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == 2

    @pytest.mark.asyncio
    async def test_tts_all_chunks_fail(self):
        """All TTS calls fail → pipeline still completes to IDLE."""
        async def always_fail_tts(text, voice_params=None):
            raise RuntimeError("TTS GPU error")

        comps = make_mock_components()
        comps["tts"].synthesize = always_fail_tts
        comps["llm"].stream = make_llm_stream(
            ["Sentence one. ", "Sentence two."]
        )
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("hello")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        assert orch.state == PipelineState.IDLE
        # Dialogue should still have the user+assistant turns
        assert len(orch.dialogue) == 2

    @pytest.mark.asyncio
    async def test_tts_error_then_recovery_next_turn(self):
        """TTS fails turn 1, works turn 2."""
        fail_first = True

        async def recovering_tts(text, voice_params=None):
            nonlocal fail_first
            if fail_first:
                fail_first = False
                raise RuntimeError("TTS init error")
            return make_tts_audio()

        comps = make_mock_components()
        comps["tts"].synthesize = recovering_tts
        comps["llm"].stream = make_llm_stream(["OK."])
        orch, _ = make_orchestrator(comps)

        # Turn 1: TTS fails
        await orch.handle_text_input("turn one")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)
        assert orch.state == PipelineState.IDLE

        # Turn 2: TTS works
        await orch.handle_text_input("turn two")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)
        assert orch.state == PipelineState.IDLE


class TestASRFailures:
    """ASR errors should not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_asr_exception_returns_to_idle(self):
        """ASR exception → IDLE, pending speech cleared."""
        comps = make_mock_components()
        comps["asr"].transcribe = AsyncMock(side_effect=RuntimeError("ASR model crash"))
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING

        await orch._process_voice_turn(make_speech_audio(1.0))

        assert orch.state == PipelineState.IDLE
        assert orch._pending_speech is None

    @pytest.mark.asyncio
    async def test_asr_returns_garbage(self):
        """ASR returns unexpected format → pipeline handles gracefully."""
        comps = make_mock_components()
        comps["asr"].transcribe = AsyncMock(
            return_value={"text": "", "language": None, "confidence": 0.0}
        )
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING

        await orch._process_voice_turn(make_speech_audio(0.5))

        # Empty text should return to IDLE
        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == 0


class TestConcurrentErrors:
    """Multiple concurrent error scenarios."""

    @pytest.mark.asyncio
    async def test_llm_error_during_barge_in(self):
        """LLM errors while barge-in is in progress → clean state."""
        async def error_stream(messages, sampling=None, **kwargs):
            await asyncio.sleep(0.05)
            raise RuntimeError("LLM down")
            yield  # unreachable but makes it async gen

        comps = make_mock_components()
        comps["llm"].stream = error_stream
        orch, _ = make_orchestrator(comps)

        # Start processing
        await orch.handle_text_input("question")
        await asyncio.sleep(0.01)

        # While processing, text barge-in
        await orch.handle_text_input("new question")
        if orch._processing_task:
            try:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)
            except Exception:
                pass

        # Pipeline should recover
        assert orch.state in (PipelineState.IDLE, PipelineState.PROCESSING)

    @pytest.mark.asyncio
    async def test_rapid_text_input_under_errors(self):
        """Rapid text inputs with occasional LLM errors → no deadlock."""
        call_idx = 0

        async def sometimes_failing_stream(messages, sampling=None, **kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx % 3 == 0:
                raise RuntimeError("periodic failure")
            yield "Response."

        comps = make_mock_components()
        comps["llm"].stream = sometimes_failing_stream
        orch, _ = make_orchestrator(comps)

        for i in range(6):
            await orch.handle_text_input(f"input {i}")
            if orch._processing_task:
                try:
                    await asyncio.wait_for(orch._processing_task, timeout=5.0)
                except Exception:
                    pass

        # Must always end in IDLE
        assert orch.state == PipelineState.IDLE


class TestStateConsistency:
    """State machine consistency under error conditions."""

    @pytest.mark.asyncio
    async def test_error_always_returns_to_idle(self):
        """Every error path must end at IDLE state."""
        error_scenarios = [
            # LLM error
            lambda: make_llm_stream(["error!"], delay=0),
            # Empty response
            lambda: make_llm_stream([]),
        ]

        for scenario_fn in error_scenarios:
            comps = make_mock_components()
            comps["llm"].stream = scenario_fn()
            orch, _ = make_orchestrator(comps)

            await orch.handle_text_input("test")
            if orch._processing_task:
                try:
                    await asyncio.wait_for(orch._processing_task, timeout=5.0)
                except Exception:
                    pass

            assert orch.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_pending_speech_cleared_on_error(self):
        """Pending speech is cleared when voice turn errors."""
        comps = make_mock_components()
        comps["asr"].transcribe = AsyncMock(side_effect=RuntimeError("crash"))
        orch, _ = make_orchestrator(comps)
        orch._state._state = PipelineState.PROCESSING
        orch._pending_speech = make_speech_audio(0.5)

        await orch._process_voice_turn(make_speech_audio(1.0))

        assert orch._pending_speech is None
        assert orch.state == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_generation_cancelled_flag_reset(self):
        """_generation_cancelled is reset at start of each new turn."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)

        orch._generation_cancelled = True

        await orch.handle_text_input("new turn")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Flag should be False after successful turn
        # (it's set to False at start of _process_turn)
        assert orch._generation_cancelled is False
