"""Tests for ProcessManager — startup sequencing, error escalation, degraded modes."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_assistant.core.state_machine import PipelineMode
from voice_assistant.process.manager import ProcessManager, STABILITY_WINDOW_S


class MockSoulConfig:
    name = "TestBot"
    personality = "You are a test bot."
    backstory = "Created for testing."


class MockSettings:
    llm_model = "/nonexistent/models/test.gguf"
    llm_port = 8080
    llm_ctx_size = 8192
    llm_gpu_layers = 999
    llm_threads = 4
    llm_batch_size = 512
    llm_flash_attn = True
    voice_languages = []
    default_language = "en"
    sampling = {}


class MockVRAMGuard:
    def __init__(self, free_gb=20.0, total_gb=24.0, under_pressure=False):
        self._free = int(free_gb * 1024**3)
        self._total = int(total_gb * 1024**3)
        self._pressure = under_pressure

    def check(self):
        state = MagicMock()
        state.free_bytes = self._free
        state.total_bytes = self._total
        state.used_bytes = self._total - self._free
        state.under_pressure = self._pressure
        return state


class MockEmitter:
    def __init__(self):
        self.events = []

    def emit_signal(self, signal_type, **data):
        self.events.append({"type": signal_type, **data})

    def emit_error(self, source, message, consecutive=1):
        self.events.append({"event": "error", "source": source, "message": message,
                            "consecutive": consecutive})

    def emit_state_change(self, state):
        self.events.append({"event": "state_change", "state": state})


class TestProcessManager:
    def test_initial_mode_disabled(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm.mode == PipelineMode.DISABLED

    def test_determine_mode_full(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm._determine_mode(llm_ok=True, tts_ok=True) == PipelineMode.FULL

    def test_determine_mode_text_only(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm._determine_mode(llm_ok=True, tts_ok=False) == PipelineMode.TEXT_ONLY

    def test_determine_mode_transcribe(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm._determine_mode(llm_ok=False, tts_ok=True) == PipelineMode.TRANSCRIBE

    def test_determine_mode_disabled(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm._determine_mode(llm_ok=False, tts_ok=False) == PipelineMode.DISABLED


class TestErrorEscalation:
    def test_first_failure_returns_1(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm.record_failure("llm") == 1

    def test_consecutive_failures_increment(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm.record_failure("llm") == 1
        assert pm.record_failure("llm") == 2
        assert pm.record_failure("llm") == 3

    def test_independent_component_counters(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        assert pm.record_failure("llm") == 1
        assert pm.record_failure("tts") == 1
        assert pm.record_failure("llm") == 2
        assert pm.record_failure("tts") == 2

    def test_counter_resets_after_stability_window(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        pm.record_failure("llm")
        pm.record_failure("llm")
        assert pm._failure_counts["llm"] == 2

        # Simulate time passing beyond stability window
        pm._failure_times["llm"] = time.monotonic() - STABILITY_WINDOW_S - 1
        assert pm.record_failure("llm") == 1  # reset

    def test_counter_does_not_reset_within_window(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        pm.record_failure("llm")
        # Time is within window — no reset
        assert pm.record_failure("llm") == 2

    @pytest.mark.asyncio
    async def test_handle_failure_emits_error(self):
        emitter = MockEmitter()
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), emitter, settings=MockSettings())
        pm._mode = PipelineMode.FULL

        await pm.handle_failure("tts")

        error_events = [e for e in emitter.events if e.get("event") == "error"]
        assert len(error_events) >= 1
        assert error_events[0]["source"] == "tts"

    @pytest.mark.asyncio
    async def test_handle_failure_third_time_degrades(self):
        emitter = MockEmitter()
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), emitter, settings=MockSettings())
        pm._mode = PipelineMode.FULL

        # Simulate 3 failures
        await pm.handle_failure("tts")
        await pm.handle_failure("tts")
        mode = await pm.handle_failure("tts")

        # After 3 failures, TTS is down — mode should degrade
        assert mode != PipelineMode.FULL


class TestStartupSequence:
    @pytest.mark.asyncio
    async def test_startup_emits_events(self):
        """Startup should emit process_state_change events."""
        emitter = MockEmitter()
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), emitter, settings=MockSettings())

        # Startup will fail (no real llama-server) but should emit events
        mode = await pm.startup()

        state_events = [e for e in emitter.events if e.get("type") == "process_state_change"]
        assert len(state_events) >= 1
        # First event should be system starting
        assert state_events[0]["component"] == "system"

    @pytest.mark.asyncio
    async def test_startup_without_llm_returns_degraded(self):
        """If LLM fails to start, mode should be degraded."""
        emitter = MockEmitter()
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), emitter, settings=MockSettings())
        mode = await pm.startup()

        # No real binary — LLM will fail
        assert mode in (PipelineMode.TRANSCRIBE, PipelineMode.DISABLED, PipelineMode.TEXT_ONLY)

    @pytest.mark.asyncio
    async def test_startup_voice_clone_skipped_for_description(self):
        """Voice clone step should be skipped when method is 'description'."""
        emitter = MockEmitter()
        soul = MockSoulConfig()
        soul.voice_method = "description"
        pm = ProcessManager(soul, MockVRAMGuard(), emitter, settings=MockSettings())
        await pm.startup()

        clone_events = [e for e in emitter.events if e.get("type") == "voice_clone_progress"]
        assert len(clone_events) == 0

    @pytest.mark.asyncio
    async def test_shutdown_resets_mode(self):
        emitter = MockEmitter()
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), emitter, settings=MockSettings())
        pm._mode = PipelineMode.FULL
        await pm.shutdown()
        assert pm.mode == PipelineMode.DISABLED

    @pytest.mark.asyncio
    async def test_shutdown_safe_when_nothing_started(self):
        pm = ProcessManager(MockSoulConfig(), MockVRAMGuard(), MockEmitter(), settings=MockSettings())
        await pm.shutdown()  # should not raise


class TestMainEntryPoint:
    def test_main_module_importable(self):
        """main.py should be importable."""
        from voice_assistant.main import main
        assert asyncio.iscoroutinefunction(main)
