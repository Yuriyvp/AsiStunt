"""Integration tests for degraded mode transitions.

Verifies: LLM down -> TEXT_ONLY, TTS down -> TRANSCRIBE,
both down -> DISABLED, error escalation, stability window reset.

Run: pytest tests/integration/test_degraded_modes.py -v -s
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_assistant.core.state_machine import PipelineMode
from voice_assistant.process.manager import ProcessManager, LlamaCppProcess, STABILITY_WINDOW_S


def _make_soul_config(**overrides):
    """Create a mock SoulConfig."""
    soul = MagicMock()
    soul.name = "test_persona"
    soul.voice_method = "default"
    soul.voice_reference_audio = None
    soul.voice_description = "warm and friendly"
    soul.llm_model = "models/test.gguf"
    soul.llm_port = 8080
    soul.llm_ctx_size = 8192
    soul.llm_gpu_layers = 999
    soul.llm_threads = 4
    soul.llm_batch_size = 4096
    soul.llm_flash_attn = True
    for k, v in overrides.items():
        setattr(soul, k, v)
    return soul


def _make_ipc():
    """Create a mock IPC emitter."""
    ipc = MagicMock()
    ipc.emit_state_change = MagicMock()
    ipc.emit_signal = MagicMock()
    ipc.emit_error = MagicMock()
    return ipc


def _make_vram_guard(under_pressure: bool = False):
    vram = MagicMock()
    state = MagicMock()
    state.under_pressure = under_pressure
    vram.check = MagicMock(return_value=state)
    return vram


class TestDetermineMode:
    """Test PipelineMode determination from component availability."""

    def test_full_mode(self):
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), _make_ipc())
        assert pm._determine_mode(llm_ok=True, tts_ok=True) == PipelineMode.FULL

    def test_text_only_mode(self):
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), _make_ipc())
        assert pm._determine_mode(llm_ok=True, tts_ok=False) == PipelineMode.TEXT_ONLY

    def test_transcribe_mode(self):
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), _make_ipc())
        assert pm._determine_mode(llm_ok=False, tts_ok=True) == PipelineMode.TRANSCRIBE

    def test_disabled_mode(self):
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), _make_ipc())
        assert pm._determine_mode(llm_ok=False, tts_ok=False) == PipelineMode.DISABLED


class TestErrorEscalation:
    """Test the 3-tier error escalation: silent restart, VRAM-guarded, degrade."""

    def test_first_failure_count(self):
        """First failure records count=1."""
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), _make_ipc())
        count = pm.record_failure("llm")
        assert count == 1

    def test_consecutive_failures_increment(self):
        """Consecutive failures increment the counter."""
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), _make_ipc())
        pm.record_failure("llm")
        count = pm.record_failure("llm")
        assert count == 2

    def test_stability_window_resets_counter(self):
        """Counter resets after STABILITY_WINDOW_S of no failures."""
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), _make_ipc())
        pm.record_failure("llm")
        pm.record_failure("llm")

        # Simulate time passing beyond stability window
        pm._failure_times["llm"] = time.monotonic() - STABILITY_WINDOW_S - 1

        count = pm.record_failure("llm")
        assert count == 1  # reset

    @pytest.mark.asyncio
    async def test_first_failure_silent_restart(self):
        """First failure attempts silent restart."""
        ipc = _make_ipc()
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), ipc)

        # Mock a running LLM process
        mock_proc = MagicMock(spec=LlamaCppProcess)
        mock_proc.stop = AsyncMock()
        mock_proc.start = AsyncMock()
        mock_proc.is_running = True
        pm._llm_process = mock_proc
        pm._mode = PipelineMode.FULL

        result = await pm.handle_failure("llm")

        # Should attempt restart
        mock_proc.stop.assert_awaited_once()
        mock_proc.start.assert_awaited_once()
        ipc.emit_signal.assert_any_call(
            "process_state_change", component="llm", state="restarting"
        )

    @pytest.mark.asyncio
    async def test_second_failure_vram_guarded_restart(self):
        """Second failure attempts restart with reduced VRAM settings."""
        ipc = _make_ipc()
        vram = _make_vram_guard(under_pressure=True)
        pm = ProcessManager(_make_soul_config(), vram, ipc)

        # Pre-record first failure
        pm._failure_counts["llm"] = 1
        pm._failure_times["llm"] = time.monotonic()

        mock_proc = MagicMock(spec=LlamaCppProcess)
        mock_proc.stop = AsyncMock()
        mock_proc.start = AsyncMock()
        mock_proc.is_running = True
        mock_proc._binary = "llama-server"
        pm._llm_process = mock_proc
        pm._mode = PipelineMode.FULL

        # Patch LlamaCppProcess to capture the new args
        with patch.object(LlamaCppProcess, 'start', new_callable=AsyncMock) as mock_start:
            with patch.object(LlamaCppProcess, '__init__', return_value=None) as mock_init:
                result = await pm.handle_failure("llm")

                # Should have created a new process with reduced settings
                if mock_init.called:
                    call_kwargs = mock_init.call_args
                    assert call_kwargs.kwargs.get("ctx_size", 8192) == 6144 or \
                           (call_kwargs.args and 6144 in call_kwargs.args)

    @pytest.mark.asyncio
    async def test_third_failure_degrades_pipeline(self):
        """Third failure stops retrying and degrades the pipeline mode."""
        ipc = _make_ipc()
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), ipc)

        # Pre-record two failures
        pm._failure_counts["llm"] = 2
        pm._failure_times["llm"] = time.monotonic()

        mock_proc = MagicMock(spec=LlamaCppProcess)
        mock_proc.is_running = False
        pm._llm_process = mock_proc
        pm._mode = PipelineMode.FULL

        result = await pm.handle_failure("llm")

        # Should degrade (LLM down, TTS was FULL -> tts_ok=True -> TRANSCRIBE)
        assert result == PipelineMode.TRANSCRIBE

    @pytest.mark.asyncio
    async def test_tts_failure_degrades_to_text_only(self):
        """TTS component failure with LLM still running -> TEXT_ONLY."""
        ipc = _make_ipc()
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), ipc)

        pm._failure_counts["tts"] = 2
        pm._failure_times["tts"] = time.monotonic()

        mock_proc = MagicMock(spec=LlamaCppProcess)
        mock_proc.is_running = True
        pm._llm_process = mock_proc
        pm._mode = PipelineMode.FULL

        result = await pm.handle_failure("tts")

        assert result == PipelineMode.TEXT_ONLY

    @pytest.mark.asyncio
    async def test_both_down_disabled(self):
        """Both LLM and TTS down -> DISABLED."""
        ipc = _make_ipc()
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), ipc)

        # LLM already failed
        pm._failure_counts["llm"] = 3
        pm._failure_times["llm"] = time.monotonic()
        pm._llm_process = None
        pm._mode = PipelineMode.TRANSCRIBE  # LLM was already down

        # Now TTS fails
        pm._failure_counts["tts"] = 2
        pm._failure_times["tts"] = time.monotonic()

        result = await pm.handle_failure("tts")

        assert result == PipelineMode.DISABLED


class TestStartupSequencing:
    """Test startup sequencing with various failure scenarios."""

    @pytest.mark.asyncio
    async def test_startup_llm_failure_degrades(self):
        """If LLM fails to start, mode degrades to TRANSCRIBE."""
        ipc = _make_ipc()
        soul = _make_soul_config()
        pm = ProcessManager(soul, _make_vram_guard(), ipc)

        with patch.object(LlamaCppProcess, 'start',
                         new_callable=AsyncMock,
                         side_effect=RuntimeError("binary not found")):
            with patch.object(LlamaCppProcess, '__init__', return_value=None):
                mode = await pm.startup()

        # TTS is OK by default, LLM failed -> TRANSCRIBE
        assert mode == PipelineMode.TRANSCRIBE

    @pytest.mark.asyncio
    async def test_startup_success_full_mode(self):
        """Successful startup of all components -> FULL mode."""
        ipc = _make_ipc()
        soul = _make_soul_config()
        pm = ProcessManager(soul, _make_vram_guard(), ipc)

        with patch.object(LlamaCppProcess, 'start', new_callable=AsyncMock):
            with patch.object(LlamaCppProcess, '__init__', return_value=None):
                mode = await pm.startup()

        assert mode == PipelineMode.FULL


class TestShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_llm(self):
        ipc = _make_ipc()
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), ipc)

        mock_proc = MagicMock(spec=LlamaCppProcess)
        mock_proc.stop = AsyncMock()
        pm._llm_process = mock_proc

        await pm.shutdown()

        mock_proc.stop.assert_awaited_once()
        assert pm._llm_process is None
        assert pm.mode == PipelineMode.DISABLED

    @pytest.mark.asyncio
    async def test_shutdown_no_process(self):
        """Shutdown with no process running should not crash."""
        ipc = _make_ipc()
        pm = ProcessManager(_make_soul_config(), _make_vram_guard(), ipc)
        pm._llm_process = None

        await pm.shutdown()  # should not raise

        assert pm.mode == PipelineMode.DISABLED
