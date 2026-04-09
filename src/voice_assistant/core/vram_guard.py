"""VRAM guard — polls GPU memory, detects pressure, emits signals.

This MUST be the first infrastructure module. All model-loading code
checks the VRAM guard before allocating GPU memory.

Threshold: 0.5 GB free — pressure detected.
Actions on pressure: suggest KV cache quantization (q4_0), ctx reduction to 6144.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

PRESSURE_THRESHOLD_BYTES = 512 * 1024 * 1024  # 0.5 GB

POLL_INTERVAL_IDLE = 5.0
POLL_INTERVAL_ACTIVE = 1.0


@dataclass
class VRAMState:
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    under_pressure: bool = False
    pressure_mitigations: list[str] = field(default_factory=list)


class VRAMGuard:
    """Monitors GPU VRAM and signals when free memory is critically low.

    Usage:
        guard = VRAMGuard(device_index=0)
        guard.on_pressure(callback)
        state = guard.check()
        await guard.start_polling()
    """

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._pressure_callbacks: list[Callable[[VRAMState], None]] = []
        self._polling_task: asyncio.Task | None = None
        self._was_under_pressure = False
        self._nvml_initialized = False

    def _ensure_nvml(self) -> None:
        if self._nvml_initialized:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as e:
            logger.error("Failed to initialize NVML: %s. VRAM guard disabled.", e)
            raise RuntimeError(f"NVML init failed: {e}") from e

    def check(self) -> VRAMState:
        """One-shot VRAM check. Call BEFORE any heavy GPU operation."""
        self._ensure_nvml()
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        state = VRAMState(
            total_bytes=mem_info.total,
            used_bytes=mem_info.used,
            free_bytes=mem_info.free,
        )

        state.under_pressure = state.free_bytes < PRESSURE_THRESHOLD_BYTES

        if state.under_pressure:
            state.pressure_mitigations = [
                "cache-type-k=q4_0",
                "cache-type-v=q4_0",
                "ctx-size=6144",
            ]
            logger.warning(
                "VRAM pressure detected: %.1f MB free (threshold: %.1f MB). "
                "Mitigations: %s",
                state.free_bytes / 1024 / 1024,
                PRESSURE_THRESHOLD_BYTES / 1024 / 1024,
                state.pressure_mitigations,
            )

        return state

    def on_pressure(self, callback: Callable[[VRAMState], None]) -> None:
        """Register a callback for pressure events."""
        self._pressure_callbacks.append(callback)

    def _notify(self, state: VRAMState) -> None:
        for cb in self._pressure_callbacks:
            try:
                cb(state)
            except Exception:
                logger.exception("Error in VRAM pressure callback")

    async def start_polling(self) -> None:
        if self._polling_task is not None:
            return
        self._polling_task = asyncio.create_task(self._poll_loop())
        logger.info("VRAM guard polling started (idle=%.1fs, active=%.1fs)",
                     POLL_INTERVAL_IDLE, POLL_INTERVAL_ACTIVE)

    async def stop_polling(self) -> None:
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None

    async def _poll_loop(self) -> None:
        while True:
            try:
                state = self.check()
                if state.under_pressure and not self._was_under_pressure:
                    self._notify(state)
                elif not state.under_pressure and self._was_under_pressure:
                    self._notify(state)
                self._was_under_pressure = state.under_pressure
                interval = POLL_INTERVAL_ACTIVE if state.under_pressure else POLL_INTERVAL_IDLE
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("VRAM guard poll error")
                await asyncio.sleep(POLL_INTERVAL_IDLE)

    def shutdown(self) -> None:
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
