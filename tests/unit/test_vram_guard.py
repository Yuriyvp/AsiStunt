"""Tests for VRAM guard module."""
import asyncio
import pytest
from voice_assistant.core.vram_guard import VRAMGuard, PRESSURE_THRESHOLD_BYTES


def test_check_returns_state():
    guard = VRAMGuard(device_index=0)
    try:
        state = guard.check()
        assert state.total_bytes > 0
        assert state.free_bytes >= 0
        assert state.used_bytes >= 0
        assert state.total_bytes == state.free_bytes + state.used_bytes
        print(f"VRAM: {state.free_bytes / 1024**3:.1f} GB free / {state.total_bytes / 1024**3:.1f} GB total")
    finally:
        guard.shutdown()


def test_pressure_callback():
    guard = VRAMGuard(device_index=0)
    events = []
    guard.on_pressure(lambda s: events.append(s))
    state = guard.check()
    guard.shutdown()


@pytest.mark.asyncio
async def test_polling_starts_stops():
    guard = VRAMGuard(device_index=0)
    try:
        await guard.start_polling()
        assert guard._polling_task is not None
        await guard.stop_polling()
        assert guard._polling_task is None
    finally:
        guard.shutdown()
