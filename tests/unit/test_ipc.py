"""Tests for IPC layer — StdoutEmitter and StdinReader."""
import asyncio
import json
import io
import time

import pytest

from voice_assistant.core.ipc import StdoutEmitter, StdinReader, MAX_EMIT_QUEUE, THROTTLE_INTERVALS
from voice_assistant.debug.signal_types import SignalType


class TestSignalTypes:
    def test_all_signal_types_are_strings(self):
        for st in SignalType:
            assert isinstance(st.value, str)

    def test_signal_count(self):
        assert len(SignalType) >= 20

    def test_specific_signals_exist(self):
        assert SignalType.SPEECH_START.value == "speech_start"
        assert SignalType.STATE_CHANGE.value == "state_change"
        assert SignalType.ERROR.value == "error"
        assert SignalType.BARGE_IN.value == "barge_in"
        assert SignalType.VRAM_USAGE.value == "vram_usage"


class TestStdoutEmitter:
    @pytest.fixture
    def buf_emitter(self):
        """Create an emitter that writes to a StringIO buffer."""
        buf = io.StringIO()
        emitter = StdoutEmitter(output_stream=buf)
        return emitter, buf

    @pytest.mark.asyncio
    async def test_emit_state_change(self, buf_emitter):
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_state_change("LISTENING")
        await asyncio.sleep(0.05)
        await emitter.stop()

        data = json.loads(buf.getvalue().strip())
        assert data["event"] == "state_change"
        assert data["state"] == "LISTENING"

    @pytest.mark.asyncio
    async def test_emit_transcript(self, buf_emitter):
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_transcript("Hello world", "en", 0.95)
        await asyncio.sleep(0.05)
        await emitter.stop()

        data = json.loads(buf.getvalue().strip())
        assert data["event"] == "transcript"
        assert data["text"] == "Hello world"
        assert data["lang"] == "en"
        assert data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_emit_token(self, buf_emitter):
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_token("Hello")
        await asyncio.sleep(0.05)
        await emitter.stop()

        data = json.loads(buf.getvalue().strip())
        assert data["event"] == "token"
        assert data["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_emit_error(self, buf_emitter):
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_error("llm", "Connection refused", consecutive=2)
        await asyncio.sleep(0.05)
        await emitter.stop()

        data = json.loads(buf.getvalue().strip())
        assert data["event"] == "error"
        assert data["source"] == "llm"
        assert data["message"] == "Connection refused"
        assert data["consecutive_failures"] == 2

    @pytest.mark.asyncio
    async def test_emit_signal(self, buf_emitter):
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_signal("speech_start", timestamp=1234.5)
        await asyncio.sleep(0.05)
        await emitter.stop()

        data = json.loads(buf.getvalue().strip())
        assert data["event"] == "signal"
        assert data["type"] == "speech_start"
        assert data["timestamp"] == 1234.5

    @pytest.mark.asyncio
    async def test_multiple_events_ordered(self, buf_emitter):
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_state_change("IDLE")
        emitter.emit_state_change("LISTENING")
        emitter.emit_state_change("PROCESSING")
        await asyncio.sleep(0.05)
        await emitter.stop()

        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 3
        states = [json.loads(l)["state"] for l in lines]
        assert states == ["IDLE", "LISTENING", "PROCESSING"]

    @pytest.mark.asyncio
    async def test_throttling(self, buf_emitter):
        """High-frequency signals should be throttled."""
        emitter, buf = buf_emitter
        await emitter.start()

        # Emit 50 audio.input_level signals rapidly
        for i in range(50):
            emitter.emit_signal("audio.input_level", value=i / 50)

        await asyncio.sleep(0.05)
        await emitter.stop()

        lines = buf.getvalue().strip().split("\n")
        # Should be heavily throttled — only 1-2 should get through
        level_lines = [l for l in lines if l and "input_level" in l]
        assert len(level_lines) <= 3

    @pytest.mark.asyncio
    async def test_non_throttled_events_pass_through(self, buf_emitter):
        """Non-throttled events should all pass through."""
        emitter, buf = buf_emitter
        await emitter.start()

        for i in range(10):
            emitter.emit_state_change(f"STATE_{i}")

        await asyncio.sleep(0.05)
        await emitter.stop()

        lines = [l for l in buf.getvalue().strip().split("\n") if l]
        assert len(lines) == 10

    @pytest.mark.asyncio
    async def test_queue_overflow_drops_oldest(self):
        """When queue is full, oldest events should be dropped."""
        emitter = StdoutEmitter(output_stream=io.StringIO())
        # Don't start the write loop — let queue fill up
        for i in range(MAX_EMIT_QUEUE + 50):
            emitter.emit({"event": "test", "seq": i})

        # Queue should be at max capacity
        assert emitter._queue.qsize() <= MAX_EMIT_QUEUE

    @pytest.mark.asyncio
    async def test_emit_unicode(self, buf_emitter):
        """Unicode text should pass through correctly."""
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_transcript("Dobar dan, kako ste?", "hr", 0.9)
        await asyncio.sleep(0.05)
        await emitter.stop()

        data = json.loads(buf.getvalue().strip())
        assert data["text"] == "Dobar dan, kako ste?"

    @pytest.mark.asyncio
    async def test_newline_delimited(self, buf_emitter):
        """Each event should be on its own line."""
        emitter, buf = buf_emitter
        await emitter.start()
        emitter.emit_state_change("A")
        emitter.emit_state_change("B")
        await asyncio.sleep(0.05)
        await emitter.stop()

        # Each non-empty line should be valid JSON
        for line in buf.getvalue().strip().split("\n"):
            if line:
                json.loads(line)  # should not raise


class TestStdinReader:
    @pytest.mark.asyncio
    async def test_on_command_registers_handler(self):
        reader = StdinReader()
        handler_calls = []
        reader.on_command(lambda cmd: handler_calls.append(cmd))
        assert len(reader._handlers) == 1

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        reader = StdinReader()
        calls_a = []
        calls_b = []
        reader.on_command(lambda cmd: calls_a.append(cmd))
        reader.on_command(lambda cmd: calls_b.append(cmd))
        assert len(reader._handlers) == 2

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        """Stop should be safe to call even if never started."""
        reader = StdinReader()
        await reader.stop()  # should not raise
