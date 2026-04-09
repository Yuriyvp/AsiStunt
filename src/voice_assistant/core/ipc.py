"""IPC layer — stdin/stdout newline-delimited JSON.

StdinReader: reads commands from Tauri (async, line-by-line).
StdoutEmitter: writes events to Tauri (non-blocking, queue-based, drops on overflow).

stdout backpressure mitigation:
- Internal asyncio queue with drop-oldest on overflow
- High-frequency signals throttled (input_level: max 100ms interval)
- Debug-only signals suppressed when debug UI is closed
"""
import asyncio
import json
import logging
import sys
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

MAX_EMIT_QUEUE = 256
THROTTLE_INTERVALS = {
    "audio.input_level": 0.100,  # 100ms
    "vram_usage": 1.0,           # 1s
}


class StdinReader:
    """Reads newline-delimited JSON commands from stdin.

    Usage:
        reader = StdinReader()
        reader.on_command(handler_fn)
        await reader.start()
    """

    def __init__(self):
        self._handlers: list[Callable[[dict], None]] = []
        self._task: asyncio.Task | None = None

    def on_command(self, handler: Callable[[dict], None]) -> None:
        self._handlers.append(handler)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break  # stdin closed
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON on stdin: %s", text[:100])
                    continue

                for handler in self._handlers:
                    try:
                        handler(cmd)
                    except Exception:
                        logger.exception("Error handling command: %s", cmd.get("cmd"))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("stdin read error")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


class StdoutEmitter:
    """Writes newline-delimited JSON events to stdout.

    Non-blocking: uses an asyncio queue. Drops oldest on overflow.
    Throttles high-frequency signals.
    """

    def __init__(self, debug_enabled: bool = False, output_stream=None):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_EMIT_QUEUE)
        self._task: asyncio.Task | None = None
        self._debug_enabled = debug_enabled
        self._last_emit_times: dict[str, float] = {}
        self._output_stream = output_stream

    def emit(self, event: dict) -> None:
        """Queue an event for emission. Non-blocking, drops oldest on overflow."""
        # Throttle check
        signal_type = event.get("type", event.get("event", ""))
        interval = THROTTLE_INTERVALS.get(signal_type)
        if interval:
            now = time.monotonic()
            last = self._last_emit_times.get(signal_type, 0)
            if now - last < interval:
                return
            self._last_emit_times[signal_type] = now

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def emit_state_change(self, state: str) -> None:
        self.emit({"event": "state_change", "state": state})

    def emit_transcript(self, text: str, lang: str, confidence: float) -> None:
        self.emit({"event": "transcript", "text": text, "lang": lang, "confidence": confidence})

    def emit_token(self, text: str) -> None:
        self.emit({"event": "token", "text": text})

    def emit_error(self, source: str, message: str, consecutive: int = 1) -> None:
        self.emit({"event": "error", "source": source, "message": message,
                    "consecutive_failures": consecutive})

    def emit_signal(self, signal_type: str, **data) -> None:
        self.emit({"event": "signal", "type": signal_type, **data})

    async def start(self) -> None:
        self._task = asyncio.create_task(self._write_loop())

    async def _write_loop(self) -> None:
        while True:
            try:
                event = await self._queue.get()
                line = json.dumps(event, ensure_ascii=False) + "\n"
                out = self._output_stream or sys.stdout
                out.write(line)
                out.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("stdout write error")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
