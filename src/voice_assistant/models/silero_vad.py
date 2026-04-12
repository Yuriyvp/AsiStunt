"""Silero VAD via sherpa-onnx ONNX Runtime.

Concrete module (no ABC) — VAD is not swappable in this system.
Consumes 30ms chunks (480 samples at 16kHz) from AudioInput,
emits speech_start/speech_end events via callbacks.

Architecture constraints:
- Runs on CPU only, no VRAM impact
- sherpa-onnx Silero VAD window_size is 512 samples (32ms at 16kHz)
- We buffer incoming 480-sample chunks and feed 512-sample windows to the VAD
- Speech end requires ~300ms of silence (configurable)
"""
import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)


@dataclass
class VADEvent:
    type: str  # "speech_start" or "speech_end"
    timestamp: float
    duration_ms: float = 0.0  # duration of speech segment (speech_end only)


class SileroVAD:
    """Silero VAD wrapper using sherpa-onnx VoiceActivityDetector.

    The sherpa-onnx VAD requires fixed 512-sample windows. Our audio pipeline
    produces 480-sample chunks (30ms at 16kHz), so we maintain an internal
    buffer and feed complete windows to the VAD.

    Speech segments are detected internally by sherpa-onnx based on
    min_silence_duration and min_speech_duration. We track transitions
    via is_speech_detected() and emit VADEvent callbacks.
    """

    def __init__(
        self,
        model_path: str = "models/silero_vad.onnx",
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        min_speech_duration_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self._sample_rate = sample_rate
        self._callbacks: list[Callable[[VADEvent], None]] = []

        # State tracking
        self._is_speech = False
        self._speech_start_time: float = 0.0

        # Pre-allocated ring buffer for accumulating samples until we have a full window.
        # Avoids np.concatenate() on every 30ms chunk (was allocating ~23 times/sec).
        self._buffer = np.empty(480 + 512, dtype=np.float32)  # chunk + window
        self._buf_len = 0

        # sherpa-onnx VAD initialization
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model_path
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = min_silence_duration_ms / 1000.0
        config.silero_vad.min_speech_duration = min_speech_duration_ms / 1000.0
        config.sample_rate = sample_rate

        self._window_size = config.silero_vad.window_size  # 512

        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        logger.info(
            "Silero VAD loaded: threshold=%.2f, min_silence=%dms, min_speech=%dms, "
            "window_size=%d",
            threshold, min_silence_duration_ms, min_speech_duration_ms, self._window_size,
        )

    def on_event(self, callback: Callable[[VADEvent], None]) -> None:
        """Register callback for speech_start / speech_end events."""
        self._callbacks.append(callback)

    def _emit(self, event: VADEvent) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("Error in VAD callback")

    def process_chunk(self, chunk: np.ndarray) -> VADEvent | None:
        """Process a 30ms audio chunk (480 samples at 16kHz).

        Buffers samples internally and feeds complete 512-sample windows to
        the sherpa-onnx VAD. Returns a VADEvent if a speech transition
        occurred during this chunk, else None.
        """
        # Append chunk to pre-allocated buffer (no allocation)
        n = len(chunk)
        self._buffer[self._buf_len:self._buf_len + n] = chunk
        self._buf_len += n
        last_event = None

        # Process all complete windows in the buffer
        while self._buf_len >= self._window_size:
            # Feed window to VAD BEFORE shifting — window is a view into buffer
            self._vad.accept_waveform(self._buffer[:self._window_size])
            # Now shift remaining data to front
            remaining = self._buf_len - self._window_size
            if remaining > 0:
                self._buffer[:remaining] = self._buffer[self._window_size:self._buf_len]
            self._buf_len = remaining

            now_speech = self._vad.is_speech_detected()

            if now_speech and not self._is_speech:
                # Transition: silence → speech
                self._is_speech = True
                self._speech_start_time = time.monotonic()
                last_event = VADEvent(type="speech_start", timestamp=self._speech_start_time)
                self._emit(last_event)
                logger.debug("VAD: speech_start")

            elif not now_speech and self._is_speech:
                # Transition: speech → silence
                self._is_speech = False
                now = time.monotonic()
                duration = (now - self._speech_start_time) * 1000
                last_event = VADEvent(
                    type="speech_end",
                    timestamp=now,
                    duration_ms=duration,
                )
                self._emit(last_event)
                logger.debug("VAD: speech_end (%.0fms)", duration)

        return last_event

    def get_speech_samples(self) -> np.ndarray | None:
        """Retrieve buffered speech segment samples from the VAD.

        Call after speech_end to get the complete speech audio for ASR.
        Returns None if no segment is available.
        """
        if self._vad.empty():
            return None
        samples = self._vad.front.samples
        self._vad.pop()
        return np.array(samples, dtype=np.float32)

    def drain_speech_samples(self) -> np.ndarray | None:
        """Drain all buffered speech segments, concatenating them.

        Useful after speech_end to get the full utterance audio.
        """
        segments = []
        while not self._vad.empty():
            segments.append(np.array(self._vad.front.samples, dtype=np.float32))
            self._vad.pop()
        if not segments:
            return None
        return np.concatenate(segments)

    def clear_segments(self) -> int:
        """Discard stored speech segments without resetting detection state.

        Use during barge-in: clears orphaned segments but keeps tracking
        current speech (user is still speaking).
        """
        count = 0
        while not self._vad.empty():
            self._vad.pop()
            count += 1
        return count

    def reset(self) -> None:
        """Reset VAD state (e.g., after barge-in)."""
        self._is_speech = False
        self._buf_len = 0
        self._vad.reset()

    def warmup(self) -> None:
        """Run a dummy chunk to trigger ONNX Runtime first-call overhead."""
        dummy = np.zeros(self._window_size, dtype=np.float32)
        self._vad.accept_waveform(dummy)
        self._vad.reset()

    def flush(self) -> None:
        """Flush the VAD — forces end-of-speech for any remaining audio."""
        self._vad.flush()

    @property
    def is_speech(self) -> bool:
        return self._is_speech
