"""Audio input pipeline: Mic → RNNoise → Normalizer → CaptureRing.

All audio is 16kHz mono float32 numpy arrays.
sounddevice callback runs in a separate thread — it pushes frames into an asyncio queue.
The processing loop runs in the asyncio event loop.

RNNoise operates at 48kHz with 480-sample frames (10ms).
We capture at 16kHz, so we resample up to 48kHz before denoising and back to 16kHz after.
"""
import asyncio
import ctypes
import logging
from pathlib import Path

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Audio parameters — fixed for the entire pipeline
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 480 samples at 16kHz

# CaptureRing: 30 seconds of audio
RING_DURATION_S = 30
RING_SIZE = SAMPLE_RATE * RING_DURATION_S


class RNNoiseWrapper:
    """Wraps the RNNoise shared library via ctypes.

    RNNoise natively operates at 48kHz with 480-sample frames (10ms).
    Our pipeline is 16kHz, so we resample up to 48kHz, denoise, resample back.
    """

    # RNNoise frame size: 480 samples at 48kHz (10ms)
    RNNOISE_FRAME_SIZE = 480

    def __init__(self, lib_path: str = "/usr/local/lib/librnnoise.so"):
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load RNNoise from {lib_path}: {e}") from e

        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_process_frame.restype = ctypes.c_float
        self._lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

        self._state = self._lib.rnnoise_create(None)
        if not self._state:
            raise RuntimeError("Failed to create RNNoise state")

        # Pre-allocate resampling index arrays for 16kHz↔48kHz conversion
        self._upsample_indices = np.linspace(0, 159, self.RNNOISE_FRAME_SIZE)
        self._downsample_indices = np.linspace(0, self.RNNOISE_FRAME_SIZE - 1, 160)

        logger.info("RNNoise loaded from %s", lib_path)

    def process_frame_16khz(self, frame_160: np.ndarray) -> np.ndarray:
        """Process a 160-sample frame (10ms at 16kHz) through RNNoise.

        Resamples to 48kHz internally. Returns denoised 160 samples at 16kHz.
        """
        # Linear interpolation upsample 16k→48k (3x)
        frame_48k = np.interp(
            self._upsample_indices,
            np.arange(160),
            frame_160,
        ).astype(np.float32)

        # RNNoise expects samples in [-32768, 32767] range (int16 scale)
        frame_48k_scaled = frame_48k * 32767.0

        out_buf = (ctypes.c_float * self.RNNOISE_FRAME_SIZE)()
        in_buf = (ctypes.c_float * self.RNNOISE_FRAME_SIZE)(*frame_48k_scaled)

        self._lib.rnnoise_process_frame(self._state, out_buf, in_buf)

        # Convert back to [-1, 1] float32
        denoised_48k = np.array(out_buf, dtype=np.float32) / 32767.0

        # Downsample 48k→16k
        denoised_16k = np.interp(
            self._downsample_indices,
            np.arange(self.RNNOISE_FRAME_SIZE),
            denoised_48k,
        ).astype(np.float32)

        return denoised_16k

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a 30ms chunk (480 samples at 16kHz) — splits into 3 × 10ms sub-frames."""
        if len(chunk) != FRAME_SIZE:
            raise ValueError(f"Expected {FRAME_SIZE} samples, got {len(chunk)}")
        result = np.empty_like(chunk)
        for i in range(3):
            start = i * 160
            end = start + 160
            result[start:end] = self.process_frame_16khz(chunk[start:end])
        return result

    def destroy(self) -> None:
        if self._state:
            self._lib.rnnoise_destroy(self._state)
            self._state = None


class Normalizer:
    """RMS-based normalizer with fast attack, slow release.

    Targets -20 dBFS. DC offset removal via per-frame mean subtraction.
    Gain clamped to ±20dB (0.1 to 10.0 linear).
    """

    def __init__(self, target_rms_db: float = -20.0):
        self.target_rms = 10 ** (target_rms_db / 20.0)
        self.gain = 1.0
        self.attack = 0.01
        self.release = 0.001

    def process(self, frame: np.ndarray) -> np.ndarray:
        frame = frame - np.mean(frame)  # DC removal
        rms = np.sqrt(np.mean(frame ** 2)) + 1e-10
        desired_gain = np.clip(self.target_rms / rms, 0.1, 10.0)
        alpha = self.attack if desired_gain < self.gain else self.release
        self.gain = self.gain * (1 - alpha) + desired_gain * alpha
        return np.clip(frame * self.gain, -1.0, 1.0)


class CaptureRing:
    """30-second circular buffer of clean (denoised + normalized) audio.

    Thread-safe for single-writer (processing loop) + single-reader (VAD/ASR).
    """

    def __init__(self, capacity: int = RING_SIZE):
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._write_pos = 0
        self._capacity = capacity

    def write(self, chunk: np.ndarray) -> None:
        n = len(chunk)
        end = self._write_pos + n
        if end <= self._capacity:
            self._buffer[self._write_pos:end] = chunk
        else:
            first = self._capacity - self._write_pos
            self._buffer[self._write_pos:] = chunk[:first]
            self._buffer[:n - first] = chunk[first:]
        self._write_pos = end % self._capacity

    def read_last(self, num_samples: int) -> np.ndarray:
        """Read the last N samples from the ring (for ASR drain)."""
        if num_samples > self._capacity:
            num_samples = self._capacity
        start = (self._write_pos - num_samples) % self._capacity
        if start < self._write_pos:
            return self._buffer[start:self._write_pos].copy()
        else:
            return np.concatenate([
                self._buffer[start:],
                self._buffer[:self._write_pos],
            ])


class AudioInput:
    """Complete audio input pipeline.

    Manages: sounddevice InputStream → RNNoise → Normalizer → CaptureRing.
    Exposes an asyncio queue of clean 30ms chunks for downstream consumers (VAD).
    """

    def __init__(
        self,
        rnnoise_lib_path: str = "/usr/local/lib/librnnoise.so",
        device: int | str | None = None,
    ):
        self._device = device
        self._rnnoise = RNNoiseWrapper(rnnoise_lib_path)
        self._normalizer = Normalizer()
        self.ring = CaptureRing()
        self._chunk_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — runs in a separate thread."""
        if status:
            logger.warning("sounddevice status: %s", status)
        chunk = indata[:, 0].copy()
        try:
            self._loop.call_soon_threadsafe(self._chunk_queue.put_nowait, chunk)
        except asyncio.QueueFull:
            pass  # Drop frame rather than block the audio thread

    async def start(self) -> None:
        """Open the mic stream and start the processing loop."""
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SIZE,
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("Audio input started: %s Hz, %d ms frames", SAMPLE_RATE, FRAME_DURATION_MS)

    async def read_chunk(self) -> np.ndarray:
        """Get the next processed (denoised + normalized) 30ms chunk.

        Blocks until a chunk is available. Used by VAD.
        """
        raw = await self._chunk_queue.get()
        denoised = self._rnnoise.process_chunk(raw)
        normalized = self._normalizer.process(denoised)
        self.ring.write(normalized)
        return normalized

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._rnnoise.destroy()
        logger.info("Audio input stopped")
