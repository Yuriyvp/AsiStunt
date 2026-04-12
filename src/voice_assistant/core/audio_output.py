"""Audio output pipeline: Playlist -> PlaybackManager -> Speaker.

Playlist: ordered list of AudioChunks, mutable by orchestrator.
PlaybackManager: reads samples via callback, handles crossfade.
FillerCache: pre-rendered filler audio with rate-limiting.

Audio goes directly to sounddevice — never crosses IPC boundary.
Playback runs in a sounddevice callback thread.
Mutations only from the asyncio thread.
"""
import asyncio
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24000
CROSSFADE_SAMPLES = int(0.030 * OUTPUT_SAMPLE_RATE)  # 30ms crossfade


@dataclass
class AudioChunk:
    """A chunk of audio with associated text metadata."""
    audio: np.ndarray  # f32, OUTPUT_SAMPLE_RATE
    text: str = ""
    source: str = "tts"  # "tts" or "filler"
    played: bool = False
    position: int = 0  # playhead position within this chunk


class Playlist:
    """Mutable ordered list of AudioChunks. Orchestrator-owned.

    Operations: append, insert_at, drop_future, fade_out, clear.
    Tracking: text_played(), text_remaining(), wait_until_done().

    Thread safety: read_samples() is called from a sounddevice callback thread,
    while append/clear/wait_until_done are called from the asyncio thread.
    We use threading.Event (not asyncio.Event) for cross-thread signaling.
    """

    def __init__(self):
        self._chunks: list[AudioChunk] = []
        self._current_index: int = 0
        self._played_text_acc: str = ""  # accumulated text from freed chunks
        self._lock = threading.Lock()
        self._done_flag = threading.Event()
        self._done_flag.set()  # empty playlist starts as "done"

    def append(self, chunk: AudioChunk) -> None:
        with self._lock:
            self._chunks.append(chunk)
            self._done_flag.clear()

    def insert_at(self, index: int, chunk: AudioChunk) -> None:
        with self._lock:
            self._chunks.insert(index, chunk)
            self._done_flag.clear()

    def drop_future(self) -> list[AudioChunk]:
        """Drop all chunks after the currently playing one. Returns dropped chunks."""
        with self._lock:
            dropped = self._chunks[self._current_index + 1:]
            self._chunks = self._chunks[:self._current_index + 1]
            return dropped

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._current_index = 0
            self._played_text_acc = ""
            self._done_flag.set()

    def text_played(self) -> str:
        """Return concatenated text of all fully played chunks."""
        with self._lock:
            current = " ".join(c.text for c in self._chunks[:self._current_index] if c.text)
            if self._played_text_acc and current:
                return f"{self._played_text_acc} {current}"
            return self._played_text_acc or current

    def text_remaining(self) -> str:
        """Return concatenated text of unplayed/partially-played chunks."""
        with self._lock:
            remaining = self._chunks[self._current_index:]
            return " ".join(c.text for c in remaining if c.text)

    def read_samples(self, n: int) -> np.ndarray | None:
        """Read n samples for playback. Called from the playback thread.

        Returns None if playlist is exhausted.
        """
        with self._lock:
            if self._current_index >= len(self._chunks):
                self._done_flag.set()
                return None

            result = np.zeros(n, dtype=np.float32)
            filled = 0

            while filled < n and self._current_index < len(self._chunks):
                chunk = self._chunks[self._current_index]
                remaining_in_chunk = len(chunk.audio) - chunk.position
                to_copy = min(n - filled, remaining_in_chunk)

                result[filled:filled + to_copy] = chunk.audio[chunk.position:chunk.position + to_copy]
                chunk.position += to_copy
                filled += to_copy

                if chunk.position >= len(chunk.audio):
                    chunk.played = True
                    self._current_index += 1

            if filled == 0:
                self._done_flag.set()
                return None
            if filled < n:
                result = result[:filled]
            return result

    async def wait_until_done(self) -> None:
        """Wait until all chunks have been played.

        Uses polling because the done flag is set from the sounddevice callback
        thread, and asyncio.Event is not thread-safe.
        """
        while not self._done_flag.is_set():
            await asyncio.sleep(0.02)

    def free_played(self) -> None:
        """Free played chunks to prevent RSS growth. Call from asyncio thread only.

        Removes fully-played chunks from the list and adjusts the index.
        Safe to call between turns — never from the sounddevice callback.
        """
        with self._lock:
            if self._current_index == 0:
                return
            freed = self._chunks[:self._current_index]
            text_parts = [c.text for c in freed if c.text]
            if text_parts:
                freed_text = " ".join(text_parts)
                if self._played_text_acc:
                    self._played_text_acc += " " + freed_text
                else:
                    self._played_text_acc = freed_text
            del self._chunks[:self._current_index]
            self._current_index = 0

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._current_index >= len(self._chunks)


class PlaybackManager:
    """Reads from Playlist, outputs to speaker via sounddevice.

    Handles fade-out for barge-in.
    Runs a sounddevice OutputStream with a callback.
    """

    def __init__(self, playlist: Playlist, device: int | str | None = None):
        self._playlist = playlist
        self._device = device
        self._stream: sd.OutputStream | None = None
        self._active = False
        self._fade_remaining: int = 0

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("Playback status: %s", status)

        samples = self._playlist.read_samples(frames)

        if samples is None:
            outdata[:] = 0
            return

        # Apply fade-out if active (barge-in)
        if self._fade_remaining > 0:
            fade_len = min(len(samples), self._fade_remaining)
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            samples[:fade_len] *= fade
            if fade_len < len(samples):
                samples[fade_len:] = 0
            self._fade_remaining -= fade_len

        # Pad if needed
        if len(samples) < frames:
            outdata[:len(samples), 0] = samples
            outdata[len(samples):] = 0
        else:
            outdata[:, 0] = samples[:frames]

    async def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            device=self._device,
        )
        self._stream.start()
        self._active = True
        logger.info("Playback started at %d Hz", OUTPUT_SAMPLE_RATE)

    def fade_out(self, duration_ms: int = 30) -> None:
        """Initiate a fade-out (for barge-in). Non-blocking."""
        self._fade_remaining = int(OUTPUT_SAMPLE_RATE * duration_ms / 1000)

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active


class FillerCache:
    """Pre-rendered filler audio with rate-limiting.

    Fillers: 8-12 neutral phrases ("hmm", "uh", "let me think").
    Rate limits: max 1 filler per 3 turns, max 4 per minute.
    Recency tracking: don't repeat same filler within 5 uses.
    Conditional: only play if estimated latency > 450ms.
    """

    def __init__(self):
        self._fillers: list[tuple[str, np.ndarray]] = []  # (label, audio)
        self._recent: deque[int] = deque(maxlen=5)
        self._turn_count_since_filler: int = 0
        self._minute_timestamps: deque[float] = deque()
        self._latency_history: deque[float] = deque(maxlen=5)

    def load_from_dir(self, filler_dir: str) -> None:
        """Load pre-rendered filler f32 files from a directory."""
        from pathlib import Path
        d = Path(filler_dir)
        if not d.exists():
            logger.warning("Filler directory not found: %s", filler_dir)
            return
        for f in sorted(d.glob("*.f32")):
            audio = np.fromfile(f, dtype=np.float32)
            self._fillers.append((f.stem, audio))
        logger.info("Loaded %d fillers from %s", len(self._fillers), filler_dir)

    def add_filler(self, label: str, audio: np.ndarray) -> None:
        """Add a filler programmatically (e.g., from TTS pre-render)."""
        self._fillers.append((label, audio))

    def record_turn(self) -> None:
        """Call at the start of each conversational turn."""
        self._turn_count_since_filler += 1

    def record_latency(self, latency_ms: float) -> None:
        """Record pipeline latency for conditional playback decisions."""
        self._latency_history.append(latency_ms)

    def should_play(self) -> bool:
        """Decide whether to play a filler based on rate limits and latency."""
        if not self._fillers:
            return False

        # Check latency condition: estimated > 450ms
        if self._latency_history:
            avg_latency = sum(self._latency_history) / len(self._latency_history)
            if avg_latency <= 450:
                return False

        # Rate limit: max 1 per 3 turns
        if self._turn_count_since_filler < 3:
            return False

        # Rate limit: max 4 per minute
        now = time.monotonic()
        self._minute_timestamps = deque(
            t for t in self._minute_timestamps if now - t < 60
        )
        if len(self._minute_timestamps) >= 4:
            return False

        return True

    def get_filler(self) -> AudioChunk | None:
        """Get a filler AudioChunk, respecting recency. Returns None if shouldn't play."""
        if not self.should_play():
            return None

        # Pick a filler not in recent history
        available = [i for i in range(len(self._fillers)) if i not in self._recent]
        if not available:
            available = list(range(len(self._fillers)))

        idx = random.choice(available)
        label, audio = self._fillers[idx]

        self._recent.append(idx)
        self._turn_count_since_filler = 0
        self._minute_timestamps.append(time.monotonic())

        logger.debug("Filler selected: %s", label)
        return AudioChunk(audio=audio.copy(), text="", source="filler")
