"""Sentence chunker — splits LLM text stream into TTS-ready chunks.

First chunk: aggressive — 15+ chars, fires on clause boundary (, ; — :).
Subsequent chunks: 50–150 chars, fires on sentence boundary (.!?…).
Safety valve: force flush at 150 chars on nearest word boundary.

Abbreviation handling: regex skip list (Mr., Dr., vs., etc.).
"""
import logging
import re

logger = logging.getLogger(__name__)

ABBREVIATIONS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|dept|est|govt|inc|ltd|vol|no|fig)\.$",
    re.IGNORECASE,
)

CLAUSE_BOUNDARIES = re.compile(r"[,;:\u2014\u2013]")
SENTENCE_BOUNDARIES = re.compile(r"[.!?\u2026]")


class SentenceChunker:
    """Accumulates tokens and emits speakable chunks.

    Usage:
        chunker = SentenceChunker()
        for token in cleaned_stream:
            chunk = chunker.feed(token)
            if chunk:
                tts_queue.put(chunk)
        final = chunker.flush()
        if final:
            tts_queue.put(final)
    """

    def __init__(self, first_chunk_min: int = 15, normal_chunk_min: int = 50,
                 max_chunk: int = 150):
        self._buffer = ""
        self._chunk_count = 0
        self._first_min = first_chunk_min
        self._normal_min = normal_chunk_min
        self._max = max_chunk

    def feed(self, text: str) -> str | None:
        """Feed text (possibly partial token). Returns a chunk if boundary hit, else None."""
        self._buffer += text

        if not self._buffer.strip():
            return None

        min_len = self._first_min if self._chunk_count == 0 else self._normal_min

        if len(self._buffer) >= self._max:
            return self._force_flush()

        if len(self._buffer) < min_len:
            return None

        if self._chunk_count == 0:
            return self._try_split(CLAUSE_BOUNDARIES) or self._try_split(SENTENCE_BOUNDARIES)
        else:
            return self._try_split(SENTENCE_BOUNDARIES)

    def _try_split(self, pattern: re.Pattern) -> str | None:
        """Try to split at the last matching boundary in the buffer."""
        matches = list(pattern.finditer(self._buffer))
        if not matches:
            return None

        match = matches[-1]
        pos = match.end()

        candidate = self._buffer[:pos]
        if ABBREVIATIONS.search(candidate):
            for m in reversed(matches[:-1]):
                candidate = self._buffer[:m.end()]
                if not ABBREVIATIONS.search(candidate):
                    pos = m.end()
                    break
            else:
                return None

        chunk = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos:].lstrip()
        if chunk:
            self._chunk_count += 1
            return chunk
        return None

    def _force_flush(self) -> str:
        """Force flush at nearest word boundary before max."""
        idx = self._buffer.rfind(" ", 0, self._max)
        if idx == -1:
            idx = self._max
        chunk = self._buffer[:idx].strip()
        self._buffer = self._buffer[idx:].lstrip()
        self._chunk_count += 1
        return chunk

    def flush(self) -> str | None:
        """Flush remaining buffer. Call after stream ends."""
        if self._buffer.strip():
            chunk = self._buffer.strip()
            self._buffer = ""
            self._chunk_count += 1
            return chunk
        self._buffer = ""
        return None

    def reset(self) -> None:
        self._buffer = ""
        self._chunk_count = 0
