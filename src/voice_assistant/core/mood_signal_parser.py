"""Mood signal parser — strips <mood_signal> tags from LLM token stream.

Sits between LLM output and sentence chunker. Intercepts mood tags silently.
If no tag after 50 tokens: assume neutral/0.5. Malformed tags: fallback neutral.

The parser must handle partial tokens — the tag may be split across multiple yields.
"""
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

MOOD_PATTERN = re.compile(
    r"<mood_signal>\s*user_tone\s*=\s*(\w+)\s*,\s*intensity\s*=\s*([\d.]+)\s*</mood_signal>"
)

TAG_OPEN = "<mood_signal>"


def _could_be_tag_prefix(text: str) -> bool:
    """Check if text ends with a possible prefix of '<mood_signal>'."""
    for i in range(1, len(TAG_OPEN) + 1):
        if text.endswith(TAG_OPEN[:i]):
            return True
    return False


class MoodSignalParser:
    """Filters mood_signal tags from a token stream.

    Usage:
        parser = MoodSignalParser(on_mood=callback)
        for token in llm_stream:
            clean = parser.feed(token)
            if clean:
                sentence_chunker.feed(clean)
        parser.finalize()  # flush any remaining buffer
    """

    def __init__(self, on_mood: Callable[[str, float], None] | None = None):
        self._on_mood = on_mood
        self._buffer = ""
        self._token_count = 0
        self._mood_emitted = False

    def feed(self, token: str) -> str:
        """Feed a token. Returns clean text (with mood tags stripped).

        May return empty string if currently accumulating a potential tag.
        """
        self._token_count += 1
        self._buffer += token

        # Check if we have or might be inside a mood tag
        if TAG_OPEN in self._buffer:
            match = MOOD_PATTERN.search(self._buffer)
            if match:
                user_tone = match.group(1)
                intensity = float(match.group(2))
                self._mood_emitted = True
                logger.debug("Mood signal: tone=%s, intensity=%.2f", user_tone, intensity)
                if self._on_mood:
                    self._on_mood(user_tone, intensity)
                clean = self._buffer[:match.start()] + self._buffer[match.end():]
                self._buffer = ""
                return clean.strip()
            # Partial tag — still accumulating
            if "</mood_signal>" not in self._buffer:
                if len(self._buffer) > 200:
                    result = self._buffer
                    self._buffer = ""
                    return result
                return ""

        # Check if buffer ends with a partial prefix of <mood_signal>
        if _could_be_tag_prefix(self._buffer):
            if len(self._buffer) > 200:
                result = self._buffer
                self._buffer = ""
                return result
            return ""

        # No tag involvement — emit buffer
        result = self._buffer
        self._buffer = ""

        # After 50 tokens with no mood: emit default
        if self._token_count >= 50 and not self._mood_emitted:
            self._mood_emitted = True
            logger.debug("No mood signal after 50 tokens, defaulting to neutral/0.5")
            if self._on_mood:
                self._on_mood("neutral", 0.5)

        return result

    def finalize(self) -> str:
        """Flush any remaining buffer. Call after stream ends."""
        if not self._mood_emitted and self._on_mood:
            self._on_mood("neutral", 0.5)
        result = self._buffer
        self._buffer = ""
        result = re.sub(r"<mood_signal>.*", "", result)
        return result.strip()

    def reset(self) -> None:
        self._buffer = ""
        self._token_count = 0
        self._mood_emitted = False
