"""Mood state management.

Moods: calm, warm, playful, concerned.
The LLM emits <mood_signal> tags, parsed by MoodSignalParser.
Mood decays toward default each turn (intensity *= 0.85).
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

USER_TONE_TO_MOOD = {
    "happy": "playful",
    "sad": "concerned",
    "angry": "calm",
    "neutral": "warm",
    "anxious": "concerned",
    "affectionate": "warm",
    "playful": "playful",
    "frustrated": "concerned",
}

MOOD_VOICE_MAP = {
    "calm":      {"speed": 1.0, "pitch_shift": 0,  "energy": 0.8, "tags": []},
    "warm":      {"speed": 1.0, "pitch_shift": 0,  "energy": 1.0, "tags": []},
    "playful":   {"speed": 1.1, "pitch_shift": +1, "energy": 1.1, "tags": ["<laugh>"]},
    "concerned": {"speed": 0.9, "pitch_shift": -1, "energy": 0.9, "tags": []},
}


@dataclass
class MoodState:
    mood: str = "warm"
    intensity: float = 0.5
    user_tone: str = "neutral"

    def update(self, user_tone: str, intensity: float) -> str | None:
        """Update mood from LLM signal. Returns new mood name if changed, else None."""
        self.user_tone = user_tone
        self.intensity = max(0.0, min(1.0, intensity))
        new_mood = USER_TONE_TO_MOOD.get(user_tone, "warm")
        changed = new_mood != self.mood
        self.mood = new_mood
        if changed:
            logger.info("Mood changed to %s (tone=%s, intensity=%.2f)", self.mood, user_tone, self.intensity)
        return new_mood if changed else None

    def decay(self, default_mood: str = "warm") -> None:
        """Decay intensity toward default. Call once per turn."""
        self.intensity *= 0.85
        if self.intensity < 0.2:
            self.mood = default_mood
            self.intensity = 0.5

    def get_voice_params(self) -> dict:
        """Get OmniVoice parameters for current mood, scaled by intensity."""
        base = MOOD_VOICE_MAP.get(self.mood, MOOD_VOICE_MAP["warm"])
        neutral = MOOD_VOICE_MAP["warm"]
        scale = self.intensity
        return {
            "speed": neutral["speed"] + (base["speed"] - neutral["speed"]) * scale,
            "pitch_shift": int(base["pitch_shift"] * scale),
            "energy": neutral["energy"] + (base["energy"] - neutral["energy"]) * scale,
            "tags": base["tags"] if scale > 0.5 else [],
        }
