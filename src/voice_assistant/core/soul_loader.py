"""SOUL.yaml loader — personality configuration only.

Infrastructure settings (LLM, TTS, voice, memory) live in config/settings.yaml
and are loaded by settings_loader.py. This file only handles personality:
name, personality, backstory, mood.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SoulConfig:
    """Personality configuration — sent to LLM as system prompt context."""
    name: str
    personality: str = ""
    backstory: str = ""
    mood_default: str = "warm"
    mood_range: list[str] = field(default_factory=lambda: ["calm", "warm", "playful", "concerned"])
    raw: dict = field(default_factory=dict)

    def persona_card(self) -> str:
        """Generate the persona card for the system prompt."""
        parts = []
        if self.backstory:
            parts.append(self.backstory.strip())
        if self.personality:
            parts.append(self.personality.strip())
        return "\n\n".join(parts)


# Keep VoiceLanguageConfig importable from here for backwards compat
from voice_assistant.core.settings_loader import VoiceLanguageConfig  # noqa: E402, F401


def load_soul(path: str | Path) -> SoulConfig:
    """Load and validate a SOUL.yaml personality file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SOUL file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"SOUL file must be a YAML mapping, got {type(raw).__name__}")

    if "name" not in raw:
        raise ValueError("SOUL file missing required field: name")

    mood = raw.get("mood", {})

    return SoulConfig(
        name=raw["name"],
        personality=raw.get("personality", ""),
        backstory=raw.get("backstory", ""),
        mood_default=mood.get("default", "warm"),
        mood_range=mood.get("range", ["calm", "warm", "playful", "concerned"]),
        raw=raw,
    )
