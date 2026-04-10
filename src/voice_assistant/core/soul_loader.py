"""SOUL.yaml loader — single source of truth for persona configuration.

Parses, validates against schema, and distributes config to each module.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
import jsonschema

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "soul" / "schema.json"


@dataclass
class VoiceLanguageConfig:
    """Voice config for a single language."""
    id: str
    reference_audio: str | None = None


@dataclass
class SoulConfig:
    """Parsed and validated SOUL configuration."""
    name: str
    version: int
    voice_languages: list[VoiceLanguageConfig]
    mood_default: str
    mood_range: list[str]
    summary_style: str
    summary_trigger: float
    default_language: str
    llm_model: str
    llm_ctx_size: int
    llm_port: int
    llm_gpu_layers: int
    llm_threads: int
    llm_batch_size: int
    llm_flash_attn: bool
    sampling: dict
    personality: str
    backstory: str
    session_restore_hours: float
    raw: dict  # original parsed YAML

    def persona_card(self) -> str:
        """Generate the persona card for the system prompt."""
        parts = []
        if self.backstory:
            parts.append(self.backstory.strip())
        if self.personality:
            parts.append(self.personality.strip())
        return "\n\n".join(parts)


def load_soul(path: str | Path) -> SoulConfig:
    """Load and validate a SOUL.yaml file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SOUL file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"SOUL file must be a YAML mapping, got {type(raw).__name__}")

    # Validate against schema
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        try:
            jsonschema.validate(raw, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"SOUL.yaml validation failed: {e.message}") from e
    else:
        logger.warning("Schema file not found at %s, skipping validation", SCHEMA_PATH)

    voice = raw.get("voice", {})
    mood = raw.get("mood", {})
    memory = raw.get("memory", {})
    language = raw.get("language", {})
    llm = raw.get("llm", {})
    sampling = llm.get("sampling", {})

    # Parse per-language voice configs
    voice_langs_raw = voice.get("languages", [])
    voice_languages = []
    for vl in voice_langs_raw:
        if isinstance(vl, dict):
            voice_languages.append(VoiceLanguageConfig(
                id=vl.get("id", "en"),
                reference_audio=vl.get("reference_audio"),
            ))
        elif isinstance(vl, str):
            voice_languages.append(VoiceLanguageConfig(id=vl))
    if not voice_languages:
        voice_languages = [VoiceLanguageConfig(id="en")]

    return SoulConfig(
        name=raw["name"],
        version=raw.get("version", 2),
        voice_languages=voice_languages,
        mood_default=mood.get("default", "warm"),
        mood_range=mood.get("range", ["calm", "warm", "playful", "concerned"]),
        summary_style=memory.get("summary_style", "emotional"),
        summary_trigger=memory.get("summary_budget_trigger", 0.8),
        default_language=language.get("default", "en"),
        llm_model=llm.get("model", ""),
        llm_ctx_size=llm.get("ctx_size", 8192),
        llm_port=llm.get("port", 8080),
        llm_gpu_layers=llm.get("gpu_layers", 999),
        llm_threads=llm.get("threads", 4),
        llm_batch_size=llm.get("batch_size", 512),
        llm_flash_attn=llm.get("flash_attn", True),
        sampling=sampling,
        personality=raw.get("personality", ""),
        backstory=raw.get("backstory", ""),
        session_restore_hours=raw.get("session_restore_hours", 4),
        raw=raw,
    )
