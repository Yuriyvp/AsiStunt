"""Infrastructure settings loader — separate from SOUL personality.

Reads config/settings.yaml for LLM, TTS, voice, language, and memory settings.
These are used at startup by ProcessManager and main.py, never sent to the LLM.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class VoiceLanguageConfig:
    """Voice config for a single language."""
    id: str
    reference_audio: str | None = None


@dataclass
class Settings:
    """Infrastructure settings — loaded once at startup."""
    # LLM server
    llm_model: str = ""
    llm_ctx_size: int = 8192
    llm_port: int = 8080
    llm_gpu_layers: int = 999
    llm_threads: int = 4
    llm_batch_size: int = 512
    llm_flash_attn: bool = True
    sampling: dict = field(default_factory=dict)

    # Voice / language
    voice_languages: list[VoiceLanguageConfig] = field(default_factory=list)
    default_language: str = "en"

    # Memory
    summary_style: str = "emotional"
    summary_trigger: float = 0.8
    session_restore_hours: float = 4

    # Raw YAML for save-back
    raw: dict = field(default_factory=dict)
    path: str = ""


def load_settings(path: str | Path) -> Settings:
    """Load infrastructure settings from a YAML file."""
    path = Path(path)
    if not path.exists():
        logger.warning("Settings file not found: %s — using defaults", path)
        return Settings()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    llm = raw.get("llm", {})
    voice = raw.get("voice", {})
    language = raw.get("language", {})
    memory = raw.get("memory", {})

    # Parse voice languages
    voice_langs = []
    for vl in voice.get("languages", []):
        if isinstance(vl, dict):
            voice_langs.append(VoiceLanguageConfig(
                id=vl.get("id", "en"),
                reference_audio=vl.get("reference_audio"),
            ))
        elif isinstance(vl, str):
            voice_langs.append(VoiceLanguageConfig(id=vl))
    if not voice_langs:
        voice_langs = [VoiceLanguageConfig(id="en")]

    return Settings(
        llm_model=llm.get("model", ""),
        llm_ctx_size=llm.get("ctx_size", 8192),
        llm_port=llm.get("port", 8080),
        llm_gpu_layers=llm.get("gpu_layers", 999),
        llm_threads=llm.get("threads", 4),
        llm_batch_size=llm.get("batch_size", 512),
        llm_flash_attn=llm.get("flash_attn", True),
        sampling=llm.get("sampling", {}),
        voice_languages=voice_langs,
        default_language=language.get("default", "en"),
        summary_style=memory.get("summary_style", "emotional"),
        summary_trigger=memory.get("summary_budget_trigger", 0.8),
        session_restore_hours=raw.get("session_restore_hours", 4),
        raw=raw,
        path=str(path),
    )
