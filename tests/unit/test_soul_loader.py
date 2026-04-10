"""Tests for SOUL.yaml loader."""
import pytest
from pathlib import Path

from voice_assistant.core.soul_loader import load_soul, SoulConfig

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "soul_files"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestLoadSoul:
    def test_load_default_soul(self):
        """Load the real default.soul.yaml."""
        soul = load_soul(PROJECT_ROOT / "soul" / "default.soul.yaml")
        assert soul.name == "Aria"
        assert soul.version == 2
        assert len(soul.voice_languages) == 3
        assert soul.voice_languages[0].id == "hr"
        assert soul.default_language == "hr"
        assert soul.llm_ctx_size == 8192
        assert soul.llm_flash_attn is True
        assert soul.sampling["temperature"] == 0.75

    def test_load_valid_fixture(self):
        """Load a valid test fixture."""
        soul = load_soul(FIXTURES / "valid.soul.yaml")
        assert soul.name == "TestBot"
        assert len(soul.voice_languages) >= 1
        assert soul.mood_default == "calm"
        assert soul.mood_range == ["calm", "warm"]
        assert soul.llm_port == 9090
        assert soul.llm_gpu_layers == 10
        assert soul.llm_flash_attn is False
        assert soul.sampling["temperature"] == 0.5
        assert soul.session_restore_hours == 2

    def test_persona_card(self):
        """Persona card combines backstory + personality."""
        soul = load_soul(FIXTURES / "valid.soul.yaml")
        card = soul.persona_card()
        assert "Created for testing." in card
        assert "You are a test bot." in card
        assert card.index("Created for testing.") < card.index("You are a test bot.")

    def test_persona_card_empty_backstory(self):
        """Persona card works with no backstory."""
        soul = load_soul(FIXTURES / "valid.soul.yaml")
        soul.backstory = ""
        card = soul.persona_card()
        assert "You are a test bot." in card
        assert "\n\n" not in card

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_soul("/nonexistent/path.yaml")

    def test_invalid_missing_name(self):
        """Schema validation catches missing required 'name' field."""
        with pytest.raises(ValueError, match="validation failed"):
            load_soul(FIXTURES / "invalid_missing_name.soul.yaml")

    def test_raw_preserved(self):
        """Original YAML dict is preserved in .raw."""
        soul = load_soul(FIXTURES / "valid.soul.yaml")
        assert isinstance(soul.raw, dict)
        assert soul.raw["name"] == "TestBot"
        assert soul.raw["llm"]["model"] == "models/test.gguf"

    def test_defaults_for_optional_fields(self):
        """Missing optional fields get sensible defaults."""
        soul = load_soul(PROJECT_ROOT / "soul" / "default.soul.yaml")
        assert isinstance(soul.summary_style, str)
        assert 0 <= soul.summary_trigger <= 1

    def test_voice_languages(self):
        """Voice languages are parsed correctly."""
        soul = load_soul(PROJECT_ROOT / "soul" / "default.soul.yaml")
        lang_ids = [vl.id for vl in soul.voice_languages]
        assert "hr" in lang_ids
        assert "en" in lang_ids
        # All reference_audio should be None initially
        for vl in soul.voice_languages:
            assert vl.reference_audio is None

    def test_sampling_dict(self):
        """Sampling params are a dict passable to LLM adapter."""
        soul = load_soul(PROJECT_ROOT / "soul" / "default.soul.yaml")
        assert isinstance(soul.sampling, dict)
        assert "temperature" in soul.sampling
        assert "top_p" in soul.sampling
