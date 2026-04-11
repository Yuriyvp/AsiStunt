"""Tests for SOUL.yaml loader (personality only) and settings loader."""
import pytest
from pathlib import Path

from voice_assistant.core.soul_loader import load_soul, SoulConfig
from voice_assistant.core.settings_loader import load_settings, Settings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "soul_files"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestLoadSoul:
    def test_load_default_soul(self):
        """Load the real default.soul.yaml (personality only)."""
        soul = load_soul(PROJECT_ROOT / "soul" / "default.soul.yaml")
        assert soul.name == "Joi"
        assert "warm" in soul.personality or "Peka" in soul.personality

    def test_load_valid_fixture(self):
        """Load a valid test fixture — only personality fields matter."""
        soul = load_soul(FIXTURES / "valid.soul.yaml")
        assert soul.name == "TestBot"
        assert soul.mood_default == "calm"
        assert soul.mood_range == ["calm", "warm"]

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
        """Missing required 'name' field raises ValueError."""
        with pytest.raises(ValueError, match="missing required field: name"):
            load_soul(FIXTURES / "invalid_missing_name.soul.yaml")

    def test_raw_preserved(self):
        """Original YAML dict is preserved in .raw."""
        soul = load_soul(FIXTURES / "valid.soul.yaml")
        assert isinstance(soul.raw, dict)
        assert soul.raw["name"] == "TestBot"

    def test_no_infrastructure_fields(self):
        """SoulConfig should NOT have infrastructure fields."""
        soul = load_soul(FIXTURES / "valid.soul.yaml")
        assert not hasattr(soul, "llm_model")
        assert not hasattr(soul, "sampling")
        assert not hasattr(soul, "voice_languages")
        assert not hasattr(soul, "default_language")


class TestLoadSettings:
    def test_load_default_settings(self):
        """Load the real config/settings.yaml."""
        s = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
        assert s.llm_model != ""
        assert s.llm_ctx_size == 8192
        assert s.llm_flash_attn is True
        assert len(s.voice_languages) == 3
        assert s.default_language == "ru"
        assert s.sampling["temperature"] == 0.75

    def test_voice_languages_parsed(self):
        """Voice languages have id and optional reference_audio."""
        s = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
        lang_ids = [vl.id for vl in s.voice_languages]
        assert "hr" in lang_ids
        assert "en" in lang_ids
        assert "ru" in lang_ids
        for vl in s.voice_languages:
            assert vl.reference_audio is None or isinstance(vl.reference_audio, str)

    def test_missing_file_returns_defaults(self):
        """Missing settings file returns defaults, no crash."""
        s = load_settings("/nonexistent/settings.yaml")
        assert s.llm_model == ""
        assert s.llm_ctx_size == 8192
        assert s.default_language == "en"

    def test_sampling_dict(self):
        """Sampling params are a dict passable to LLM adapter."""
        s = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
        assert isinstance(s.sampling, dict)
        assert "temperature" in s.sampling
        assert "top_p" in s.sampling

    def test_path_stored(self):
        """Settings stores its own path for save-back."""
        s = load_settings(PROJECT_ROOT / "config" / "settings.yaml")
        assert "settings.yaml" in s.path
