"""Unit tests for multi-strategy language detection.

Covers: Cyrillic detection, diacritics detection, langdetect fallback,
word-hint fallback, short text rejection, unsupported language rejection.
"""
import pytest

from voice_assistant.core.language_detector import detect_language


SUPPORTED = {"en", "hr", "ru"}


class TestCyrillicDetection:
    """Strategy 1: Cyrillic script → Russian."""

    def test_russian_text(self):
        assert detect_language("У меня все отлично. Я сегодня хорошо провел день.", SUPPORTED) == "ru"

    def test_russian_short_phrase(self):
        assert detect_language("Привет как дела у тебя сегодня", SUPPORTED) == "ru"

    def test_russian_question(self):
        assert detect_language("Где ты был вчера вечером?", SUPPORTED) == "ru"


class TestDiacriticsDetection:
    """Strategy 2: South Slavic diacritics → Croatian."""

    def test_croatian_with_dj(self):
        assert detect_language("Oni se dođu u centr s prijatelicami.", SUPPORTED) == "hr"

    def test_croatian_with_s_caron(self):
        assert detect_language("Idem u školu svaki dan.", SUPPORTED) == "hr"

    def test_croatian_with_z_caron(self):
        assert detect_language("Molim vas, možete li mi pomoći?", SUPPORTED) == "hr"


class TestLangdetectFallback:
    """Strategy 3: langdetect filtered by supported languages."""

    def test_croatian_without_diacritics_detected(self):
        result = detect_language("Kako si? Dobro sam, hvala na pitanju.", SUPPORTED)
        assert result == "hr"

    def test_english_longer_text(self):
        assert detect_language("Yeah today I drive uh my daughter with the car to the city centre", SUPPORTED) == "en"

    def test_croatian_informal(self):
        assert detect_language("Imam puno posla za danas.", SUPPORTED) == "hr"


class TestWordHintFallback:
    """Strategy 4: common word matching when langdetect fails."""

    def test_english_greeting(self):
        # langdetect detects this as Somali — word hints catch it
        assert detect_language("Hey, hi, how are you today?", SUPPORTED) == "en"

    def test_english_casual(self):
        assert detect_language("What is going on here today?", SUPPORTED) == "en"

    def test_croatian_no_diacritics(self):
        # langdetect detects this as Indonesian — word hints catch it
        assert detect_language("Dobar dan, kako ste danas?", SUPPORTED) == "hr"

    def test_english_simple(self):
        assert detect_language("Oh that sounds like a nice little trip", SUPPORTED) == "en"

    def test_croatian_common_words(self):
        assert detect_language("Ja sam dobro danas hvala", SUPPORTED) == "hr"


class TestShortTextRejection:
    """Text below MIN_CHARS should return None."""

    def test_too_short(self):
        assert detect_language("ok", SUPPORTED) is None

    def test_barely_short(self):
        assert detect_language("hello world", SUPPORTED) is None

    def test_empty(self):
        assert detect_language("", SUPPORTED) is None

    def test_whitespace(self):
        assert detect_language("   ", SUPPORTED) is None


class TestUnsupportedLanguage:
    """Languages not in supported set should return None or alias."""

    def test_german_not_supported(self):
        # German text, but "de" not in supported set
        result = detect_language("Alles hat ein Ende, nur die Wurst hat zwei.", SUPPORTED)
        assert result is None or result in SUPPORTED

    def test_serbian_aliases_to_croatian(self):
        # If langdetect returns "sr", it should alias to "hr"
        supported_with_hr = {"hr", "en"}
        result = detect_language("Kako si? Dobro sam, hvala na pitanju.", supported_with_hr)
        assert result == "hr"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_mixed_script_cyrillic_dominant(self):
        # Cyrillic dominant → Russian
        assert detect_language("Привет hello как дела сегодня", SUPPORTED) == "ru"

    def test_numbers_only(self):
        assert detect_language("123456789012345", SUPPORTED) is None

    def test_supported_languages_as_list(self):
        # Should accept list as well as set
        assert detect_language("Hello, how are you doing today?", ["en", "hr", "ru"]) == "en"

    def test_single_supported_language(self):
        assert detect_language("Hello, how are you doing today?", {"en"}) == "en"
