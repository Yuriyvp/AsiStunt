"""Tests for MoodSignalParser and MoodState."""
import pytest

from voice_assistant.core.mood_signal_parser import MoodSignalParser
from voice_assistant.core.mood import MoodState


class TestMoodSignalParser:
    def test_extracts_mood_tag(self):
        """Should extract mood signal and return clean text."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone
            mood_data["intensity"] = intensity

        parser = MoodSignalParser(on_mood=on_mood)
        tokens = "<mood_signal>user_tone=happy, intensity=0.8</mood_signal> Sure, I would love to help!".split()
        clean_parts = []
        for t in tokens:
            clean = parser.feed(t + " ")
            if clean:
                clean_parts.append(clean)
        remaining = parser.finalize()
        if remaining:
            clean_parts.append(remaining)

        assert mood_data["tone"] == "happy"
        assert mood_data["intensity"] == 0.8
        full_text = " ".join(clean_parts).strip()
        assert "<mood_signal>" not in full_text
        assert "help" in full_text

    def test_default_mood_after_50_tokens(self):
        """Should emit neutral/0.5 if no mood tag after 50 tokens."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone
            mood_data["intensity"] = intensity

        parser = MoodSignalParser(on_mood=on_mood)
        for i in range(55):
            parser.feed(f"word{i} ")

        assert mood_data.get("tone") == "neutral"
        assert mood_data.get("intensity") == 0.5

    def test_default_mood_on_finalize(self):
        """Should emit default mood on finalize if no tag seen."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone
            mood_data["intensity"] = intensity

        parser = MoodSignalParser(on_mood=on_mood)
        parser.feed("Hello ")
        parser.finalize()

        assert mood_data["tone"] == "neutral"

    def test_partial_tag_accumulation(self):
        """Should handle mood tag split across multiple tokens."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone
            mood_data["intensity"] = intensity

        parser = MoodSignalParser(on_mood=on_mood)
        # Split the tag across tokens
        parts = ["<mood_", "signal>user_tone=", "sad, intensity=", "0.6</mood_signal>", " I'm sorry."]
        clean_parts = []
        for p in parts:
            clean = parser.feed(p)
            if clean:
                clean_parts.append(clean)

        assert mood_data["tone"] == "sad"
        assert mood_data["intensity"] == 0.6

    def test_no_callback_no_crash(self):
        """Should work without a callback."""
        parser = MoodSignalParser()
        clean = parser.feed("<mood_signal>user_tone=happy, intensity=0.8</mood_signal> Hello!")
        assert "<mood_signal>" not in clean

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        parser = MoodSignalParser()
        parser.feed("some text")
        parser.reset()
        assert parser._buffer == ""
        assert parser._token_count == 0
        assert parser._mood_emitted is False


class TestMoodState:
    def test_default_state(self):
        m = MoodState()
        assert m.mood == "warm"
        assert m.intensity == 0.5

    def test_update_changes_mood(self):
        m = MoodState()
        result = m.update("happy", 0.8)
        assert result == "playful"
        assert m.mood == "playful"
        assert m.intensity == 0.8

    def test_update_same_mood_returns_none(self):
        m = MoodState()
        m.update("neutral", 0.5)  # warm -> warm (neutral maps to warm)
        result = m.update("neutral", 0.6)
        assert result is None

    def test_decay_reduces_intensity(self):
        m = MoodState()
        m.update("happy", 0.8)
        m.decay()
        assert m.intensity == pytest.approx(0.68)

    def test_decay_resets_to_default(self):
        m = MoodState()
        m.update("happy", 0.2)
        m.decay()  # 0.2 * 0.85 = 0.17 < 0.2
        assert m.mood == "warm"
        assert m.intensity == 0.5

    def test_voice_params_warm(self):
        m = MoodState()
        params = m.get_voice_params()
        assert params["speed"] == 1.0
        assert params["energy"] == 1.0

    def test_voice_params_playful(self):
        m = MoodState()
        m.update("happy", 0.9)
        params = m.get_voice_params()
        assert params["speed"] > 1.0
        assert params["energy"] > 1.0
        assert "<laugh>" in params["tags"]

    def test_intensity_clamped(self):
        m = MoodState()
        m.update("happy", 1.5)
        assert m.intensity == 1.0
        m.update("sad", -0.5)
        assert m.intensity == 0.0
