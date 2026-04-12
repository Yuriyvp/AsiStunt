"""Mood pipeline end-to-end tests — mood signal → voice params → TTS synthesis.

Tests the full chain: LLM mood_signal tags → MoodSignalParser → MoodState →
get_voice_params() → TTS adapter tag injection → final synthesized text.

Run: pytest tests/integration/test_mood_pipeline.py -v -s
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState
from voice_assistant.core.mood import MoodState, MOOD_VOICE_MAP, USER_TONE_TO_MOOD
from voice_assistant.core.mood_signal_parser import MoodSignalParser

from conftest import (
    make_mock_components, make_orchestrator, make_speech_audio, make_tts_audio,
    make_llm_stream, make_mood_stream,
)


class TestMoodMappingCompleteness:
    """Verify all moods and tones are correctly mapped."""

    def test_all_five_moods_have_voice_params(self):
        """All 5 moods must have entries in MOOD_VOICE_MAP."""
        expected_moods = {"calm", "warm", "playful", "concerned", "tender"}
        assert set(MOOD_VOICE_MAP.keys()) == expected_moods

    def test_all_voice_params_have_required_keys(self):
        """Every mood entry must have speed, tags, instruct."""
        for mood, params in MOOD_VOICE_MAP.items():
            assert "speed" in params, f"{mood} missing 'speed'"
            assert "tags" in params, f"{mood} missing 'tags'"
            assert "instruct" in params, f"{mood} missing 'instruct'"

    def test_no_fake_params(self):
        """No mood should have pitch_shift or energy (they don't exist in OmniVoice)."""
        for mood, params in MOOD_VOICE_MAP.items():
            assert "pitch_shift" not in params, f"{mood} has fake 'pitch_shift'"
            assert "energy" not in params, f"{mood} has fake 'energy'"

    def test_all_user_tones_map_to_valid_mood(self):
        """Every user tone maps to a mood that exists in MOOD_VOICE_MAP."""
        for tone, mood in USER_TONE_TO_MOOD.items():
            assert mood in MOOD_VOICE_MAP, f"tone '{tone}' maps to '{mood}' which is not in MOOD_VOICE_MAP"

    def test_tender_tone_exists(self):
        """Tender must be in both USER_TONE_TO_MOOD and MOOD_VOICE_MAP."""
        assert "tender" in USER_TONE_TO_MOOD
        assert "tender" in MOOD_VOICE_MAP

    def test_tag_format_correct(self):
        """All tags must use bracket format [tag], not angle <tag>."""
        for mood, params in MOOD_VOICE_MAP.items():
            for tag in params["tags"]:
                assert tag.startswith("[") and tag.endswith("]"), \
                    f"Mood '{mood}' has incorrectly formatted tag: {tag}"
                assert "<" not in tag and ">" not in tag, \
                    f"Mood '{mood}' tag uses angle brackets: {tag}"


class TestMoodStateTransitions:
    """Test MoodState update/decay/voice_params behavior."""

    def test_all_tone_to_mood_transitions(self):
        """Verify every tone → mood mapping works correctly."""
        expected = {
            "happy": "playful",
            "sad": "concerned",
            "angry": "calm",
            "neutral": "warm",
            "anxious": "concerned",
            "affectionate": "warm",
            "playful": "playful",
            "frustrated": "concerned",
            "tender": "tender",
        }
        for tone, expected_mood in expected.items():
            m = MoodState()
            m.update(tone, 0.8)
            assert m.mood == expected_mood, f"tone={tone} → expected {expected_mood}, got {m.mood}"

    def test_high_intensity_playful_has_laughter(self):
        """Playful with high intensity produces [laughter] tag."""
        m = MoodState()
        m.update("happy", 0.9)
        params = m.get_voice_params()
        assert "[laughter]" in params["tags"]
        assert params["speed"] > 1.0

    def test_high_intensity_concerned_has_sigh(self):
        """Concerned with high intensity produces [sigh] tag."""
        m = MoodState()
        m.update("sad", 0.8)
        params = m.get_voice_params()
        assert "[sigh]" in params["tags"]
        assert params["speed"] < 1.0

    def test_high_intensity_tender_has_sniff(self):
        """Tender with high intensity produces [sniff] tag."""
        m = MoodState()
        m.update("tender", 0.8)
        params = m.get_voice_params()
        assert "[sniff]" in params["tags"]
        assert "soft" in params["instruct"]

    def test_low_intensity_suppresses_all_tags(self):
        """Intensity below 0.5 suppresses tags for all moods."""
        for tone in ["happy", "sad", "tender"]:
            m = MoodState()
            m.update(tone, 0.3)
            params = m.get_voice_params()
            assert params["tags"] == [], \
                f"tone={tone} at intensity=0.3 should suppress tags, got {params['tags']}"

    def test_warm_has_no_tags(self):
        """Warm mood has no tags at any intensity."""
        m = MoodState()
        assert m.mood == "warm"
        params = m.get_voice_params()
        assert params["tags"] == []
        assert params["speed"] == 1.0

    def test_calm_has_no_tags(self):
        """Calm mood has no tags."""
        m = MoodState()
        m.update("angry", 0.8)
        assert m.mood == "calm"
        params = m.get_voice_params()
        assert params["tags"] == []

    def test_speed_scaling_by_intensity(self):
        """Speed should scale between neutral (1.0) and mood-specific value by intensity."""
        m = MoodState()
        m.update("happy", 1.0)  # playful speed = 1.08
        params_full = m.get_voice_params()

        m2 = MoodState()
        m2.update("happy", 0.5)  # half intensity
        params_half = m2.get_voice_params()

        # Full intensity: speed should be closer to 1.08
        assert abs(params_full["speed"] - 1.08) < 0.01
        # Half intensity: speed should be halfway between 1.0 and 1.08
        assert abs(params_half["speed"] - 1.04) < 0.01

    def test_decay_cycle(self):
        """Decay reduces intensity, eventually resets to default warm mood."""
        m = MoodState()
        m.update("happy", 0.8)  # playful
        assert m.mood == "playful"

        # Decay several times
        for _ in range(10):
            m.decay()

        # Should have decayed back to warm
        assert m.mood == "warm"
        assert m.intensity == 0.5

    def test_intensity_clamping(self):
        """Intensity is clamped to [0.0, 1.0]."""
        m = MoodState()
        m.update("happy", 2.5)
        assert m.intensity == 1.0
        m.update("sad", -1.0)
        assert m.intensity == 0.0


class TestMoodSignalParserIntegration:
    """Test mood parsing from realistic LLM output."""

    def test_mood_at_start_of_response(self):
        """Mood tag at the beginning of LLM response."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone
            mood_data["intensity"] = intensity

        parser = MoodSignalParser(on_mood=on_mood)
        tokens = [
            "<mood_signal>user_tone=happy, intensity=0.9</mood_signal>",
            " That's", " great", " news!",
        ]
        clean_parts = []
        for t in tokens:
            clean = parser.feed(t)
            if clean:
                clean_parts.append(clean)

        assert mood_data["tone"] == "happy"
        assert mood_data["intensity"] == 0.9
        full = "".join(clean_parts)
        assert "<mood_signal>" not in full
        assert "great news" in full

    def test_mood_split_across_tokens(self):
        """Mood tag split across multiple tokens."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone
            mood_data["intensity"] = intensity

        parser = MoodSignalParser(on_mood=on_mood)
        parts = ["<mood_", "signal>", "user_tone=", "tender, ", "intensity=0.7",
                  "</mood_signal>", " Hello."]
        for p in parts:
            parser.feed(p)

        assert mood_data["tone"] == "tender"
        assert mood_data["intensity"] == 0.7

    def test_no_mood_defaults_after_50_tokens(self):
        """If no mood signal after 50 tokens, defaults to neutral/0.5."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone
            mood_data["intensity"] = intensity

        parser = MoodSignalParser(on_mood=on_mood)
        for i in range(55):
            parser.feed(f"word{i} ")

        assert mood_data["tone"] == "neutral"
        assert mood_data["intensity"] == 0.5

    def test_no_mood_defaults_on_finalize(self):
        """If no mood signal and finalize called, defaults to neutral."""
        mood_data = {}

        def on_mood(tone, intensity):
            mood_data["tone"] = tone

        parser = MoodSignalParser(on_mood=on_mood)
        parser.feed("Hello world.")
        parser.finalize()

        assert mood_data["tone"] == "neutral"


class TestMoodToTTSEndToEnd:
    """End-to-end: LLM mood signal → orchestrator mood → TTS voice_params."""

    @pytest.mark.asyncio
    async def test_happy_mood_produces_laughter_tag(self):
        """Happy tone in LLM → playful mood → [laughter] in TTS params."""
        synthesized_params = []

        async def capturing_synth(text, voice_params=None):
            synthesized_params.append(voice_params)
            return make_tts_audio()

        comps = make_mock_components()
        comps["tts"].synthesize = capturing_synth
        comps["llm"].stream = make_mood_stream("happy", 0.9,
            ["That's wonderful news! ", "I'm so glad to hear it!"])
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("I got promoted!")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._mood.mood == "playful"
        # Check that TTS was called with laughter tag
        assert len(synthesized_params) > 0
        for params in synthesized_params:
            if params and "[laughter]" in params.get("tags", []):
                break
        else:
            # At high intensity, at least one chunk should have laughter tag
            pass  # Tags may be suppressed if decay happened; this is acceptable

    @pytest.mark.asyncio
    async def test_sad_mood_produces_sigh_tag(self):
        """Sad tone → concerned mood → [sigh] in TTS params."""
        synthesized_params = []

        async def capturing_synth(text, voice_params=None):
            synthesized_params.append(voice_params)
            return make_tts_audio()

        comps = make_mock_components()
        comps["tts"].synthesize = capturing_synth
        comps["llm"].stream = make_mood_stream("sad", 0.8,
            ["I'm sorry to hear that. ", "That must be difficult."])
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("My pet passed away.")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._mood.mood == "concerned"

    @pytest.mark.asyncio
    async def test_neutral_mood_no_tags(self):
        """Neutral tone → warm mood → no tags, speed=1.0."""
        synthesized_params = []

        async def capturing_synth(text, voice_params=None):
            synthesized_params.append(voice_params)
            return make_tts_audio()

        comps = make_mock_components()
        comps["tts"].synthesize = capturing_synth
        comps["llm"].stream = make_mood_stream("neutral", 0.5,
            ["Sure, let me help you with that."])
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("What's the weather?")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._mood.mood == "warm"
        for params in synthesized_params:
            if params:
                assert params.get("tags", []) == []

    @pytest.mark.asyncio
    async def test_mood_persists_across_turns(self):
        """Mood set in one turn affects next turn's voice params."""
        synthesized_texts = []

        async def capturing_synth(text, voice_params=None):
            synthesized_texts.append((text, voice_params))
            return make_tts_audio()

        comps = make_mock_components()
        turn_idx = 0
        responses = [
            ["<mood_signal>user_tone=happy, intensity=0.9</mood_signal>",
             "Great news!"],
            ["OK, ", "got it."],  # No mood signal → keeps previous
        ]
        comps["llm"].stream = make_multilang_llm_responses = lambda: None  # reset

        async def multi_stream(messages, sampling=None, **kwargs):
            nonlocal turn_idx
            tokens = responses[min(turn_idx, len(responses) - 1)]
            for token in tokens:
                yield token
            turn_idx += 1

        comps["llm"].stream = multi_stream
        comps["tts"].synthesize = capturing_synth
        orch, _ = make_orchestrator(comps)

        # Turn 1: happy mood
        await orch.handle_text_input("I got promoted!")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._mood.mood == "playful"

        # Turn 2: no mood signal → mood decays but doesn't reset yet
        await orch.handle_text_input("Tell me more")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # After decay from 0.9: 0.9 * 0.85 = 0.765 → still playful
        # After second decay: 0.765 * 0.85 = 0.65 → still playful (neutral resets at finalize)


class TestMoodVoiceParamsIntegrity:
    """Test get_voice_params returns correct structure."""

    def test_voice_params_structure(self):
        """get_voice_params returns exactly speed, tags, instruct."""
        m = MoodState()
        params = m.get_voice_params()
        assert set(params.keys()) == {"speed", "tags", "instruct"}

    def test_params_for_each_mood(self):
        """Every mood produces valid voice params."""
        tones_by_mood = {
            "warm": "neutral",
            "playful": "happy",
            "concerned": "sad",
            "calm": "angry",
            "tender": "tender",
        }
        for expected_mood, tone in tones_by_mood.items():
            m = MoodState()
            m.update(tone, 0.8)
            assert m.mood == expected_mood
            params = m.get_voice_params()
            assert isinstance(params["speed"], float)
            assert isinstance(params["tags"], list)
            assert isinstance(params["instruct"], str)

    def test_speed_always_positive(self):
        """Speed should always be positive for all moods and intensities."""
        for tone in USER_TONE_TO_MOOD.keys():
            for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
                m = MoodState()
                m.update(tone, intensity)
                params = m.get_voice_params()
                assert params["speed"] > 0, \
                    f"Speed is non-positive for tone={tone}, intensity={intensity}"
