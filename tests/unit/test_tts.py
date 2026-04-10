"""Tests for OmniVoice TTS adapter and voice clone utilities.

Uses mocks — does not require the actual OmniVoice model to be downloaded.
"""
import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from voice_assistant.adapters.omnivoice_tts import OmniVoiceTTS
from voice_assistant.core.voice_clone import (
    get_profile_path,
    load_voice_clone_prompt,
    PROFILES_DIR,
)


# --- voice_clone.py tests ---


class TestGetProfilePath:
    def test_returns_path(self):
        p = get_profile_path("Aria")
        assert p == PROFILES_DIR / "Aria.voiceprofile"


class TestLoadVoiceClonePrompt:
    def test_roundtrip(self, tmp_path):
        profile_path = tmp_path / "test.voiceprofile"
        tokens = torch.randn(1, 100)
        torch.save(
            {"ref_audio_tokens": tokens, "ref_text": "hello world", "ref_rms": 0.05},
            profile_path,
        )
        prompt = load_voice_clone_prompt(profile_path)
        assert prompt.ref_text == "hello world"
        assert abs(prompt.ref_rms - 0.05) < 1e-6
        assert torch.allclose(prompt.ref_audio_tokens, tokens)


# --- OmniVoiceTTS tests ---


class MockOmniVoice:
    """Mock OmniVoice model for testing without GPU/model download."""

    def eval(self):
        return self

    def generate(self, text, speed=1.0, voice_clone_prompt=None, instruct=None, language=None):
        n_samples = int(len(text) * 0.06 * 24000)
        return [torch.randn(1, max(n_samples, 100))]


class TestOmniVoiceTTS:
    @pytest.fixture
    def tts(self):
        return OmniVoiceTTS()

    def test_instantiation(self, tts):
        assert tts._model is None
        assert tts.SAMPLE_RATE == 24000

    @pytest.mark.asyncio
    async def test_synthesize_without_load_raises(self, tts):
        with pytest.raises(RuntimeError, match="not loaded"):
            await tts.synthesize("hello")

    @pytest.mark.asyncio
    async def test_load_model(self, tts):
        with patch("omnivoice.OmniVoice") as MockClass:
            MockClass.from_pretrained.return_value = MockOmniVoice()
            tts._load_sync()
        assert tts._model is not None

    @pytest.mark.asyncio
    async def test_synthesize_with_clone_prompt(self, tts):
        tts._model = MockOmniVoice()
        from omnivoice.models.omnivoice import VoiceClonePrompt
        prompt = VoiceClonePrompt(
            ref_audio_tokens=torch.randn(1, 50), ref_text="ref", ref_rms=0.05
        )
        tts._voice_prompts["en"] = prompt
        tts._active_language = "en"
        result = await tts.synthesize("cloned speech test")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_synthesize_no_profile_raises(self, tts):
        tts._model = MockOmniVoice()
        with pytest.raises(RuntimeError, match="No voice profile"):
            await tts.synthesize("hello")

    @pytest.mark.asyncio
    async def test_per_language_profiles(self, tts):
        tts._model = MockOmniVoice()
        from omnivoice.models.omnivoice import VoiceClonePrompt
        prompt_en = VoiceClonePrompt(ref_audio_tokens=torch.randn(1, 50), ref_text="en", ref_rms=0.05)
        prompt_hr = VoiceClonePrompt(ref_audio_tokens=torch.randn(1, 50), ref_text="hr", ref_rms=0.05)
        tts._voice_prompts["en"] = prompt_en
        tts._voice_prompts["hr"] = prompt_hr
        assert tts.available_languages == ["en", "hr"]
        tts.set_language("hr")
        assert tts._active_language == "hr"

    @pytest.mark.asyncio
    async def test_shutdown(self, tts):
        tts._model = MockOmniVoice()
        await tts.shutdown()
        assert tts._model is None
        assert tts._voice_prompts == {}


def _mock_import(name, *args, **kwargs):
    """Allow normal imports but intercept omnivoice."""
    if name == "omnivoice":
        mock = MagicMock()
        mock.OmniVoice.from_pretrained.return_value = MockOmniVoice()
        return mock
    return original_import(name, *args, **kwargs)


import builtins
original_import = builtins.__import__
