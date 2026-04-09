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
    compute_cache_key,
    get_profile_path,
    is_profile_valid,
    load_voice_clone_prompt,
    PROFILES_DIR,
)


# --- voice_clone.py tests ---


class TestComputeCacheKey:
    def test_description_only(self):
        key = compute_cache_key(None, "young woman, warm alto")
        assert len(key) == 16
        assert key == compute_cache_key(None, "young woman, warm alto")

    def test_different_descriptions_differ(self):
        k1 = compute_cache_key(None, "young woman")
        k2 = compute_cache_key(None, "old man")
        assert k1 != k2

    def test_with_audio_file(self, tmp_path):
        audio_file = tmp_path / "ref.wav"
        audio_file.write_bytes(b"fake audio data")
        key = compute_cache_key(str(audio_file), "warm alto")
        assert len(key) == 16

    def test_audio_changes_key(self, tmp_path):
        f1 = tmp_path / "ref1.wav"
        f2 = tmp_path / "ref2.wav"
        f1.write_bytes(b"audio one")
        f2.write_bytes(b"audio two")
        k1 = compute_cache_key(str(f1), "same desc")
        k2 = compute_cache_key(str(f2), "same desc")
        assert k1 != k2

    def test_missing_audio_falls_back(self):
        key = compute_cache_key("/nonexistent/file.wav", "desc")
        key_none = compute_cache_key(None, "desc")
        assert key == key_none


class TestIsProfileValid:
    def test_no_profile(self, tmp_path):
        assert not is_profile_valid("TestVoice", "abc123")

    def test_valid_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path)
        profile = tmp_path / "Aria.voiceprofile"
        keyfile = tmp_path / "Aria.cachekey"
        profile.write_bytes(b"data")
        keyfile.write_text("abc123")
        assert is_profile_valid("Aria", "abc123")

    def test_stale_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path)
        profile = tmp_path / "Aria.voiceprofile"
        keyfile = tmp_path / "Aria.cachekey"
        profile.write_bytes(b"data")
        keyfile.write_text("old_key")
        assert not is_profile_valid("Aria", "new_key")


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

    def generate(self, text, speed=1.0, voice_clone_prompt=None, instruct=None):
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
    async def test_load_description_mode(self, tts):
        with patch("omnivoice.OmniVoice") as MockClass:
            MockClass.from_pretrained.return_value = MockOmniVoice()
            tts._load_sync("description", "warm female voice", None)
        assert tts._model is not None
        assert tts._voice_description == "warm female voice"
        assert tts._voice_clone_prompt is None

    @pytest.mark.asyncio
    async def test_synthesize_returns_f32_array(self, tts):
        tts._model = MockOmniVoice()
        tts._voice_description = "warm voice"
        result = await tts.synthesize("Hello, world!")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_params(self, tts):
        tts._model = MockOmniVoice()
        tts._voice_description = "warm voice"
        result = await tts.synthesize("test", {"speed": 1.1, "tags": ["[laughter]"]})
        assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_synthesize_with_clone_prompt(self, tts):
        tts._model = MockOmniVoice()
        from omnivoice.models.omnivoice import VoiceClonePrompt
        tts._voice_clone_prompt = VoiceClonePrompt(
            ref_audio_tokens=torch.randn(1, 50), ref_text="ref", ref_rms=0.05
        )
        result = await tts.synthesize("cloned speech test")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_shutdown(self, tts):
        tts._model = MockOmniVoice()
        await tts.shutdown()
        assert tts._model is None
        assert tts._voice_clone_prompt is None


def _mock_import(name, *args, **kwargs):
    """Allow normal imports but intercept omnivoice."""
    if name == "omnivoice":
        mock = MagicMock()
        mock.OmniVoice.from_pretrained.return_value = MockOmniVoice()
        return mock
    return original_import(name, *args, **kwargs)


import builtins
original_import = builtins.__import__
