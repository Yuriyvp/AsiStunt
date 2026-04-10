"""Integration tests for voice clone profile management.

Verifies: profile path generation, profile loading, profile directory structure.

Run: pytest tests/integration/test_voice_clone.py -v -s
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from voice_assistant.core.voice_clone import (
    get_profile_path,
    load_voice_clone_prompt,
    PROFILES_DIR,
)


class TestGetProfilePath:
    """Verify profile path generation."""

    def test_profile_path(self):
        path = get_profile_path("my_assistant")
        assert path == PROFILES_DIR / "my_assistant.voiceprofile"
        assert str(path).endswith(".voiceprofile")

    def test_per_language_profile_path(self):
        path = get_profile_path("Aria_en")
        assert path == PROFILES_DIR / "Aria_en.voiceprofile"

    def test_profile_path_with_special_chars(self):
        path = get_profile_path("Aria_hr")
        assert "Aria_hr" in str(path)


class TestLoadVoiceClonePrompt:
    """Verify profile loading from disk."""

    def test_load_valid_profile(self, tmp_path):
        """Load a valid .voiceprofile file."""
        profile_path = tmp_path / "test.voiceprofile"
        torch.save(
            {
                "ref_audio_tokens": torch.randn(8, 100),
                "ref_text": "Hello, this is a test.",
                "ref_rms": 0.05,
            },
            profile_path,
        )
        prompt = load_voice_clone_prompt(profile_path)
        assert prompt.ref_audio_tokens.shape == (8, 100)
        assert prompt.ref_text == "Hello, this is a test."
        assert prompt.ref_rms == pytest.approx(0.05)

    def test_load_missing_file_raises(self):
        """Loading a non-existent file should raise."""
        with pytest.raises(FileNotFoundError):
            load_voice_clone_prompt("/nonexistent/path.voiceprofile")

    def test_load_corrupted_file_raises(self, tmp_path):
        """Loading a corrupted file should raise."""
        bad_file = tmp_path / "bad.voiceprofile"
        bad_file.write_bytes(b"not a torch file")
        with pytest.raises(Exception):
            load_voice_clone_prompt(bad_file)
