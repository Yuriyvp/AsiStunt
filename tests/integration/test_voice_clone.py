"""Integration tests for voice clone cache invalidation.

Verifies: cache key computation, profile validity checking,
cache invalidation on voice description change, re-clone trigger.

Run: pytest tests/integration/test_voice_clone.py -v -s
"""
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_assistant.core.voice_clone import (
    compute_cache_key,
    is_profile_valid,
    get_profile_path,
    PROFILES_DIR,
)


class TestCacheKeyComputation:
    """Cache key = SHA256(file_contents + description)[:16]."""

    def test_same_inputs_same_key(self):
        """Identical inputs produce identical cache keys."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            f.flush()
            key1 = compute_cache_key(f.name, "warm and friendly")
            key2 = compute_cache_key(f.name, "warm and friendly")
        assert key1 == key2
        assert len(key1) == 16

    def test_different_description_different_key(self):
        """Changing voice description changes the cache key."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            f.flush()
            key1 = compute_cache_key(f.name, "warm and friendly")
            key2 = compute_cache_key(f.name, "cold and distant")
        assert key1 != key2

    def test_different_audio_different_key(self):
        """Changing reference audio changes the cache key."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1:
            f1.write(b"audio version 1")
            f1.flush()
            key1 = compute_cache_key(f1.name, "warm")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
            f2.write(b"audio version 2")
            f2.flush()
            key2 = compute_cache_key(f2.name, "warm")

        assert key1 != key2

    def test_no_audio_file(self):
        """Cache key with no audio file uses only description."""
        key1 = compute_cache_key(None, "warm and friendly")
        key2 = compute_cache_key(None, "warm and friendly")
        assert key1 == key2
        assert len(key1) == 16

    def test_missing_audio_file(self):
        """Cache key with non-existent audio path uses only description."""
        key1 = compute_cache_key("/nonexistent/path.wav", "warm")
        key2 = compute_cache_key(None, "warm")
        assert key1 == key2  # both fall back to description-only


class TestProfileValidity:
    """Profile validity checking against cache key."""

    def test_valid_profile(self, tmp_path):
        """Profile is valid when cache key matches stored key."""
        with patch("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path):
            soul_name = "test_persona"
            cache_key = "abcdef1234567890"

            # Create profile and key files
            (tmp_path / f"{soul_name}.voiceprofile").write_bytes(b"fake profile")
            (tmp_path / f"{soul_name}.cachekey").write_text(cache_key)

            assert is_profile_valid(soul_name, cache_key) is True

    def test_invalid_profile_key_mismatch(self, tmp_path):
        """Profile is invalid when cache key doesn't match."""
        with patch("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path):
            soul_name = "test_persona"

            (tmp_path / f"{soul_name}.voiceprofile").write_bytes(b"fake profile")
            (tmp_path / f"{soul_name}.cachekey").write_text("old_key_12345678")

            assert is_profile_valid(soul_name, "new_key_87654321") is False

    def test_invalid_profile_missing_files(self, tmp_path):
        """Profile is invalid when files don't exist."""
        with patch("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path):
            assert is_profile_valid("nonexistent", "anykey") is False

    def test_invalid_profile_missing_key_file(self, tmp_path):
        """Profile is invalid when .cachekey file is missing."""
        with patch("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path):
            soul_name = "test_persona"
            (tmp_path / f"{soul_name}.voiceprofile").write_bytes(b"fake")
            # No .cachekey file

            assert is_profile_valid(soul_name, "anykey") is False


class TestCacheInvalidation:
    """Cache invalidation when voice description changes in SOUL.yaml."""

    def test_description_change_invalidates_cache(self, tmp_path):
        """Changing voice.description in SOUL.yaml should invalidate the cache."""
        with patch("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path):
            soul_name = "assistant"

            # Create a reference audio file
            ref_audio = tmp_path / "ref.wav"
            ref_audio.write_bytes(b"reference audio content")

            # Initial clone with description A
            desc_a = "warm, friendly, slightly husky"
            key_a = compute_cache_key(str(ref_audio), desc_a)
            (tmp_path / f"{soul_name}.voiceprofile").write_bytes(b"profile data")
            (tmp_path / f"{soul_name}.cachekey").write_text(key_a)

            # Profile is valid with original description
            assert is_profile_valid(soul_name, key_a) is True

            # User changes description in SOUL.yaml
            desc_b = "calm, professional, deep"
            key_b = compute_cache_key(str(ref_audio), desc_b)

            # Profile is now invalid with new description
            assert is_profile_valid(soul_name, key_b) is False
            assert key_a != key_b

    def test_audio_change_invalidates_cache(self, tmp_path):
        """Changing reference audio should invalidate the cache."""
        with patch("voice_assistant.core.voice_clone.PROFILES_DIR", tmp_path):
            soul_name = "assistant"
            desc = "warm and friendly"

            # Initial reference audio
            ref_audio_v1 = tmp_path / "ref_v1.wav"
            ref_audio_v1.write_bytes(b"audio version 1")
            key_v1 = compute_cache_key(str(ref_audio_v1), desc)

            (tmp_path / f"{soul_name}.voiceprofile").write_bytes(b"profile v1")
            (tmp_path / f"{soul_name}.cachekey").write_text(key_v1)

            assert is_profile_valid(soul_name, key_v1) is True

            # New reference audio
            ref_audio_v2 = tmp_path / "ref_v2.wav"
            ref_audio_v2.write_bytes(b"audio version 2 - different")
            key_v2 = compute_cache_key(str(ref_audio_v2), desc)

            assert is_profile_valid(soul_name, key_v2) is False


class TestReCloneTrigger:
    """Verify that ProcessManager triggers re-clone when cache is invalid."""

    @pytest.mark.asyncio
    async def test_startup_triggers_reclone_on_mismatch(self, tmp_path):
        """ProcessManager.startup() should trigger voice cloning when cache key mismatches."""
        from voice_assistant.process.manager import ProcessManager, LlamaCppProcess

        ipc = MagicMock()
        ipc.emit_state_change = MagicMock()
        ipc.emit_signal = MagicMock()
        ipc.emit_error = MagicMock()

        vram = MagicMock()
        vram.check = MagicMock()

        soul = MagicMock()
        soul.name = "test_persona"
        soul.voice_method = "clone"
        soul.voice_reference_audio = str(tmp_path / "ref.wav")
        soul.voice_description = "new description that differs"
        soul.llm_model = "models/test.gguf"
        soul.llm_port = 8080
        soul.llm_ctx_size = 8192
        soul.llm_gpu_layers = 999
        soul.llm_threads = 4
        soul.llm_batch_size = 4096
        soul.llm_flash_attn = True

        # Create reference audio
        (tmp_path / "ref.wav").write_bytes(b"test audio")

        pm = ProcessManager(soul, vram, ipc)

        clone_called = False

        async def mock_clone(*args, **kwargs):
            nonlocal clone_called
            clone_called = True
            return tmp_path / "test_persona.voiceprofile"

        with patch("voice_assistant.core.voice_clone.is_profile_valid", return_value=False), \
             patch("voice_assistant.core.voice_clone.run_voice_cloning", mock_clone), \
             patch("voice_assistant.core.voice_clone.compute_cache_key", return_value="newkey123"), \
             patch.object(LlamaCppProcess, 'start', new_callable=AsyncMock), \
             patch.object(LlamaCppProcess, '__init__', return_value=None):
            await pm.startup()

        assert clone_called, "Voice cloning should have been triggered"
        ipc.emit_signal.assert_any_call("voice_clone_progress", status="starting")

    @pytest.mark.asyncio
    async def test_startup_skips_clone_when_cache_valid(self, tmp_path):
        """ProcessManager.startup() should skip cloning when cache is valid."""
        from voice_assistant.process.manager import ProcessManager, LlamaCppProcess

        ipc = MagicMock()
        ipc.emit_state_change = MagicMock()
        ipc.emit_signal = MagicMock()
        ipc.emit_error = MagicMock()

        vram = MagicMock()

        soul = MagicMock()
        soul.name = "test_persona"
        soul.voice_method = "clone"
        soul.voice_reference_audio = str(tmp_path / "ref.wav")
        soul.voice_description = "unchanged description"
        soul.llm_model = "models/test.gguf"
        soul.llm_port = 8080
        soul.llm_ctx_size = 8192
        soul.llm_gpu_layers = 999
        soul.llm_threads = 4
        soul.llm_batch_size = 4096
        soul.llm_flash_attn = True

        (tmp_path / "ref.wav").write_bytes(b"test audio")

        pm = ProcessManager(soul, vram, ipc)

        with patch("voice_assistant.core.voice_clone.is_profile_valid", return_value=True), \
             patch("voice_assistant.core.voice_clone.run_voice_cloning") as mock_clone, \
             patch("voice_assistant.core.voice_clone.compute_cache_key", return_value="validkey"), \
             patch.object(LlamaCppProcess, 'start', new_callable=AsyncMock), \
             patch.object(LlamaCppProcess, '__init__', return_value=None):
            await pm.startup()

        mock_clone.assert_not_called()


class TestGetProfilePath:
    """Verify profile path generation."""

    def test_profile_path(self):
        path = get_profile_path("my_assistant")
        assert path == PROFILES_DIR / "my_assistant.voiceprofile"
        assert str(path).endswith(".voiceprofile")
