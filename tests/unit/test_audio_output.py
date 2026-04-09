"""Tests for audio output pipeline: Playlist, FillerCache.

PlaybackManager tests are limited to construction — actual audio output
requires a sound device and is tested via the guide's manual verification.
"""
import time
from collections import deque

import numpy as np
import pytest

from voice_assistant.core.audio_output import (
    AudioChunk,
    FillerCache,
    Playlist,
    PlaybackManager,
    OUTPUT_SAMPLE_RATE,
)


# --- Playlist tests ---


class TestPlaylist:
    def test_append_and_read(self):
        pl = Playlist()
        chunk = AudioChunk(audio=np.ones(2400, dtype=np.float32) * 0.5, text="Hello")
        pl.append(chunk)
        samples = pl.read_samples(1200)
        assert samples is not None
        assert len(samples) == 1200
        np.testing.assert_allclose(samples, 0.5)

    def test_read_across_chunks(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32) * 0.3, text="A"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32) * 0.7, text="B"))
        samples = pl.read_samples(150)
        assert len(samples) == 150
        np.testing.assert_allclose(samples[:100], 0.3)
        np.testing.assert_allclose(samples[100:], 0.7)

    def test_read_exhausted_returns_none(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(50, dtype=np.float32), text="short"))
        pl.read_samples(50)
        assert pl.read_samples(10) is None

    def test_empty_playlist_returns_none(self):
        pl = Playlist()
        assert pl.read_samples(100) is None

    def test_text_played_and_remaining(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="Hello"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="World"))
        pl.read_samples(100)  # consume first chunk
        assert pl.text_played() == "Hello"
        assert pl.text_remaining() == "World"

    def test_drop_future(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="A"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="B"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="C"))
        pl.read_samples(50)  # partially into chunk A
        dropped = pl.drop_future()
        assert len(dropped) == 2
        assert [c.text for c in dropped] == ["B", "C"]

    def test_clear(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="A"))
        pl.clear()
        assert pl.is_empty
        assert pl.read_samples(10) is None

    def test_is_empty(self):
        pl = Playlist()
        assert pl.is_empty
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32)))
        assert not pl.is_empty

    def test_partial_read_returns_available(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(50, dtype=np.float32) * 0.5))
        samples = pl.read_samples(200)
        assert len(samples) == 50

    def test_insert_at(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32) * 0.1, text="A"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32) * 0.3, text="C"))
        pl.insert_at(1, AudioChunk(audio=np.ones(100, dtype=np.float32) * 0.2, text="B"))
        assert pl.text_remaining() == "A B C"


# --- FillerCache tests ---


class TestFillerCache:
    def _make_cache(self, n_fillers=10):
        fc = FillerCache()
        for i in range(n_fillers):
            fc.add_filler(f"hmm_{i}", np.random.randn(12000).astype(np.float32) * 0.1)
        return fc

    def test_empty_cache_no_play(self):
        fc = FillerCache()
        assert not fc.should_play()

    def test_turn_rate_limit(self):
        fc = self._make_cache()
        for _ in range(5):
            fc.record_latency(600.0)
        fc.record_turn()
        assert not fc.should_play()  # < 3 turns

    def test_plays_after_enough_turns(self):
        fc = self._make_cache()
        for _ in range(5):
            fc.record_latency(600.0)
        for _ in range(4):
            fc.record_turn()
        assert fc.should_play()

    def test_low_latency_suppresses(self):
        fc = self._make_cache()
        for _ in range(5):
            fc.record_latency(200.0)
        for _ in range(4):
            fc.record_turn()
        assert not fc.should_play()

    def test_get_filler_returns_chunk(self):
        fc = self._make_cache()
        for _ in range(5):
            fc.record_latency(600.0)
        for _ in range(4):
            fc.record_turn()
        filler = fc.get_filler()
        assert filler is not None
        assert isinstance(filler, AudioChunk)
        assert filler.source == "filler"
        assert len(filler.audio) > 0

    def test_get_filler_resets_turn_count(self):
        fc = self._make_cache()
        for _ in range(5):
            fc.record_latency(600.0)
        for _ in range(4):
            fc.record_turn()
        fc.get_filler()
        assert not fc.should_play()  # turn count reset to 0

    def test_recency_avoids_repeats(self):
        fc = FillerCache()
        fc.add_filler("a", np.ones(100, dtype=np.float32))
        fc.add_filler("b", np.ones(100, dtype=np.float32))
        for _ in range(5):
            fc.record_latency(600.0)

        seen = set()
        for _ in range(4):
            for _ in range(4):
                fc.record_turn()
            filler = fc.get_filler()
            if filler:
                seen.add(id(filler.audio))
        # Should have gotten fillers (not None)
        assert len(seen) > 0

    def test_load_from_dir(self, tmp_path):
        # Write a fake .f32 file
        audio = np.ones(2400, dtype=np.float32) * 0.1
        audio.tofile(tmp_path / "hmm.f32")
        fc = FillerCache()
        fc.load_from_dir(str(tmp_path))
        assert len(fc._fillers) == 1
        assert fc._fillers[0][0] == "hmm"

    def test_load_from_nonexistent_dir(self):
        fc = FillerCache()
        fc.load_from_dir("/nonexistent/path")
        assert len(fc._fillers) == 0

    def test_per_minute_rate_limit(self):
        fc = self._make_cache()
        for _ in range(5):
            fc.record_latency(600.0)
        # Get 4 fillers (max per minute)
        for _ in range(4):
            for _ in range(4):
                fc.record_turn()
            filler = fc.get_filler()
            assert filler is not None
        # 5th should be blocked
        for _ in range(4):
            fc.record_turn()
        assert not fc.should_play()


# --- PlaybackManager tests ---


class TestPlaybackManager:
    def test_construction(self):
        pl = Playlist()
        pm = PlaybackManager(pl)
        assert not pm.is_active

    def test_fade_out_sets_remaining(self):
        pl = Playlist()
        pm = PlaybackManager(pl)
        pm.fade_out(30)
        assert pm._fade_remaining == int(OUTPUT_SAMPLE_RATE * 30 / 1000)
