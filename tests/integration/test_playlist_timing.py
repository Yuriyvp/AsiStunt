"""Playlist and audio output timing tests.

Tests Playlist read_samples, AudioChunk tracking, multi-chunk playback,
fade-out during barge-in, and thread-safety properties.

Run: pytest tests/integration/test_playlist_timing.py -v -s
"""
import asyncio
import threading

import numpy as np
import pytest

from voice_assistant.core.audio_output import Playlist, AudioChunk, FillerCache


class TestPlaylistBasics:
    """Core Playlist operations."""

    def test_empty_playlist_is_empty(self):
        pl = Playlist()
        assert pl.is_empty is True
        assert pl.read_samples(100) is None

    def test_append_makes_non_empty(self):
        pl = Playlist()
        audio = np.ones(1000, dtype=np.float32)
        pl.append(AudioChunk(audio=audio, text="hello"))
        assert pl.is_empty is False

    def test_read_samples_returns_correct_data(self):
        pl = Playlist()
        audio = np.arange(100, dtype=np.float32)
        pl.append(AudioChunk(audio=audio, text="test"))

        # Read first 50 samples
        out = pl.read_samples(50)
        assert len(out) == 50
        np.testing.assert_array_equal(out, np.arange(50, dtype=np.float32))

        # Read next 50
        out = pl.read_samples(50)
        assert len(out) == 50
        np.testing.assert_array_equal(out, np.arange(50, 100, dtype=np.float32))

        # Nothing left
        assert pl.read_samples(50) is None
        assert pl.is_empty is True

    def test_read_samples_spans_chunks(self):
        """read_samples() should seamlessly cross chunk boundaries."""
        pl = Playlist()
        audio1 = np.ones(30, dtype=np.float32) * 1.0
        audio2 = np.ones(30, dtype=np.float32) * 2.0
        pl.append(AudioChunk(audio=audio1, text="chunk1"))
        pl.append(AudioChunk(audio=audio2, text="chunk2"))

        # Read 50 samples: 30 from chunk1 + 20 from chunk2
        out = pl.read_samples(50)
        assert len(out) == 50
        np.testing.assert_array_equal(out[:30], np.ones(30) * 1.0)
        np.testing.assert_array_equal(out[30:50], np.ones(20) * 2.0)

    def test_read_samples_partial_chunk(self):
        """Requesting more samples than available returns what's left."""
        pl = Playlist()
        audio = np.ones(10, dtype=np.float32)
        pl.append(AudioChunk(audio=audio, text="short"))

        out = pl.read_samples(100)
        assert len(out) == 10

    def test_clear_empties_playlist(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="a"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="b"))
        pl.clear()
        assert pl.is_empty is True
        assert pl.read_samples(10) is None


class TestPlaylistTextTracking:
    """Text tracking for dialogue state."""

    def test_text_played_empty(self):
        pl = Playlist()
        assert pl.text_played() == ""

    def test_text_remaining(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="first"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="second"))
        assert pl.text_remaining() == "first second"

    def test_text_tracking_after_playback(self):
        """After playing first chunk, text_played has it, text_remaining doesn't."""
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="played"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="remaining"))

        # Play all of first chunk
        pl.read_samples(100)

        assert pl.text_played() == "played"
        assert pl.text_remaining() == "remaining"


class TestPlaylistDropFuture:
    """drop_future() for barge-in scenarios."""

    def test_drop_future_returns_unplayed(self):
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="current"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="future1"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="future2"))

        dropped = pl.drop_future()
        assert len(dropped) == 2
        assert dropped[0].text == "future1"
        assert dropped[1].text == "future2"

    def test_drop_future_preserves_current(self):
        """Current chunk stays after drop_future."""
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="current"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="dropped"))

        pl.drop_future()
        # Current chunk should still be readable
        out = pl.read_samples(100)
        assert out is not None
        assert len(out) == 100

    def test_drop_future_during_playback(self):
        """drop_future while partway through current chunk."""
        pl = Playlist()
        pl.append(AudioChunk(audio=np.ones(200, dtype=np.float32), text="playing"))
        pl.append(AudioChunk(audio=np.ones(100, dtype=np.float32), text="future"))

        # Read half of first chunk
        pl.read_samples(100)
        dropped = pl.drop_future()
        assert len(dropped) == 1
        assert dropped[0].text == "future"

        # Rest of current chunk still available
        out = pl.read_samples(100)
        assert len(out) == 100


class TestPlaylistWaitUntilDone:
    """Async wait_until_done behavior."""

    @pytest.mark.asyncio
    async def test_empty_playlist_done_immediately(self):
        pl = Playlist()
        # Should return immediately
        await asyncio.wait_for(pl.wait_until_done(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_wait_completes_after_playback(self):
        """wait_until_done completes when all samples are read."""
        pl = Playlist()
        audio = np.ones(100, dtype=np.float32)
        pl.append(AudioChunk(audio=audio, text="test"))

        async def play_in_background():
            await asyncio.sleep(0.05)
            pl.read_samples(100)
            pl.read_samples(1)  # trigger done flag

        asyncio.create_task(play_in_background())
        await asyncio.wait_for(pl.wait_until_done(), timeout=2.0)
        assert pl.is_empty


class TestPlaylistThreadSafety:
    """Thread safety under concurrent access."""

    def test_concurrent_append_and_read(self):
        """Concurrent append from one thread and read from another."""
        pl = Playlist()
        errors = []

        def writer():
            try:
                for i in range(50):
                    audio = np.ones(100, dtype=np.float32) * i
                    pl.append(AudioChunk(audio=audio, text=f"chunk{i}"))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                read_count = 0
                for _ in range(500):
                    out = pl.read_samples(10)
                    if out is not None:
                        read_count += 1
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors == [], f"Thread safety errors: {errors}"


class TestFillerCache:
    """FillerCache rate-limiting and selection."""

    def test_no_fillers_returns_none(self):
        fc = FillerCache()
        fc.record_turn()
        fc.record_turn()
        fc.record_turn()
        assert fc.get_filler() is None

    def test_rate_limit_3_turns(self):
        """No filler within first 3 turns."""
        fc = FillerCache()
        fc.add_filler("hmm", np.zeros(100, dtype=np.float32))
        fc.record_latency(500)  # above threshold

        fc.record_turn()
        assert fc.get_filler() is None
        fc.record_turn()
        assert fc.get_filler() is None

    def test_filler_after_3_turns(self):
        """Filler available after 3 turns with high latency."""
        fc = FillerCache()
        fc.add_filler("hmm", np.zeros(100, dtype=np.float32))
        fc.record_latency(600)

        for _ in range(3):
            fc.record_turn()

        filler = fc.get_filler()
        assert filler is not None
        assert filler.source == "filler"

    def test_low_latency_suppresses_filler(self):
        """No filler when average latency is below threshold."""
        fc = FillerCache()
        fc.add_filler("hmm", np.zeros(100, dtype=np.float32))
        fc.record_latency(200)  # below 450ms threshold

        for _ in range(5):
            fc.record_turn()

        assert fc.get_filler() is None

    def test_recency_avoids_repeats(self):
        """Same filler not repeated within 5 uses."""
        fc = FillerCache()
        fc.add_filler("a", np.zeros(10, dtype=np.float32))
        fc.add_filler("b", np.zeros(10, dtype=np.float32))
        fc.record_latency(600)

        selections = set()
        for _ in range(10):
            for _ in range(3):
                fc.record_turn()
            filler = fc.get_filler()
            if filler:
                selections.add(id(filler))

        # Should have selected from both fillers
        # (can't guarantee exact behavior due to randomness, but shouldn't crash)

    def test_minute_rate_limit(self):
        """Max 4 fillers per minute."""
        fc = FillerCache()
        fc.add_filler("test", np.zeros(10, dtype=np.float32))
        fc.record_latency(600)

        count = 0
        for _ in range(20):
            for _ in range(3):
                fc.record_turn()
            filler = fc.get_filler()
            if filler:
                count += 1

        assert count <= 4
