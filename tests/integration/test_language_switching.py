"""Integration tests for language switching.

Verifies: language detection from ASR, language tracking in orchestrator,
multi-language dialogue, language switch across turns.

Run: pytest tests/integration/test_language_switching.py -v -s
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState


def _make_tts_audio(duration_s: float = 0.5, sr: int = 24000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _make_audio(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _make_components(**overrides):
    audio = AsyncMock()
    audio.start = AsyncMock()
    audio.stop = AsyncMock()
    audio.read_chunk = AsyncMock(return_value=np.zeros(480, dtype=np.float32))

    vad = MagicMock()
    vad.process_chunk = MagicMock(return_value=None)
    vad.is_speech = False
    vad.reset = MagicMock()
    vad.drain_speech_samples = MagicMock(return_value=None)

    asr = AsyncMock()
    asr.transcribe = AsyncMock(
        return_value={"text": "hello", "language": "en", "confidence": 0.95}
    )

    llm = AsyncMock()

    async def _stream(messages, sampling=None, **kwargs):
        for token in ["Response", "."]:
            yield token
    llm.stream = _stream
    llm.cancel = AsyncMock()

    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=_make_tts_audio())
    tts.set_language = MagicMock()

    playlist = MagicMock()
    playlist.append = MagicMock()
    playlist.drop_future = MagicMock(return_value=[])
    playlist.clear = MagicMock()
    playlist.text_played = MagicMock(return_value="")
    playlist.text_remaining = MagicMock(return_value="")
    playlist.is_empty = True
    playlist.wait_until_done = AsyncMock()

    playback = AsyncMock()
    playback.start = AsyncMock()
    playback.stop = AsyncMock()
    playback.fade_out = MagicMock()
    playback.is_active = False

    filler_cache = MagicMock()
    filler_cache.record_turn = MagicMock()
    filler_cache.get_filler = MagicMock(return_value=None)

    components = {
        "audio_input": audio,
        "vad": vad,
        "asr": asr,
        "llm": llm,
        "tts": tts,
        "playlist": playlist,
        "playback": playback,
        "filler_cache": filler_cache,
    }
    components.update(overrides)
    return components


class TestLanguageDetection:
    """Language detection from ASR updates orchestrator state."""

    @pytest.mark.asyncio
    async def test_english_detection(self):
        """ASR returns English -> orchestrator tracks 'en'."""
        components = _make_components()
        components["asr"].transcribe = AsyncMock(
            return_value={"text": "hello world", "language": "en", "confidence": 0.95}
        )

        orch = Orchestrator(**components)
        orch._state._state = PipelineState.PROCESSING

        await orch._process_voice_turn(_make_audio())
        if orch._processing_task and not orch._processing_task.done():
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._current_language == "en"

    @pytest.mark.asyncio
    async def test_croatian_detection(self):
        """ASR returns Croatian -> orchestrator tracks 'hr'."""
        components = _make_components()
        components["asr"].transcribe = AsyncMock(
            return_value={"text": "dobar dan", "language": "hr", "confidence": 0.88}
        )

        orch = Orchestrator(**components)
        orch._state._state = PipelineState.PROCESSING

        await orch._process_voice_turn(_make_audio())

        assert orch._current_language == "hr"

    @pytest.mark.asyncio
    async def test_german_detection(self):
        """ASR returns German -> orchestrator tracks 'de'."""
        components = _make_components()
        components["asr"].transcribe = AsyncMock(
            return_value={"text": "guten tag", "language": "de", "confidence": 0.92}
        )

        orch = Orchestrator(**components)
        orch._state._state = PipelineState.PROCESSING

        await orch._process_voice_turn(_make_audio())

        assert orch._current_language == "de"


class TestLanguageSwitching:
    """Language switches between turns."""

    @pytest.mark.asyncio
    async def test_english_to_croatian_switch(self):
        """Switch from English to Croatian across voice turns."""
        call_count = 0
        asr_responses = [
            {"text": "hello", "language": "en", "confidence": 0.95},
            {"text": "kako si", "language": "hr", "confidence": 0.90},
        ]

        async def switching_transcribe(audio):
            nonlocal call_count
            result = asr_responses[min(call_count, len(asr_responses) - 1)]
            call_count += 1
            return result

        components = _make_components()
        components["asr"].transcribe = switching_transcribe

        orch = Orchestrator(**components)

        # First turn: English
        orch._state._state = PipelineState.PROCESSING
        await orch._process_voice_turn(_make_audio())
        assert orch._current_language == "en"

        # Second turn: Croatian
        orch._state._state = PipelineState.PROCESSING
        await orch._process_voice_turn(_make_audio())
        assert orch._current_language == "hr"

    @pytest.mark.asyncio
    async def test_language_detected_on_text_input(self):
        """Text input detects language and updates current language."""
        components = _make_components()
        orch = Orchestrator(**components, supported_languages=["hr", "en", "ru"])
        orch._current_language = "hr"

        await orch.handle_text_input("test in english is short")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Text input now detects language — long enough English text switches to "en"
        assert orch._current_language == "en"

    @pytest.mark.asyncio
    async def test_definitive_word_switches_language(self):
        """Short definitive words like 'ok' should switch the language."""
        components = _make_components()
        orch = Orchestrator(**components, supported_languages=["hr", "en", "ru"])
        orch._current_language = "hr"

        await orch.handle_text_input("ok")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # "ok" is a definitive English word — language switches
        assert orch._current_language == "en"

    @pytest.mark.asyncio
    async def test_language_preserved_on_ambiguous_short_text(self):
        """Short ambiguous text (not a definitive word) preserves language."""
        components = _make_components()
        orch = Orchestrator(**components, supported_languages=["hr", "en", "ru"])
        orch._current_language = "hr"

        await orch.handle_text_input("hmm")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # "hmm" is not a definitive word — language preserved
        assert orch._current_language == "hr"

    @pytest.mark.asyncio
    async def test_roundtrip_language_switch(self):
        """English -> Croatian -> English roundtrip."""
        call_count = 0
        asr_responses = [
            {"text": "hello", "language": "en", "confidence": 0.95},
            {"text": "zdravo", "language": "hr", "confidence": 0.88},
            {"text": "goodbye", "language": "en", "confidence": 0.93},
        ]

        async def roundtrip_transcribe(audio):
            nonlocal call_count
            result = asr_responses[min(call_count, len(asr_responses) - 1)]
            call_count += 1
            return result

        components = _make_components()
        components["asr"].transcribe = roundtrip_transcribe

        orch = Orchestrator(**components)

        for _ in range(3):
            orch._state._state = PipelineState.PROCESSING
            await orch._process_voice_turn(_make_audio())

        assert orch._current_language == "en"
        # Should have 6 turns (3 user + 3 assistant)
        assert len(orch.dialogue) == 6

    @pytest.mark.asyncio
    async def test_language_fallback_on_missing(self):
        """If ASR doesn't return language, keep previous."""
        components = _make_components()
        components["asr"].transcribe = AsyncMock(
            return_value={"text": "something", "confidence": 0.7}  # no language key
        )

        orch = Orchestrator(**components)
        orch._current_language = "de"
        orch._state._state = PipelineState.PROCESSING

        await orch._process_voice_turn(_make_audio())

        # Should keep the previous language
        assert orch._current_language == "de"


class TestLanguageInDialogue:
    """Verify language context flows through the pipeline."""

    @pytest.mark.asyncio
    async def test_dialogue_contains_multilingual_turns(self):
        """Multiple languages appear in dialogue from different voice turns."""
        call_count = 0

        async def multilang_transcribe(audio):
            nonlocal call_count
            texts = [
                {"text": "hello there", "language": "en", "confidence": 0.95},
                {"text": "wie geht es dir", "language": "de", "confidence": 0.91},
            ]
            result = texts[min(call_count, len(texts) - 1)]
            call_count += 1
            return result

        turn_idx = 0

        async def lang_aware_stream(messages, sampling=None, **kwargs):
            nonlocal turn_idx
            responses = [
                ["I'm ", "fine ", "thanks!"],
                ["Mir ", "geht ", "es ", "gut!"],
            ]
            for token in responses[min(turn_idx, len(responses) - 1)]:
                yield token
            turn_idx += 1

        components = _make_components()
        components["asr"].transcribe = multilang_transcribe
        components["llm"].stream = lang_aware_stream

        orch = Orchestrator(**components)

        for _ in range(2):
            orch._state._state = PipelineState.PROCESSING
            await orch._process_voice_turn(_make_audio())

        assert len(orch.dialogue) == 4
        assert orch.dialogue[0].content == "hello there"
        assert orch.dialogue[2].content == "wie geht es dir"
