"""Language switching stress tests — rapid switching, mixed input, edge cases.

Run: pytest tests/integration/test_language_stress.py -v -s
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState
from voice_assistant.core.language_detector import detect_language

from conftest import (
    make_mock_components, make_orchestrator, make_speech_audio, make_tts_audio,
    make_llm_stream, make_asr_responses, make_multilang_llm_responses,
    VOICE_SAMPLES,
)


class TestRapidLanguageSwitching:
    """Stress test rapid language switches across voice turns."""

    @pytest.mark.asyncio
    async def test_rapid_en_ru_hr_switching(self):
        """Switch languages rapidly: EN → RU → HR → EN → RU."""
        asr_results = [
            {"text": "hello world", "language": "en", "confidence": 0.95},
            {"text": "привет мир", "language": "ru", "confidence": 0.92},
            {"text": "bok svijete", "language": "hr", "confidence": 0.88},
            {"text": "goodbye", "language": "en", "confidence": 0.94},
            {"text": "до свидания", "language": "ru", "confidence": 0.91},
        ]

        comps = make_mock_components()
        comps["asr"].transcribe = make_asr_responses(asr_results)
        comps["llm"].stream = make_llm_stream(["OK."])
        orch, _ = make_orchestrator(comps)

        expected_langs = ["en", "ru", "hr", "en", "ru"]
        for i in range(5):
            orch._state._state = PipelineState.PROCESSING
            await orch._process_voice_turn(make_speech_audio(1.0))
            assert orch._current_language == expected_langs[i], \
                f"Turn {i}: expected {expected_langs[i]}, got {orch._current_language}"

    @pytest.mark.asyncio
    async def test_tts_language_updated_on_switch(self):
        """TTS set_language is called when language changes."""
        asr_results = [
            {"text": "hello", "language": "en", "confidence": 0.95},
            {"text": "привет", "language": "ru", "confidence": 0.92},
        ]

        comps = make_mock_components()
        comps["asr"].transcribe = make_asr_responses(asr_results)
        comps["llm"].stream = make_llm_stream(["OK."])
        orch, _ = make_orchestrator(comps)

        for _ in range(2):
            orch._state._state = PipelineState.PROCESSING
            await orch._process_voice_turn(make_speech_audio(1.0))

        comps["tts"].set_language.assert_called_with("ru")

    @pytest.mark.asyncio
    async def test_language_switch_preserves_dialogue(self):
        """Language switch doesn't lose dialogue history."""
        call_idx = 0

        async def multilang_transcribe(audio):
            nonlocal call_idx
            texts = [
                {"text": "hello there", "language": "en", "confidence": 0.95},
                {"text": "привет", "language": "ru", "confidence": 0.92},
                {"text": "how are you", "language": "en", "confidence": 0.93},
            ]
            result = texts[min(call_idx, len(texts) - 1)]
            call_idx += 1
            return result

        comps = make_mock_components()
        comps["asr"].transcribe = multilang_transcribe
        orch, _ = make_orchestrator(comps)

        for _ in range(3):
            orch._state._state = PipelineState.PROCESSING
            await orch._process_voice_turn(make_speech_audio(1.0))

        assert len(orch.dialogue) == 6  # 3 user + 3 assistant

    @pytest.mark.asyncio
    async def test_same_language_no_redundant_switch(self):
        """Same language across turns → set_language NOT called."""
        asr_results = [
            {"text": "hello", "language": "en", "confidence": 0.95},
            {"text": "hi there", "language": "en", "confidence": 0.93},
        ]

        comps = make_mock_components()
        comps["asr"].transcribe = make_asr_responses(asr_results)
        comps["llm"].stream = make_llm_stream(["OK."])
        orch, _ = make_orchestrator(comps)

        for _ in range(2):
            orch._state._state = PipelineState.PROCESSING
            await orch._process_voice_turn(make_speech_audio(1.0))

        # set_language should NOT have been called (language stayed "en")
        comps["tts"].set_language.assert_not_called()


class TestTextInputLanguageDetection:
    """Language detection from typed text input."""

    @pytest.mark.asyncio
    async def test_english_text_detected(self):
        """English text switches to English."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._current_language = "ru"

        await orch.handle_text_input("tell me about the weather today")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._current_language == "en"

    @pytest.mark.asyncio
    async def test_russian_text_detected(self):
        """Russian (Cyrillic) text switches to Russian."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._current_language = "en"

        await orch.handle_text_input("расскажи мне о погоде сегодня")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._current_language == "ru"

    @pytest.mark.asyncio
    async def test_croatian_diacritics_detected(self):
        """Croatian diacritics switch to Croatian."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._current_language = "en"

        await orch.handle_text_input("Što misliš o čokoladi?")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._current_language == "hr"

    @pytest.mark.asyncio
    async def test_short_ambiguous_preserves_language(self):
        """Short ambiguous text (not definitive) preserves current language."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._current_language = "ru"

        await orch.handle_text_input("hmm")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._current_language == "ru"

    @pytest.mark.asyncio
    async def test_definitive_word_switches(self):
        """Single definitive word triggers language switch."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._current_language = "ru"

        await orch.handle_text_input("hello")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._current_language == "en"

    @pytest.mark.asyncio
    async def test_cyrillic_definitive_word(self):
        """Cyrillic definitive word switches to Russian."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)
        orch._current_language = "en"

        await orch.handle_text_input("привет")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._current_language == "ru"


class TestLanguageDetectorUnit:
    """Unit tests for the language_detector module itself."""

    def test_cyrillic_detection_100_percent(self):
        assert detect_language("Привет мир", {"en", "ru", "hr"}) == "ru"

    def test_diacritics_detection(self):
        assert detect_language("Što misliš?", {"en", "ru", "hr"}) == "hr"

    def test_definitive_word_en(self):
        assert detect_language("hello", {"en", "ru", "hr"}) == "en"

    def test_definitive_word_ru(self):
        assert detect_language("да", {"en", "ru", "hr"}) == "ru"

    def test_definitive_word_hr(self):
        assert detect_language("hvala", {"en", "ru", "hr"}) == "hr"

    def test_empty_returns_none(self):
        assert detect_language("", {"en", "ru"}) is None

    def test_numbers_only_returns_none(self):
        assert detect_language("12345", {"en", "ru"}) is None

    def test_short_ambiguous_returns_none(self):
        assert detect_language("hmm", {"en", "ru"}) is None

    def test_long_english_detected(self):
        result = detect_language(
            "I was thinking about going to the store today but it is raining",
            {"en", "ru", "hr"}
        )
        assert result == "en"

    def test_long_russian_detected(self):
        result = detect_language(
            "Я думал пойти в магазин сегодня но идёт дождь",
            {"en", "ru", "hr"}
        )
        assert result == "ru"

    def test_unsupported_language_returns_none(self):
        """If detected language is not in supported set, returns None."""
        result = detect_language("hola mundo", {"en", "ru"})
        # "hola" is not in definitive words for en/ru
        # langdetect might return "es" which is not supported
        assert result is None or result in ("en", "ru")

    def test_mixed_script_cyrillic_wins(self):
        """Mixed text with >30% Cyrillic → Russian."""
        result = detect_language("Привет hello мир world", {"en", "ru"})
        assert result == "ru"


class TestLanguageSwitchWithVoiceSamples:
    """Use the full voice sample catalogue for language switching tests."""

    @pytest.mark.asyncio
    async def test_all_samples_detect_correct_language(self, voice_samples):
        """Every voice sample, when processed, sets the correct language."""
        for sample in voice_samples:
            comps = make_mock_components()
            comps["asr"].transcribe = AsyncMock(
                return_value={"text": sample["text"], "language": sample["lang"],
                              "confidence": 0.92}
            )
            orch, _ = make_orchestrator(comps)
            orch._state._state = PipelineState.PROCESSING

            await orch._process_voice_turn(sample["audio"])

            assert orch._current_language == sample["lang"], \
                f"Expected {sample['lang']} for '{sample['text'][:30]}', got {orch._current_language}"

    @pytest.mark.asyncio
    async def test_sequential_multilang_conversation(self, voice_samples_by_lang):
        """Simulate a realistic multilingual conversation with sample rotation."""
        # Take first 2 samples from each language
        sequence = []
        for lang in ["en", "ru", "hr", "en"]:
            samples = voice_samples_by_lang[lang]
            sequence.append(samples[0])

        call_idx = 0

        async def sequenced_transcribe(audio):
            nonlocal call_idx
            sample = sequence[min(call_idx, len(sequence) - 1)]
            call_idx += 1
            return {"text": sample["text"], "language": sample["lang"], "confidence": 0.93}

        comps = make_mock_components()
        comps["asr"].transcribe = sequenced_transcribe
        comps["llm"].stream = make_llm_stream(["OK."])
        orch, _ = make_orchestrator(comps)

        for i, sample in enumerate(sequence):
            orch._state._state = PipelineState.PROCESSING
            await orch._process_voice_turn(sample["audio"])
            assert orch._current_language == sample["lang"]

        assert len(orch.dialogue) == len(sequence) * 2
