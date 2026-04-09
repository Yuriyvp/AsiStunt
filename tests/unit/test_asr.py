"""Tests for ParakeetASR — Parakeet TDT v3 INT8 via sherpa-onnx."""
import asyncio
import wave

import numpy as np
import pytest

from voice_assistant.adapters.parakeet_asr import ParakeetASR

MODEL_DIR = "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
EN_WAV = "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/en.wav"
DE_WAV = "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/de.wav"


def _load_wav_16k(path: str) -> np.ndarray:
    """Load a WAV file and resample to 16kHz float32 mono."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != 16000:
        indices = np.linspace(0, len(samples) - 1, int(len(samples) * 16000 / rate))
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
    return samples


@pytest.fixture(scope="module")
def asr():
    """Module-scoped ASR — model load is expensive (~2s)."""
    return ParakeetASR(model_dir=MODEL_DIR, supported_languages=["en", "de", "hr"])


@pytest.fixture
def en_audio():
    return _load_wav_16k(EN_WAV)


@pytest.fixture
def de_audio():
    return _load_wav_16k(DE_WAV)


def test_asr_loads(asr):
    """ASR initializes without error."""
    assert asr is not None


@pytest.mark.asyncio
async def test_transcribe_silence(asr):
    """Silence should return empty or near-empty transcript."""
    silence = np.zeros(16000, dtype=np.float32)  # 1 second
    result = await asr.transcribe(silence)
    assert "text" in result
    assert "language" in result
    assert len(result["text"]) < 10, f"Expected near-empty, got: {result['text']}"


@pytest.mark.asyncio
async def test_transcribe_english(asr, en_audio):
    """English speech should produce non-empty English transcript."""
    result = await asr.transcribe(en_audio)
    assert len(result["text"]) > 5, "Expected non-trivial transcript"
    assert result["language"] == "en"
    assert result["rtf"] > 0
    assert result["duration_s"] > 0


@pytest.mark.asyncio
async def test_transcribe_german(asr, de_audio):
    """German speech should produce non-empty German transcript."""
    result = await asr.transcribe(de_audio)
    assert len(result["text"]) > 5, "Expected non-trivial transcript"
    assert result["language"] == "de"


@pytest.mark.asyncio
async def test_rtf_reasonable(asr, en_audio):
    """RTF should be reasonable (<1.0 on modern hardware)."""
    result = await asr.transcribe(en_audio)
    assert result["rtf"] < 1.0, f"RTF too high: {result['rtf']}"
