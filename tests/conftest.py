"""Shared test fixtures and voice sample factory.

Provides:
- Realistic synthetic voice samples (15-20 samples across 3 languages)
- Common component mock factories
- Orchestrator builder fixture
- Audio generation helpers
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState, PipelineMode
from voice_assistant.core.audio_output import AudioChunk


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE_INPUT = 16000   # mic / ASR / VAD
SAMPLE_RATE_OUTPUT = 24000  # TTS / playback


def make_speech_audio(duration_s: float = 1.0, freq: float = 440.0,
                      sr: int = SAMPLE_RATE_INPUT, amplitude: float = 0.3) -> np.ndarray:
    """Generate a sine tone simulating speech audio at mic sample rate."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_tts_audio(duration_s: float = 0.5, sr: int = SAMPLE_RATE_OUTPUT) -> np.ndarray:
    """Generate silent audio at TTS sample rate."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def make_tts_audio_realistic(duration_s: float = 0.5, freq: float = 220.0,
                             sr: int = SAMPLE_RATE_OUTPUT) -> np.ndarray:
    """Generate a realistic-sounding TTS output with tone + envelope."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    # Apply fade-in/out envelope
    fade = min(int(sr * 0.02), len(audio) // 4)
    if fade > 0:
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Voice sample catalogue — 20 realistic samples across 3 languages
# ---------------------------------------------------------------------------

VOICE_SAMPLES = [
    # English — varied lengths and complexity
    {"lang": "en", "text": "Hello, how are you today?", "duration_s": 1.2, "freq": 440},
    {"lang": "en", "text": "Tell me something interesting about space.", "duration_s": 1.8, "freq": 420},
    {"lang": "en", "text": "Yeah.", "duration_s": 0.3, "freq": 460},
    {"lang": "en", "text": "OK.", "duration_s": 0.2, "freq": 450},
    {"lang": "en", "text": "I've been thinking about consciousness lately. Do you think machines can truly understand?", "duration_s": 3.5, "freq": 430},
    {"lang": "en", "text": "Sure, that sounds great!", "duration_s": 0.9, "freq": 440},
    {"lang": "en", "text": "Can you explain quantum computing in simple terms?", "duration_s": 2.2, "freq": 435},
    # Russian — varied lengths
    {"lang": "ru", "text": "Привет, как у тебя дела сегодня?", "duration_s": 1.5, "freq": 380},
    {"lang": "ru", "text": "Расскажи мне что-нибудь интересное про космос.", "duration_s": 2.0, "freq": 370},
    {"lang": "ru", "text": "Да.", "duration_s": 0.2, "freq": 390},
    {"lang": "ru", "text": "Конечно.", "duration_s": 0.4, "freq": 385},
    {"lang": "ru", "text": "Что ты думаешь о будущем технологий и искусственного интеллекта?", "duration_s": 2.8, "freq": 375},
    {"lang": "ru", "text": "Опиши идеальный день.", "duration_s": 1.0, "freq": 380},
    # Croatian — varied lengths
    {"lang": "hr", "text": "Bok, kako si danas?", "duration_s": 1.0, "freq": 410},
    {"lang": "hr", "text": "Reci mi nešto zanimljivo o Hrvatskoj.", "duration_s": 1.6, "freq": 405},
    {"lang": "hr", "text": "Da.", "duration_s": 0.2, "freq": 415},
    {"lang": "hr", "text": "Što misliš o umjetnoj inteligenciji?", "duration_s": 1.8, "freq": 400},
    # Mixed / edge cases
    {"lang": "en", "text": "What?", "duration_s": 0.3, "freq": 470},
    {"lang": "en", "text": "Let's talk about something completely different. I find bioluminescent deep-sea creatures fascinating.", "duration_s": 4.0, "freq": 425},
    {"lang": "ru", "text": "А теперь давай поговорим на русском о классической литературе.", "duration_s": 2.5, "freq": 365},
]


@pytest.fixture
def voice_samples():
    """Return all 20 voice samples with generated audio."""
    samples = []
    for s in VOICE_SAMPLES:
        audio = make_speech_audio(s["duration_s"], s["freq"])
        samples.append({**s, "audio": audio})
    return samples


@pytest.fixture
def voice_samples_by_lang(voice_samples):
    """Return voice samples grouped by language."""
    by_lang = {"en": [], "ru": [], "hr": []}
    for s in voice_samples:
        by_lang.setdefault(s["lang"], []).append(s)
    return by_lang


# ---------------------------------------------------------------------------
# Mock component factories
# ---------------------------------------------------------------------------

def make_mock_playlist():
    """Create a fully mocked playlist with working state tracking."""
    pl = MagicMock()
    pl._chunks = []
    pl._played_text = []

    def _append(chunk):
        pl._chunks.append(chunk)
        pl._done = False

    def _clear():
        pl._chunks.clear()
        pl._played_text.clear()

    def _text_played():
        return " ".join(pl._played_text)

    def _text_remaining():
        return " ".join(c.text for c in pl._chunks if c.text)

    pl.append = MagicMock(side_effect=_append)
    pl.drop_future = MagicMock(return_value=[])
    pl.clear = MagicMock(side_effect=_clear)
    pl.text_played = MagicMock(side_effect=_text_played)
    pl.text_remaining = MagicMock(side_effect=_text_remaining)
    pl.is_empty = True
    pl.wait_until_done = AsyncMock()
    return pl


def make_mock_components(**overrides):
    """Build a full mock component set for Orchestrator."""
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
        return_value={"text": "hello world", "language": "en", "confidence": 0.95}
    )

    llm = AsyncMock()

    async def _default_stream(messages, sampling=None, **kwargs):
        for token in ["Hello", ", ", "how ", "are ", "you", "?"]:
            yield token
    llm.stream = _default_stream
    llm.cancel = AsyncMock()

    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value=make_tts_audio())
    tts.set_language = MagicMock()

    playlist = make_mock_playlist()

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


def make_orchestrator(components=None, **kwargs):
    """Create an Orchestrator with mock components."""
    if components is None:
        components = make_mock_components()
    defaults = {
        "system_prompt": "You are a test assistant.",
        "default_language": "en",
        "supported_languages": ["en", "ru", "hr"],
    }
    defaults.update(kwargs)
    return Orchestrator(**components, **defaults), components


# ---------------------------------------------------------------------------
# LLM stream factories
# ---------------------------------------------------------------------------

def make_llm_stream(tokens, delay=0.0):
    """Create an async LLM stream generator from a token list."""
    async def stream(messages, sampling=None, **kwargs):
        for token in tokens:
            if delay > 0:
                await asyncio.sleep(delay)
            yield token
    return stream


def make_mood_stream(mood_tone, intensity, response_tokens):
    """Create an LLM stream that starts with a mood signal tag."""
    async def stream(messages, sampling=None, **kwargs):
        yield f"<mood_signal>user_tone={mood_tone}, intensity={intensity}</mood_signal>"
        for token in response_tokens:
            yield token
    return stream


def make_multilang_llm_responses(responses_by_turn):
    """Create a multi-turn LLM stream that varies by turn index."""
    turn_idx = 0

    async def stream(messages, sampling=None, **kwargs):
        nonlocal turn_idx
        tokens = responses_by_turn[min(turn_idx, len(responses_by_turn) - 1)]
        for token in tokens:
            yield token
        turn_idx += 1

    return stream


def make_asr_responses(responses):
    """Create a multi-call ASR mock that returns different results per call."""
    call_count = 0

    async def transcribe(audio):
        nonlocal call_count
        result = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return result

    return transcribe
