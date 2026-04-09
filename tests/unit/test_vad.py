"""Tests for SileroVAD — Silero VAD via sherpa-onnx."""
import wave

import numpy as np
import pytest

from voice_assistant.models.silero_vad import SileroVAD, VADEvent

MODEL_PATH = "models/silero_vad.onnx"
SPEECH_WAV = "tests/fixtures/audio_samples/speech_en.wav"


def _load_wav_16k(path: str) -> np.ndarray:
    """Load a WAV file and resample to 16kHz float32 mono."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    # Convert int16 to float32
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != 16000:
        # Simple linear interpolation resample
        indices = np.linspace(0, len(samples) - 1, int(len(samples) * 16000 / rate))
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
    return samples


@pytest.fixture
def vad():
    return SileroVAD(model_path=MODEL_PATH)


@pytest.fixture
def speech_audio():
    return _load_wav_16k(SPEECH_WAV)


def test_vad_loads(vad):
    """VAD initializes without error."""
    assert vad is not None
    assert vad.is_speech is False


def test_silence_no_false_triggers(vad):
    """100 silent chunks should not trigger speech."""
    events = []
    vad.on_event(events.append)
    silence = np.zeros(480, dtype=np.float32)
    for _ in range(100):
        vad.process_chunk(silence)
    assert len(events) == 0, f"Got {len(events)} unexpected events on silence"


def test_speech_triggers_start(vad, speech_audio):
    """Real speech audio should trigger speech_start."""
    events = []
    vad.on_event(events.append)

    for i in range(0, len(speech_audio) - 480, 480):
        vad.process_chunk(speech_audio[i:i + 480])

    speech_starts = [e for e in events if e.type == "speech_start"]
    assert len(speech_starts) >= 1, "Expected speech_start on real speech"


def test_speech_then_silence_triggers_end(vad, speech_audio):
    """Speech followed by silence should produce speech_start then speech_end."""
    events = []
    vad.on_event(events.append)

    # Feed speech
    for i in range(0, len(speech_audio) - 480, 480):
        vad.process_chunk(speech_audio[i:i + 480])

    # Feed 1.5s of silence (enough for min_silence_duration=300ms)
    silence = np.zeros(480, dtype=np.float32)
    for _ in range(50):
        vad.process_chunk(silence)

    types = [e.type for e in events]
    assert "speech_start" in types, "Expected speech_start"
    assert "speech_end" in types, f"Expected speech_end, got: {types}"


def test_reset_clears_state(vad, speech_audio):
    """Reset should clear speech state."""
    # Feed some speech to potentially trigger
    for i in range(0, min(len(speech_audio), 8000) - 480, 480):
        vad.process_chunk(speech_audio[i:i + 480])

    vad.reset()
    assert vad.is_speech is False


def test_callback_receives_vad_event(vad, speech_audio):
    """Callback receives proper VADEvent objects."""
    events: list[VADEvent] = []
    vad.on_event(events.append)

    # Feed speech then silence
    for i in range(0, len(speech_audio) - 480, 480):
        vad.process_chunk(speech_audio[i:i + 480])

    silence = np.zeros(480, dtype=np.float32)
    for _ in range(50):
        vad.process_chunk(silence)

    for event in events:
        assert isinstance(event, VADEvent)
        assert event.type in ("speech_start", "speech_end")
        assert event.timestamp > 0

    end_events = [e for e in events if e.type == "speech_end"]
    if end_events:
        assert end_events[0].duration_ms > 0
