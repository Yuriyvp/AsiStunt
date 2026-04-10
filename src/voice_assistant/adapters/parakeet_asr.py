"""Parakeet TDT 0.6B v3 INT8 ASR adapter via sherpa-onnx.

Batch/offline transcription — runs on CPU only.
VAD-gated: only called when speech_end fires, receives the speech audio segment.
RTF ~0.15–0.35 on Zen 3.

Uses sherpa_onnx.OfflineRecognizer.from_transducer() factory method.
Language detection via multi-strategy text analysis (Parakeet v3 transcribes
25 European languages but the TDT transducer doesn't predict language tokens).
"""
import asyncio
import logging
import time

import numpy as np

from voice_assistant.core.language_detector import detect_language
from voice_assistant.ports.asr import ASRPort

logger = logging.getLogger(__name__)


class ParakeetASR(ASRPort):
    """Offline ASR using Parakeet TDT v3 INT8 via sherpa-onnx.

    Accepts a numpy f32 array of speech audio, returns transcript + language.
    Runs synchronously in a thread executor to avoid blocking asyncio.
    """

    def __init__(
        self,
        model_dir: str = "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        num_threads: int = 4,
        supported_languages: list[str] | None = None,
    ):
        self._supported_languages = supported_languages or ["en"]
        self._model_dir = model_dir

        import sherpa_onnx

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.int8.onnx",
            decoder=f"{model_dir}/decoder.int8.onnx",
            joiner=f"{model_dir}/joiner.int8.onnx",
            tokens=f"{model_dir}/tokens.txt",
            num_threads=num_threads,
            model_type="nemo_transducer",
        )
        logger.info("Parakeet ASR loaded from %s (threads=%d)", model_dir, num_threads)

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe audio buffer.

        Returns: {"text": str, "language": str, "confidence": float, "rtf": float, "duration_s": float}

        Runs in a thread executor since sherpa-onnx decoding is synchronous
        and can take ~1-2s for a 5s utterance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio, sample_rate)

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Synchronous transcription — runs in thread executor."""
        start = time.monotonic()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        elapsed = time.monotonic() - start
        duration = len(audio) / sample_rate
        rtf = elapsed / duration if duration > 0 else 0

        logger.info("ASR: '%s' (%.1fs audio, RTF=%.2f)", text[:80], duration, rtf)

        language = detect_language(text, self._supported_languages)

        return {
            "text": text,
            "language": language,  # None means "keep current language"
            "confidence": 0.9,  # sherpa-onnx doesn't expose per-utterance confidence
            "rtf": rtf,
            "duration_s": duration,
        }

    async def shutdown(self) -> None:
        logger.info("Parakeet ASR shut down")
