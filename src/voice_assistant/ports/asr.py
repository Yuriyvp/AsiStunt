"""Abstract base class for ASR (Automatic Speech Recognition) adapters."""
from abc import ABC, abstractmethod
import numpy as np


class ASRPort(ABC):
    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe audio buffer. Returns {"text": str, "language": str, "confidence": float}."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        ...
