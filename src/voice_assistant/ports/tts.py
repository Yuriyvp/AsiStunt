"""Abstract base class for TTS adapters."""
from abc import ABC, abstractmethod
import numpy as np


class TTSPort(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice_params: dict) -> np.ndarray:
        """Synthesize text to audio. Returns f32 numpy array at 24kHz."""
        ...

    @abstractmethod
    async def load_voice_profile(self, profile_path: str) -> None:
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        ...
