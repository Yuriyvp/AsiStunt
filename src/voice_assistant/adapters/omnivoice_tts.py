"""OmniVoice TTS adapter — in-process PyTorch inference.

Loads OmniVoice in inference mode (~6 GB VRAM fp16). Synthesizes text to f32 numpy arrays.
Supports both description-based (instruct) and clone-based voices via VoiceClonePrompt.

Output: 24 kHz f32 numpy arrays.
"""
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import torch

from voice_assistant.ports.tts import TTSPort

logger = logging.getLogger(__name__)


class OmniVoiceTTS(TTSPort):
    """OmniVoice TTS — in-process, returns numpy f32 audio at 24kHz."""

    SAMPLE_RATE = 24000

    def __init__(self, model_name: str = "k2-fsa/OmniVoice"):
        self._model_name = model_name
        self._model = None
        self._voice_clone_prompt = None
        self._voice_description: str | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def load(self, voice_method: str, voice_description: str,
                   profile_path: str | None = None) -> None:
        """Load OmniVoice in inference mode.

        Args:
            voice_method: "description" or "clone"
            voice_description: text description of voice (used as instruct for description mode)
            profile_path: path to .voiceprofile (required for clone method)
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._load_sync, voice_method, voice_description, profile_path
        )

    def _load_sync(self, voice_method: str, voice_description: str,
                   profile_path: str | None) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        from omnivoice import OmniVoice

        self._voice_description = voice_description
        self._model = OmniVoice.from_pretrained(
            self._model_name, device_map=str(self._device), dtype=torch.float16
        )
        self._model.eval()

        if voice_method == "clone" and profile_path:
            from voice_assistant.core.voice_clone import load_voice_clone_prompt
            self._voice_clone_prompt = load_voice_clone_prompt(profile_path)
            logger.info("Loaded voice profile from %s", profile_path)
        else:
            logger.info("Using instruct voice: %s", voice_description[:60])

        logger.info("OmniVoice loaded in inference mode on %s", self._device)

    async def synthesize(self, text: str, voice_params: dict | None = None) -> np.ndarray:
        """Synthesize text to audio.

        Args:
            text: text to speak
            voice_params: mood-adjusted params (speed, pitch_shift, energy, tags)

        Returns: f32 numpy array at 24kHz
        """
        if self._model is None:
            raise RuntimeError("OmniVoice not loaded — call load() first")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text, voice_params or {})

    def _synthesize_sync(self, text: str, voice_params: dict) -> np.ndarray:
        start = time.monotonic()
        speed = voice_params.get("speed", 1.0)

        # Inject non-verbal tags if mood provides them (e.g., [laughter])
        tags = voice_params.get("tags", [])
        if tags:
            text = " ".join(tags) + " " + text

        kwargs = {"text": text, "speed": speed}

        if self._voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = self._voice_clone_prompt
        elif self._voice_description:
            kwargs["instruct"] = self._voice_description

        with torch.inference_mode():
            audio_list = self._model.generate(**kwargs)

        result = audio_list[0].squeeze().cpu().float().numpy()

        elapsed = time.monotonic() - start
        duration = len(result) / self.SAMPLE_RATE
        rtf = elapsed / duration if duration > 0 else 0
        logger.info("TTS: '%s' → %d samples (%.2fs audio, RTF=%.3f)",
                     text[:40], len(result), duration, rtf)
        return result

    async def load_voice_profile(self, profile_path: str) -> None:
        """Hot-load a voice profile (e.g., after re-cloning)."""
        from voice_assistant.core.voice_clone import load_voice_clone_prompt
        loop = asyncio.get_running_loop()
        self._voice_clone_prompt = await loop.run_in_executor(
            None, load_voice_clone_prompt, profile_path
        )
        logger.info("Voice profile reloaded from %s", profile_path)

    async def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._voice_clone_prompt = None
            torch.cuda.empty_cache()
            logger.info("OmniVoice unloaded, VRAM freed")
