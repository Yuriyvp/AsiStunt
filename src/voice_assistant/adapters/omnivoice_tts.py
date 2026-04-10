"""OmniVoice TTS adapter — in-process PyTorch inference, clone-only mode.

Every language uses a VoiceClonePrompt created from a reference audio sample.
No instruct/description mode — cloning is mandatory for consistent voice.

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
    """OmniVoice TTS — clone-only, per-language voice profiles."""

    SAMPLE_RATE = 24000

    def __init__(self, model_name: str = "k2-fsa/OmniVoice"):
        self._model_name = model_name
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Per-language voice clone prompts: {"en": VoiceClonePrompt, "hr": ...}
        self._voice_prompts: dict = {}
        self._active_language: str = "en"

    async def load(self) -> None:
        """Load OmniVoice model (no voice profiles yet — load those separately)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        from omnivoice import OmniVoice

        self._model = OmniVoice.from_pretrained(
            self._model_name, device_map=str(self._device), dtype=torch.float16
        )
        self._model.eval()
        logger.info("OmniVoice loaded on %s", self._device)

    def load_voice_profile_sync(self, lang: str, profile_path: str) -> None:
        """Load a cached voice profile for a language."""
        from voice_assistant.core.voice_clone import load_voice_clone_prompt
        self._voice_prompts[lang] = load_voice_clone_prompt(profile_path)
        logger.info("Loaded voice profile for '%s' from %s", lang, profile_path)

    async def load_voice_profile(self, lang: str, profile_path: str) -> None:
        """Load a cached voice profile for a language (async)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.load_voice_profile_sync, lang, profile_path)

    def clone_voice_sync(self, lang: str, reference_audio: str) -> "VoiceClonePrompt":
        """Create a voice clone prompt from reference audio for a language."""
        if self._model is None:
            raise RuntimeError("OmniVoice not loaded")

        logger.info("Cloning voice for '%s' from %s", lang, reference_audio)
        with torch.inference_mode():
            prompt = self._model.create_voice_clone_prompt(ref_audio=reference_audio)
        self._voice_prompts[lang] = prompt
        logger.info("Voice cloned for '%s'", lang)
        return prompt

    async def clone_voice(self, lang: str, reference_audio: str) -> "VoiceClonePrompt":
        """Create a voice clone prompt from reference audio (async)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.clone_voice_sync, lang, reference_audio
        )

    def set_language(self, lang: str) -> None:
        """Set the active language for synthesis."""
        self._active_language = lang

    def unload_language(self, lang: str) -> None:
        """Remove a loaded voice profile for a language."""
        if lang in self._voice_prompts:
            del self._voice_prompts[lang]
            logger.info("Unloaded voice profile for '%s'", lang)

    @property
    def available_languages(self) -> list[str]:
        """Languages that have loaded voice profiles."""
        return list(self._voice_prompts.keys())

    async def synthesize(self, text: str, voice_params: dict | None = None) -> np.ndarray:
        """Synthesize text to audio using the active language's voice profile."""
        if self._model is None:
            raise RuntimeError("OmniVoice not loaded — call load() first")

        lang = (voice_params or {}).get("language", self._active_language)
        prompt = self._voice_prompts.get(lang)
        if prompt is None:
            # Fall back to any available profile
            if self._voice_prompts:
                lang = next(iter(self._voice_prompts))
                prompt = self._voice_prompts[lang]
                logger.warning("No voice for '%s', falling back to '%s'",
                               self._active_language, lang)
            else:
                raise RuntimeError(f"No voice profile loaded for any language")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._synthesize_sync, text, voice_params or {}, prompt, lang
        )

    def _synthesize_sync(self, text: str, voice_params: dict,
                         prompt, lang: str) -> np.ndarray:
        start = time.monotonic()
        speed = voice_params.get("speed", 1.0)

        kwargs = {
            "text": text,
            "speed": speed,
            "voice_clone_prompt": prompt,
            "language": lang,
        }

        with torch.inference_mode():
            audio_list = self._model.generate(**kwargs)

        result = audio_list[0].squeeze().cpu().float().numpy()

        elapsed = time.monotonic() - start
        duration = len(result) / self.SAMPLE_RATE
        rtf = elapsed / duration if duration > 0 else 0
        logger.info("TTS [%s]: '%s' → %.2fs audio, RTF=%.3f",
                     lang, text[:40], duration, rtf)
        return result

    async def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._voice_prompts.clear()
            torch.cuda.empty_cache()
            logger.info("OmniVoice unloaded, VRAM freed")
