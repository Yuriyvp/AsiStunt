"""Voice cloning workflow — clone once, cache forever.

Cloning requires the full OmniVoice model (~6-8 GB VRAM).
Produces a VoiceClonePrompt (ref_audio_tokens + ref_text + ref_rms) saved to disk.
Cache key: hash of reference_audio contents + voice description string.

This module is called BEFORE llama.cpp loads — it needs significant GPU headroom.
"""
import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PROFILES_DIR = Path("voice_db/profiles")
FILLERS_DIR = Path("voice_db/fillers")


def compute_cache_key(reference_audio_path: str | None, description: str) -> str:
    """Compute cache key from reference audio file hash + description hash."""
    h = hashlib.sha256()
    if reference_audio_path:
        path = Path(reference_audio_path)
        if path.exists():
            h.update(path.read_bytes())
    h.update(description.encode("utf-8"))
    return h.hexdigest()[:16]


def is_profile_valid(soul_name: str, cache_key: str) -> bool:
    """Check if a cached voice profile exists and matches the current config."""
    profile_path = PROFILES_DIR / f"{soul_name}.voiceprofile"
    key_path = PROFILES_DIR / f"{soul_name}.cachekey"
    if not profile_path.exists() or not key_path.exists():
        return False
    stored_key = key_path.read_text().strip()
    return stored_key == cache_key


def get_profile_path(soul_name: str) -> Path:
    return PROFILES_DIR / f"{soul_name}.voiceprofile"


async def run_voice_cloning(
    soul_name: str,
    reference_audio_path: str,
    description: str,
    cache_key: str,
    model_name: str = "k2-fsa/OmniVoice",
    on_progress: callable = None,
) -> Path:
    """Run the full voice cloning pipeline.

    1. Load OmniVoice model
    2. Create VoiceClonePrompt from reference audio
    3. Save prompt to disk as .voiceprofile
    4. Pre-render filler audio with cloned voice
    5. Unload model, free VRAM

    Returns: path to the saved .voiceprofile file
    """
    import asyncio
    import numpy as np

    loop = asyncio.get_running_loop()
    profile_path = PROFILES_DIR / f"{soul_name}.voiceprofile"
    key_path = PROFILES_DIR / f"{soul_name}.cachekey"

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    if on_progress:
        on_progress("Loading OmniVoice for voice cloning...")

    from omnivoice import OmniVoice

    model = await loop.run_in_executor(
        None,
        lambda: OmniVoice.from_pretrained(model_name, device_map="cuda:0", dtype=torch.float16),
    )

    if on_progress:
        on_progress("Processing reference audio...")

    # Create VoiceClonePrompt from reference audio
    prompt = await loop.run_in_executor(
        None,
        lambda: model.create_voice_clone_prompt(ref_audio=reference_audio_path),
    )

    # Save the VoiceClonePrompt to disk
    torch.save(
        {
            "ref_audio_tokens": prompt.ref_audio_tokens,
            "ref_text": prompt.ref_text,
            "ref_rms": prompt.ref_rms,
        },
        profile_path,
    )
    key_path.write_text(cache_key)
    logger.info("Voice profile saved: %s", profile_path)

    if on_progress:
        on_progress("Pre-rendering filler audio...")

    # Pre-render fillers with cloned voice
    filler_dir = FILLERS_DIR / soul_name
    filler_dir.mkdir(parents=True, exist_ok=True)

    filler_phrases = [
        "hmm", "uh", "let me think", "one moment",
        "well", "so", "right", "okay",
    ]

    for phrase in filler_phrases:
        audio_list = await loop.run_in_executor(
            None,
            lambda p=phrase: model.generate(text=p, voice_clone_prompt=prompt),
        )
        audio_f32 = audio_list[0].squeeze().cpu().float().numpy()
        audio_f32.tofile(filler_dir / f"{phrase.replace(' ', '_')}.f32")

    # Free VRAM
    del model
    torch.cuda.empty_cache()

    if on_progress:
        on_progress("Voice profile saved.")

    logger.info("Voice cloning complete. Profile: %s, %d fillers", profile_path, len(filler_phrases))
    return profile_path


def load_voice_clone_prompt(profile_path: str | Path) -> "VoiceClonePrompt":
    """Load a cached VoiceClonePrompt from disk."""
    from omnivoice.models.omnivoice import VoiceClonePrompt

    data = torch.load(profile_path, map_location="cpu", weights_only=True)
    return VoiceClonePrompt(
        ref_audio_tokens=data["ref_audio_tokens"],
        ref_text=data["ref_text"],
        ref_rms=data["ref_rms"],
    )
