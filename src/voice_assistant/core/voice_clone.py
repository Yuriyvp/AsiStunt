"""Voice cloning utilities — profile paths and loading.

Profiles are saved as .voiceprofile files containing serialized VoiceClonePrompt
data (ref_audio_tokens + ref_text + ref_rms).

Cloning is done via OmniVoiceTTS.clone_voice() using the already-loaded model.
Profile saving is handled in main.py's _clone_voice_for_lang().
"""
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PROFILES_DIR = Path("voice_db/profiles")


def get_profile_path(profile_name: str) -> Path:
    """Get path for a voice profile file.

    Profile names follow the pattern: {soul_name}_{language_id}
    e.g., "Aria_en" → voice_db/profiles/Aria_en.voiceprofile
    """
    return PROFILES_DIR / f"{profile_name}.voiceprofile"


def load_voice_clone_prompt(profile_path: str | Path) -> "VoiceClonePrompt":
    """Load a cached VoiceClonePrompt from disk."""
    from omnivoice.models.omnivoice import VoiceClonePrompt

    data = torch.load(profile_path, map_location="cpu", weights_only=True)
    return VoiceClonePrompt(
        ref_audio_tokens=data["ref_audio_tokens"],
        ref_text=data["ref_text"],
        ref_rms=data["ref_rms"],
    )
