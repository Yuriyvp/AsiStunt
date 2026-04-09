"""Signal types for the debug/event system.

All modules emit typed signals through the event bus.
Signals flow via stdout to Tauri — debug UI + companion UI.
"""
import enum


class SignalType(str, enum.Enum):
    # VAD/Audio
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    BARGE_IN = "barge_in"
    INPUT_LEVEL = "audio.input_level"

    # ASR
    TRANSCRIPT = "transcript"
    DETECTED_LANGUAGE = "detected_language"
    LANGUAGE_SWITCH = "language_switch"

    # LLM
    REQUEST_START = "request_start"
    FIRST_TOKEN = "first_token"
    COMPLETE = "complete"
    TOKENS_PER_SEC = "tokens_per_sec"

    # TTS
    SYNTH_START = "synth_start"
    SYNTH_END = "synth_end"

    # State
    STATE_CHANGE = "state_change"
    MOOD_CHANGE = "mood_change"
    FILLER_PLAYED = "filler_played"

    # System
    END_TO_END_LATENCY = "end_to_end_latency"
    VRAM_USAGE = "vram_usage"
    VRAM_OPTIMIZED = "system.vram_optimized"
    DEVICE_CHANGE = "system.device_change"
    VOICE_CLONE_PROGRESS = "voice_clone_progress"
    PROCESS_STATE_CHANGE = "process_state_change"
    ERROR = "error"
