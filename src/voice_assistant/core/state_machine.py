"""Pipeline state machine.

States: IDLE, LISTENING, PROCESSING, SPEAKING, INTERRUPTED
Transitions are explicit and validated — invalid transitions are logged and rejected.
"""
import enum
import logging
import time

logger = logging.getLogger(__name__)


class PipelineState(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class PipelineMode(enum.Enum):
    FULL = "FULL"
    TEXT_ONLY = "TEXT_ONLY"
    TRANSCRIBE = "TRANSCRIBE"
    DISABLED = "DISABLED"


# Valid state transitions
VALID_TRANSITIONS = {
    PipelineState.IDLE: {PipelineState.LISTENING, PipelineState.PROCESSING},
    PipelineState.LISTENING: {PipelineState.PROCESSING, PipelineState.IDLE},
    PipelineState.PROCESSING: {PipelineState.SPEAKING, PipelineState.IDLE, PipelineState.PROCESSING},
    PipelineState.SPEAKING: {PipelineState.IDLE, PipelineState.INTERRUPTED},
    PipelineState.INTERRUPTED: {PipelineState.PROCESSING, PipelineState.IDLE, PipelineState.LISTENING},
}


class StateMachine:
    """Deterministic state machine for the voice pipeline.

    All decisions <5ms. No LLM involved in state transitions.
    """

    def __init__(self, on_change=None):
        self._state = PipelineState.IDLE
        self._mode = PipelineMode.DISABLED  # starts disabled during init
        self._on_change = on_change
        self._state_entered_at: float = time.monotonic()

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def mode(self) -> PipelineMode:
        return self._mode

    def set_mode(self, mode: PipelineMode) -> None:
        old = self._mode
        self._mode = mode
        if old != mode:
            logger.info("Pipeline mode: %s → %s", old.value, mode.value)

    def transition(self, new_state: PipelineState) -> bool:
        """Attempt a state transition. Returns True if valid and executed."""
        if new_state not in VALID_TRANSITIONS.get(self._state, set()):
            logger.warning("Invalid transition: %s → %s", self._state.value, new_state.value)
            return False

        old = self._state
        self._state = new_state
        self._state_entered_at = time.monotonic()
        logger.info("State: %s → %s", old.value, new_state.value)

        if self._on_change:
            self._on_change(old, new_state)
        return True

    @property
    def time_in_state(self) -> float:
        """Seconds spent in current state."""
        return time.monotonic() - self._state_entered_at
