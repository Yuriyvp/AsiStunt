"""Tests for pipeline state machine."""
import pytest

from voice_assistant.core.state_machine import (
    StateMachine,
    PipelineState,
    PipelineMode,
    VALID_TRANSITIONS,
)


class TestStateMachine:
    def test_initial_state(self):
        sm = StateMachine()
        assert sm.state == PipelineState.IDLE
        assert sm.mode == PipelineMode.DISABLED

    def test_valid_transition(self):
        sm = StateMachine()
        assert sm.transition(PipelineState.LISTENING)
        assert sm.state == PipelineState.LISTENING

    def test_invalid_transition_rejected(self):
        sm = StateMachine()
        # IDLE → SPEAKING is not valid
        assert not sm.transition(PipelineState.SPEAKING)
        assert sm.state == PipelineState.IDLE

    def test_full_happy_path(self):
        sm = StateMachine()
        assert sm.transition(PipelineState.LISTENING)
        assert sm.transition(PipelineState.PROCESSING)
        assert sm.transition(PipelineState.SPEAKING)
        assert sm.transition(PipelineState.IDLE)

    def test_barge_in_path(self):
        sm = StateMachine()
        sm.transition(PipelineState.LISTENING)
        sm.transition(PipelineState.PROCESSING)
        sm.transition(PipelineState.SPEAKING)
        assert sm.transition(PipelineState.INTERRUPTED)
        assert sm.transition(PipelineState.PROCESSING)

    def test_text_input_path(self):
        """IDLE → PROCESSING directly (text input skips LISTENING)."""
        sm = StateMachine()
        assert sm.transition(PipelineState.PROCESSING)
        assert sm.transition(PipelineState.SPEAKING)

    def test_processing_self_transition(self):
        sm = StateMachine()
        sm.transition(PipelineState.PROCESSING)
        assert sm.transition(PipelineState.PROCESSING)

    def test_on_change_callback(self):
        changes = []
        sm = StateMachine(on_change=lambda old, new: changes.append((old, new)))
        sm.transition(PipelineState.LISTENING)
        sm.transition(PipelineState.PROCESSING)
        assert changes == [
            (PipelineState.IDLE, PipelineState.LISTENING),
            (PipelineState.LISTENING, PipelineState.PROCESSING),
        ]

    def test_set_mode(self):
        sm = StateMachine()
        sm.set_mode(PipelineMode.FULL)
        assert sm.mode == PipelineMode.FULL
        sm.set_mode(PipelineMode.TEXT_ONLY)
        assert sm.mode == PipelineMode.TEXT_ONLY

    def test_time_in_state(self):
        sm = StateMachine()
        assert sm.time_in_state >= 0
        sm.transition(PipelineState.LISTENING)
        assert sm.time_in_state >= 0

    def test_interrupted_to_idle(self):
        sm = StateMachine()
        sm.transition(PipelineState.LISTENING)
        sm.transition(PipelineState.PROCESSING)
        sm.transition(PipelineState.SPEAKING)
        sm.transition(PipelineState.INTERRUPTED)
        assert sm.transition(PipelineState.IDLE)

    def test_interrupted_to_listening(self):
        sm = StateMachine()
        sm.transition(PipelineState.LISTENING)
        sm.transition(PipelineState.PROCESSING)
        sm.transition(PipelineState.SPEAKING)
        sm.transition(PipelineState.INTERRUPTED)
        assert sm.transition(PipelineState.LISTENING)

    def test_all_valid_transitions_work(self):
        """Verify every declared valid transition succeeds."""
        for from_state, to_states in VALID_TRANSITIONS.items():
            for to_state in to_states:
                sm = StateMachine()
                sm._state = from_state  # force state for testing
                assert sm.transition(to_state), f"{from_state} → {to_state} should be valid"
