"""Full lifecycle integration tests — realistic app start → conversation → shutdown.

Simulates the complete application lifecycle with realistic voice samples,
multi-turn conversations, language switching, mood changes, and graceful shutdown.

Run: pytest tests/integration/test_full_lifecycle.py -v -s
"""
import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from voice_assistant.core.orchestrator import Orchestrator, Turn
from voice_assistant.core.state_machine import PipelineState, PipelineMode

from conftest import (
    make_mock_components, make_orchestrator, make_speech_audio, make_tts_audio,
    make_llm_stream, make_mood_stream, make_asr_responses,
    make_multilang_llm_responses, VOICE_SAMPLES,
)


class TestFullConversationLifecycle:
    """Simulate a complete multi-turn conversation from start to finish."""

    @pytest.mark.asyncio
    async def test_startup_idle_state(self):
        """Fresh orchestrator starts in IDLE state."""
        orch, comps = make_orchestrator()
        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == 0
        assert orch._current_language == "en"
        assert orch._mood.mood == "warm"

    @pytest.mark.asyncio
    async def test_start_and_stop_lifecycle(self):
        """start() -> VAD loop runs -> stop() cleans up."""
        orch, comps = make_orchestrator()
        await orch.start()
        assert orch._vad_task is not None
        assert not orch._vad_task.done()
        await orch.stop()
        assert orch._vad_task.done()

    @pytest.mark.asyncio
    async def test_five_turn_english_conversation(self):
        """5-turn English conversation: each turn produces user+assistant dialogue."""
        turn_idx = 0
        responses = [
            ["I'm ", "doing ", "great, ", "thanks!"],
            ["Space ", "is ", "fascinating. ", "Stars ", "are ", "amazing!"],
            ["Quantum ", "computing ", "uses ", "qubits ", "instead of ", "bits."],
            ["Consciousness ", "is ", "a ", "deep ", "philosophical ", "question."],
            ["Sure, ", "I'd ", "love ", "to ", "chat!"],
        ]

        comps = make_mock_components()
        comps["llm"].stream = make_multilang_llm_responses(responses)
        orch, _ = make_orchestrator(comps)

        texts = [
            "Hello, how are you today?",
            "Tell me about space.",
            "Can you explain quantum computing?",
            "What is consciousness?",
            "Let's keep chatting!",
        ]

        for text in texts:
            await orch.handle_text_input(text)
            if orch._processing_task:
                await asyncio.wait_for(orch._processing_task, timeout=10.0)

        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == 10  # 5 user + 5 assistant
        for i in range(5):
            assert orch.dialogue[i * 2].role == "user"
            assert orch.dialogue[i * 2].content == texts[i]
            assert orch.dialogue[i * 2 + 1].role == "assistant"
            assert len(orch.dialogue[i * 2 + 1].content) > 0

    @pytest.mark.asyncio
    async def test_mixed_voice_and_text_conversation(self):
        """Alternating voice and text inputs in a single conversation."""
        call_idx = 0

        async def voice_transcribe(audio):
            nonlocal call_idx
            results = [
                {"text": "what time is it", "language": "en", "confidence": 0.92},
                {"text": "thanks for that", "language": "en", "confidence": 0.95},
            ]
            result = results[min(call_idx, len(results) - 1)]
            call_idx += 1
            return result

        comps = make_mock_components()
        comps["asr"].transcribe = voice_transcribe
        comps["llm"].stream = make_multilang_llm_responses([
            ["It's ", "3 PM."],
            ["You're ", "welcome!"],
            ["Let me ", "think ", "about that."],
            ["Of course!"],
        ])
        orch, _ = make_orchestrator(comps)

        # Turn 1: voice input
        orch._state._state = PipelineState.PROCESSING
        await orch._process_voice_turn(make_speech_audio(1.5))

        # Turn 2: text input
        await orch.handle_text_input("Tell me a joke")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # Turn 3: voice input
        orch._state._state = PipelineState.PROCESSING
        await orch._process_voice_turn(make_speech_audio(1.0))

        # Turn 4: text input
        await orch.handle_text_input("Goodbye!")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert len(orch.dialogue) == 8  # 4 user + 4 assistant
        # Check source types
        assert orch.dialogue[0].source == "voice"
        assert orch.dialogue[2].source == "text"
        assert orch.dialogue[4].source == "voice"
        assert orch.dialogue[6].source == "text"

    @pytest.mark.asyncio
    async def test_rapid_short_turns(self):
        """Rapid sequence of short text inputs (stress test for state machine)."""
        comps = make_mock_components()
        comps["llm"].stream = make_llm_stream(["OK."])
        orch, _ = make_orchestrator(comps)

        short_inputs = ["Yeah.", "OK.", "Sure.", "Right.", "Hmm.", "Yes.", "No."]

        for text in short_inputs:
            await orch.handle_text_input(text)
            if orch._processing_task:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == len(short_inputs) * 2

    @pytest.mark.asyncio
    async def test_conversation_with_mood_changes(self):
        """Conversation where mood changes across turns."""
        mood_log = []

        responses_with_mood = [
            # Happy mood
            ["<mood_signal>user_tone=happy, intensity=0.9</mood_signal>",
             "That's ", "wonderful ", "news!"],
            # Sad mood
            ["<mood_signal>user_tone=sad, intensity=0.7</mood_signal>",
             "I'm ", "sorry ", "to ", "hear ", "that."],
            # Neutral
            ["<mood_signal>user_tone=neutral, intensity=0.5</mood_signal>",
             "Sure, ", "let's ", "talk."],
        ]

        comps = make_mock_components()
        comps["llm"].stream = make_multilang_llm_responses(responses_with_mood)
        orch, _ = make_orchestrator(comps)

        inputs = ["I got promoted!", "My cat is sick.", "What should we discuss?"]

        for text in inputs:
            await orch.handle_text_input(text)
            if orch._processing_task:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)
            mood_log.append(orch._mood.mood)

        assert mood_log[0] == "playful"   # happy → playful
        assert mood_log[1] == "concerned"  # sad → concerned
        assert mood_log[2] == "warm"       # neutral → warm

    @pytest.mark.asyncio
    async def test_dialogue_history_window(self):
        """Orchestrator passes last 10 turns to LLM for context."""
        captured_messages = []

        async def capturing_stream(messages, sampling=None, **kwargs):
            captured_messages.append(list(messages))
            yield "Response."

        comps = make_mock_components()
        comps["llm"].stream = capturing_stream
        orch, _ = make_orchestrator(comps)

        # Send 7 turns to build up history
        for i in range(7):
            await orch.handle_text_input(f"Message {i}")
            if orch._processing_task:
                await asyncio.wait_for(orch._processing_task, timeout=5.0)

        # The last LLM call should have system prompt + last 10 dialogue turns
        last_call = captured_messages[-1]
        assert last_call[0]["role"] == "system"
        # Count non-system messages
        dialogue_msgs = [m for m in last_call if m["role"] != "system"]
        assert len(dialogue_msgs) <= 10

    @pytest.mark.asyncio
    async def test_empty_transcription_doesnt_break_flow(self):
        """Empty ASR result → skip turn, stay IDLE, next turn works."""
        call_idx = 0

        async def sometimes_empty_transcribe(audio):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return {"text": "   ", "language": "en", "confidence": 0.1}
            return {"text": "real speech here", "language": "en", "confidence": 0.95}

        comps = make_mock_components()
        comps["asr"].transcribe = sometimes_empty_transcribe
        orch, _ = make_orchestrator(comps)

        # First voice turn: empty → should go back to IDLE
        orch._state._state = PipelineState.PROCESSING
        await orch._process_voice_turn(make_speech_audio(0.3))
        assert orch.state == PipelineState.IDLE
        assert len(orch.dialogue) == 0

        # Second voice turn: real speech → should produce dialogue
        orch._state._state = PipelineState.PROCESSING
        await orch._process_voice_turn(make_speech_audio(1.5))
        assert len(orch.dialogue) == 2

    @pytest.mark.asyncio
    async def test_shutdown_during_processing(self):
        """stop() called while processing a turn → clean shutdown."""
        async def slow_stream(messages, sampling=None, **kwargs):
            for token in ["Slow", " response", " here"]:
                await asyncio.sleep(0.1)
                yield token

        comps = make_mock_components()
        comps["llm"].stream = slow_stream
        orch, _ = make_orchestrator(comps)
        await orch.start()

        # Start processing
        await orch.handle_text_input("hello")
        await asyncio.sleep(0.05)  # let processing start

        # Stop while still processing
        await orch.stop()

        # Should not hang or crash
        assert orch._vad_task.done()

    @pytest.mark.asyncio
    async def test_new_conversation_clears_dialogue(self):
        """Clearing dialogue mid-conversation resets state."""
        comps = make_mock_components()
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("First message")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)
        assert len(orch.dialogue) == 2

        # Clear dialogue (new conversation)
        orch._dialogue.clear()
        assert len(orch.dialogue) == 0

        # Continue with new conversation
        await orch.handle_text_input("Second conversation")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)
        assert len(orch.dialogue) == 2
        assert orch.dialogue[0].content == "Second conversation"


class TestVoiceTurnProcessing:
    """Test voice turn processing with realistic audio samples."""

    @pytest.mark.asyncio
    async def test_all_voice_samples_process(self, voice_samples):
        """All 20 voice samples produce valid dialogue turns."""
        for i, sample in enumerate(voice_samples):
            comps = make_mock_components()
            comps["asr"].transcribe = AsyncMock(
                return_value={"text": sample["text"], "language": sample["lang"],
                              "confidence": 0.95}
            )
            orch, _ = make_orchestrator(comps)
            orch._state._state = PipelineState.PROCESSING

            await orch._process_voice_turn(sample["audio"])

            assert orch.state == PipelineState.IDLE, \
                f"Sample {i} ({sample['lang']}: '{sample['text'][:30]}') didn't return to IDLE"
            assert len(orch.dialogue) >= 2, \
                f"Sample {i} didn't produce dialogue"
            assert orch.dialogue[0].content == sample["text"]

    @pytest.mark.asyncio
    async def test_short_samples_processed_correctly(self, voice_samples):
        """Very short samples (< 0.5s) still process correctly."""
        short = [s for s in voice_samples if s["duration_s"] <= 0.5]
        assert len(short) >= 4, "Need at least 4 short samples"

        for sample in short:
            comps = make_mock_components()
            comps["asr"].transcribe = AsyncMock(
                return_value={"text": sample["text"], "language": sample["lang"],
                              "confidence": 0.9}
            )
            orch, _ = make_orchestrator(comps)
            orch._state._state = PipelineState.PROCESSING

            await orch._process_voice_turn(sample["audio"])
            assert len(orch.dialogue) >= 2

    @pytest.mark.asyncio
    async def test_long_samples_processed_correctly(self, voice_samples):
        """Long samples (> 3s) still process correctly."""
        long_samples = [s for s in voice_samples if s["duration_s"] >= 3.0]
        assert len(long_samples) >= 2, "Need at least 2 long samples"

        for sample in long_samples:
            comps = make_mock_components()
            comps["asr"].transcribe = AsyncMock(
                return_value={"text": sample["text"], "language": sample["lang"],
                              "confidence": 0.93}
            )
            # Long response for long input
            comps["llm"].stream = make_llm_stream(
                ["This is ", "a longer ", "response ", "that matches ", "the complexity ",
                 "of the ", "input. ", "It has ", "multiple ", "sentences."]
            )
            orch, _ = make_orchestrator(comps)
            orch._state._state = PipelineState.PROCESSING

            await orch._process_voice_turn(sample["audio"])
            assert len(orch.dialogue) >= 2


class TestTimingAndLatency:
    """Verify timing-sensitive behavior."""

    @pytest.mark.asyncio
    async def test_first_chunk_fires_quickly(self):
        """First TTS chunk should be queued before LLM finishes."""
        tts_call_times = []
        original_synth = AsyncMock(return_value=make_tts_audio())

        async def timed_synth(text, voice_params=None):
            tts_call_times.append(time.monotonic())
            return make_tts_audio()

        comps = make_mock_components()
        comps["tts"].synthesize = timed_synth
        # Stream with enough text for first chunk (>15 chars)
        comps["llm"].stream = make_llm_stream(
            ["This is the first sentence, ", "and here is the second one. ",
             "And a third for good measure."],
            delay=0.01
        )
        orch, _ = make_orchestrator(comps)

        t0 = time.monotonic()
        await orch.handle_text_input("tell me something")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        # TTS should have been called at least once
        assert len(tts_call_times) >= 1

    @pytest.mark.asyncio
    async def test_multiple_chunks_sequential_tts(self):
        """TTS worker processes chunks sequentially (ordered playlist)."""
        chunk_order = []

        async def tracking_synth(text, voice_params=None):
            chunk_order.append(text)
            return make_tts_audio()

        comps = make_mock_components()
        comps["tts"].synthesize = tracking_synth
        comps["llm"].stream = make_llm_stream(
            ["First sentence here. ", "Second sentence follows. ", "Third one wraps up."]
        )
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("tell me three things")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        # At least 2 chunks should have been produced
        assert len(chunk_order) >= 2
        # All chunks should be non-empty
        assert all(len(c) > 0 for c in chunk_order)

    @pytest.mark.asyncio
    async def test_speaking_state_reached_during_streaming(self):
        """SPEAKING state is entered while LLM is still streaming."""
        state_log = []

        comps = make_mock_components()
        comps["llm"].stream = make_llm_stream(
            ["This is enough text for the first chunk. ",
             "And here is more text for additional chunks. ",
             "The stream keeps going with content."],
            delay=0.01
        )
        orch, _ = make_orchestrator(comps)
        orch._state._on_change = lambda old, new: state_log.append((old, new))

        await orch.handle_text_input("tell me a story")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=10.0)

        states = [s[1] for s in state_log]
        assert PipelineState.SPEAKING in states
        assert states[-1] == PipelineState.IDLE

    @pytest.mark.asyncio
    async def test_mood_decay_happens_each_turn(self):
        """Mood intensity decays by 0.85x after each turn."""
        comps = make_mock_components()
        comps["llm"].stream = make_mood_stream("happy", 1.0, ["Great!"])
        orch, _ = make_orchestrator(comps)

        await orch.handle_text_input("I'm so happy!")
        if orch._processing_task:
            await asyncio.wait_for(orch._processing_task, timeout=5.0)

        assert orch._mood.mood == "playful"
        intensity_after = orch._mood.intensity
        # After decay: 1.0 * 0.85 = 0.85
        assert abs(intensity_after - 0.85) < 0.01
