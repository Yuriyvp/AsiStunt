"""Tests for context window, rolling summary, and context builder."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_assistant.memory.context_window import (
    TOTAL_CONTEXT, FIXED_BUDGET, DIALOGUE_BUDGET,
    SUMMARY_TRIGGER_TOKENS, MIN_KEEP_TURNS,
    SINGLE_TURN_MAX, SINGLE_TURN_KEEP_HEAD, SINGLE_TURN_KEEP_TAIL,
)
from voice_assistant.memory.rolling_summary import RollingSummary
from voice_assistant.core.context_builder import ContextBuilder
from voice_assistant.core.orchestrator import Turn
from voice_assistant.core.mood import MoodState


# --- Context Window Constants ---


class TestContextWindow:
    def test_budget_math(self):
        assert DIALOGUE_BUDGET + FIXED_BUDGET == TOTAL_CONTEXT

    def test_dialogue_budget_positive(self):
        assert DIALOGUE_BUDGET > 0

    def test_summary_trigger_below_budget(self):
        assert SUMMARY_TRIGGER_TOKENS < DIALOGUE_BUDGET

    def test_min_keep_turns(self):
        assert MIN_KEEP_TURNS >= 2

    def test_single_turn_limits(self):
        assert SINGLE_TURN_KEEP_HEAD + SINGLE_TURN_KEEP_TAIL < SINGLE_TURN_MAX


# --- Rolling Summary ---


def _mock_llm(stream_tokens=None, token_count=50):
    """Create a mock LLM that streams tokens and counts."""
    llm = AsyncMock()

    async def mock_stream(messages, sampling=None, **kwargs):
        for t in (stream_tokens or ["Summary ", "of ", "conversation."]):
            yield t
    llm.stream = mock_stream
    llm.tokenize = AsyncMock(return_value=token_count)
    return llm


class TestRollingSummary:
    @pytest.mark.asyncio
    async def test_initial_empty(self):
        llm = _mock_llm()
        rs = RollingSummary(llm)
        assert rs.summary == ""
        assert rs.token_count == 0

    @pytest.mark.asyncio
    async def test_update_with_turns(self):
        llm = _mock_llm()
        rs = RollingSummary(llm)
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = await rs.update(turns)
        assert result is True
        assert rs.summary == "Summary of conversation."
        assert rs.token_count == 50

    @pytest.mark.asyncio
    async def test_update_empty_turns(self):
        llm = _mock_llm()
        rs = RollingSummary(llm)
        result = await rs.update([])
        assert result is True
        assert rs.summary == ""

    @pytest.mark.asyncio
    async def test_compression_triggered(self):
        """When token count > 600, compression pass should run."""
        call_count = 0

        async def counting_stream(messages, sampling=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "A very long summary."
            else:
                yield "Short."
        llm = AsyncMock()
        llm.stream = counting_stream
        # First tokenize returns >600 to trigger compression, then <600 after
        llm.tokenize = AsyncMock(side_effect=[700, 50])

        rs = RollingSummary(llm)
        await rs.update([{"role": "user", "content": "test"}])
        assert call_count == 2  # initial + compression

    @pytest.mark.asyncio
    async def test_persist_and_load(self, tmp_path):
        llm = _mock_llm()
        rs = RollingSummary(llm)
        rs._persisted_path = str(tmp_path / "summary.txt")
        await rs.update([{"role": "user", "content": "Hello"}])

        # Load into new instance
        rs2 = RollingSummary(llm)
        rs2._persisted_path = rs._persisted_path
        rs2.load_persisted()
        assert rs2.summary == "Summary of conversation."

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        async def slow_stream(messages, sampling=None, **kwargs):
            await asyncio.sleep(10)
            yield "too late"
        llm = AsyncMock()
        llm.stream = slow_stream

        rs = RollingSummary(llm)
        result = await rs.update([{"role": "user", "content": "test"}], timeout=0.01)
        assert result is False


# --- Context Builder ---


class TestContextBuilder:
    def _make_turns(self, n):
        turns = []
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            turns.append(Turn(
                role=role, content=f"Message {i}", source="text", timestamp=float(i),
            ))
        return turns

    @pytest.mark.asyncio
    async def test_build_basic(self):
        llm = _mock_llm(token_count=10)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("You are helpful.", "A kind assistant.")

        turns = self._make_turns(2)
        mood = MoodState()
        messages = await cb.build(turns, mood, "en")

        assert messages[0]["role"] == "system"
        assert "A kind assistant." in messages[0]["content"]
        assert "You are helpful." in messages[0]["content"]
        assert len(messages) == 3  # system + 2 turns

    @pytest.mark.asyncio
    async def test_build_includes_mood(self):
        llm = _mock_llm(token_count=10)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("personality", "backstory")

        mood = MoodState()
        mood.update("happy", 0.9)
        messages = await cb.build(self._make_turns(1), mood, "en")

        assert "Current mood:" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_build_includes_summary(self):
        llm = _mock_llm(token_count=10)
        summary = RollingSummary(llm)
        summary._summary = "User likes cats."
        cb = ContextBuilder(llm, summary)
        cb.set_persona("p", "b")

        messages = await cb.build(self._make_turns(1), MoodState(), "en")
        assert "User likes cats." in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_build_includes_language(self):
        llm = _mock_llm(token_count=10)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("p", "b")

        messages = await cb.build(self._make_turns(1), MoodState(), "ja")
        assert "Respond in ja." in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_evicts_old_turns_when_over_budget(self):
        llm = _mock_llm(token_count=10)
        # System message uses 7700 tokens → only ~2 tokens left for dialogue
        # Each turn = 10 tokens → should evict down to MIN_KEEP_TURNS
        llm.tokenize = AsyncMock(return_value=7700)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("p", "b")

        turns = self._make_turns(20)
        messages = await cb.build(turns, MoodState(), "en")
        # system + MIN_KEEP_TURNS dialogue messages
        assert len(messages) == 1 + MIN_KEEP_TURNS

    @pytest.mark.asyncio
    async def test_needs_summary_update(self):
        llm = _mock_llm(token_count=10)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("p", "b")

        # Populate token cache
        turns = self._make_turns(4)
        await cb.build(turns, MoodState(), "en")

        # With 4 turns × 10 tokens = 40, well under SUMMARY_TRIGGER_TOKENS
        assert not cb.needs_summary_update(turns)

    @pytest.mark.asyncio
    async def test_needs_summary_update_true(self):
        # Each turn = 2000 tokens → 4 turns = 8000, over trigger
        llm = _mock_llm(token_count=2000)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("p", "b")

        turns = self._make_turns(4)
        await cb.build(turns, MoodState(), "en")
        assert cb.needs_summary_update(turns)

    @pytest.mark.asyncio
    async def test_tokenize_persona(self):
        llm = _mock_llm(token_count=150)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("personality text", "backstory text")
        await cb.tokenize_persona()
        assert cb._persona_tokens == 150

    @pytest.mark.asyncio
    async def test_mood_signal_instruction_included(self):
        llm = _mock_llm(token_count=10)
        summary = RollingSummary(llm)
        cb = ContextBuilder(llm, summary)
        cb.set_persona("p", "b")

        messages = await cb.build(self._make_turns(1), MoodState(), "en")
        assert "<mood_signal>" in messages[0]["content"]
