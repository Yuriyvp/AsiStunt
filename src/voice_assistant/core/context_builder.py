"""Context builder — assembles the LLM prompt with token budgets.

Builds OpenAI-compatible messages list:
[system message (persona + state + summary), ...dialogue turns]
"""
import logging

from voice_assistant.core.mood import MoodState
from voice_assistant.core.orchestrator import Turn
from voice_assistant.memory.context_window import (
    DIALOGUE_BUDGET, GENERATION_HEADROOM, MIN_KEEP_TURNS,
    SINGLE_TURN_MAX, SINGLE_TURN_KEEP_HEAD, SINGLE_TURN_KEEP_TAIL,
    SUMMARY_TRIGGER_TOKENS,
)
from voice_assistant.memory.rolling_summary import RollingSummary
from voice_assistant.ports.llm import LLMPort

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Assembles the prompt for each LLM call, respecting token budgets."""

    def __init__(self, llm: LLMPort, summary: RollingSummary):
        self._llm = llm
        self._summary = summary
        self._persona_card: str = ""
        self._persona_tokens: int = 0
        self._turn_token_cache: dict[int, int] = {}  # id(turn) → token count

    def set_persona(self, personality: str, backstory: str) -> None:
        """Set persona card text. Tokenize once, cache."""
        self._persona_card = f"{backstory.strip()}\n\n{personality.strip()}"

    async def tokenize_persona(self) -> None:
        """Count persona tokens (call once at startup and on SOUL reload)."""
        self._persona_tokens = await self._llm.tokenize(self._persona_card)
        logger.info("Persona card: %d tokens", self._persona_tokens)

    async def build(
        self,
        dialogue: list[Turn],
        mood: MoodState,
        current_language: str,
    ) -> list[dict]:
        """Build the messages list for llama.cpp.

        Returns OpenAI-compatible messages: [system, user, assistant, user, ...]
        """
        # System message parts
        system_parts = [self._persona_card]
        system_parts.append(f"\nCurrent mood: {mood.mood} (intensity: {mood.intensity:.1f})")
        system_parts.append(f"\nRespond in {current_language}.")

        if self._summary.summary:
            system_parts.append(f"\n### What you remember from past conversations:\n{self._summary.summary}")

        # Mood signal instruction
        system_parts.append(
            "\n\nBefore each response, emit exactly one mood signal tag on its own line:\n"
            "<mood_signal>user_tone=TONE, intensity=N.N</mood_signal>\n\n"
            "TONE is one of: happy, sad, angry, neutral, anxious, affectionate, playful, frustrated\n"
            "intensity is 0.0 to 1.0\n\n"
            "Then write your response normally. The mood tag will be stripped and not shown."
        )

        system_message = "\n".join(system_parts)
        system_tokens = await self._llm.tokenize(system_message)
        available = 8192 - system_tokens - GENERATION_HEADROOM

        # Fit dialogue turns
        fitted_turns = await self._fit_dialogue(dialogue, available)

        messages = [{"role": "system", "content": system_message}]
        for turn in fitted_turns:
            messages.append({"role": turn.role, "content": turn.content})

        return messages

    async def _fit_dialogue(self, dialogue: list[Turn], budget: int) -> list[Turn]:
        """Fit dialogue turns within token budget. Evict oldest first, keep last MIN_KEEP_TURNS."""
        if not dialogue:
            return []

        # Tokenize turns (cached)
        for turn in dialogue:
            tid = id(turn)
            if tid not in self._turn_token_cache:
                self._turn_token_cache[tid] = await self._llm.tokenize(turn.content)

        # Truncate very long single turns
        for turn in dialogue:
            tid = id(turn)
            if self._turn_token_cache[tid] > SINGLE_TURN_MAX:
                mid_omit = self._turn_token_cache[tid] - SINGLE_TURN_KEEP_HEAD - SINGLE_TURN_KEEP_TAIL
                chars = len(turn.content)
                head = turn.content[:int(chars * SINGLE_TURN_KEEP_HEAD / self._turn_token_cache[tid])]
                tail = turn.content[-int(chars * SINGLE_TURN_KEEP_TAIL / self._turn_token_cache[tid]):]
                turn.content = f"{head}\n[... middle truncated, ~{mid_omit} tokens omitted ...]\n{tail}"
                self._turn_token_cache[tid] = SINGLE_TURN_KEEP_HEAD + SINGLE_TURN_KEEP_TAIL + 20

        # Evict oldest until we fit
        result = list(dialogue)
        while len(result) > MIN_KEEP_TURNS:
            total = sum(self._turn_token_cache.get(id(t), 50) for t in result)
            if total <= budget:
                break
            result.pop(0)  # evict oldest

        return result

    def needs_summary_update(self, dialogue: list[Turn]) -> bool:
        """Check if dialogue tail has exceeded the summary trigger threshold."""
        total = sum(self._turn_token_cache.get(id(t), 50) for t in dialogue)
        return total > SUMMARY_TRIGGER_TOKENS
