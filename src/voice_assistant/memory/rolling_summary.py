"""Rolling summary (L2 memory).

Compact paragraph summarizing conversation history.
Updated during IDLE state when dialogue tail exceeds 80% budget.
Update is incremental: receives previous summary + new turns → produces integrated result.
If result exceeds 600 tokens, runs a compression pass.
"""
import asyncio
import logging

from voice_assistant.ports.llm import LLMPort

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are a conversation summarizer. You will receive a previous summary and new conversation turns. Produce an updated summary that integrates both.

The summary should capture:
- Key topics discussed
- User preferences and facts learned
- Emotional arc and relationship context
- Important decisions or agreements

Keep the summary under 150 words. Write as a cohesive paragraph, not bullet points.
If the previous summary is empty, create a new summary from the turns."""


class RollingSummary:
    def __init__(self, llm: LLMPort, summary_style: str = "emotional"):
        self._llm = llm
        self._style = summary_style
        self._summary: str = ""
        self._token_count: int = 0
        self._persisted_path: str = "data/summaries/current.txt"

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def token_count(self) -> int:
        return self._token_count

    async def update(self, new_turns: list[dict], timeout: float = 3.0) -> bool:
        """Update the rolling summary with new dialogue turns.

        Returns True if update succeeded, False if cancelled/timed out.
        Hard timeout: 3 seconds. Auto-cancel to prevent blocking if user speaks.
        """
        if not new_turns:
            return True

        turns_text = "\n".join(
            f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in new_turns
        )

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Previous summary:\n{self._summary or '(none)'}\n\nNew turns:\n{turns_text}"},
        ]

        try:
            result_parts = []
            async with asyncio.timeout(timeout):
                async for token in self._llm.stream(messages, {"temperature": 0.3, "max_tokens": 300}):
                    result_parts.append(token)

            self._summary = "".join(result_parts).strip()
            self._token_count = await self._llm.tokenize(self._summary)

            # Compression pass if over budget
            if self._token_count > 600:
                await self._compress(timeout=2.0)

            # Persist
            self._persist()
            logger.info("Summary updated (%d tokens)", self._token_count)
            return True

        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.info("Summary update cancelled/timed out")
            return False
        except Exception:
            logger.exception("Summary update failed")
            return False

    async def _compress(self, timeout: float = 2.0) -> None:
        """Compress summary to fit within 600 tokens."""
        messages = [
            {"role": "system", "content": "Compress the following summary to under 100 words while keeping key facts."},
            {"role": "user", "content": self._summary},
        ]
        try:
            result_parts = []
            async with asyncio.timeout(timeout):
                async for token in self._llm.stream(messages, {"temperature": 0.2, "max_tokens": 200}):
                    result_parts.append(token)
            self._summary = "".join(result_parts).strip()
            self._token_count = await self._llm.tokenize(self._summary)
        except Exception:
            logger.warning("Summary compression failed, keeping current")

    def _persist(self) -> None:
        """Save summary to disk."""
        from pathlib import Path
        path = Path(self._persisted_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._summary, encoding="utf-8")

    def load_persisted(self) -> None:
        """Load summary from disk on startup."""
        from pathlib import Path
        path = Path(self._persisted_path)
        if path.exists():
            self._summary = path.read_text(encoding="utf-8").strip()
            logger.info("Loaded persisted summary (%d chars)", len(self._summary))
