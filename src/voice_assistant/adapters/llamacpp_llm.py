"""llama.cpp LLM adapter — OpenAI-compatible streaming client.

Connects to the llama.cpp server over HTTP localhost.
Streams tokens via SSE from /v1/chat/completions.
Supports cancellation by closing the HTTP connection (no /cancel endpoint).
"""
import asyncio
import json
import logging
from collections.abc import AsyncIterator

import aiohttp

from voice_assistant.ports.llm import LLMPort

logger = logging.getLogger(__name__)

DEFAULT_SAMPLING = {
    "temperature": 0.75,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.1,
    "repeat_last_n": 256,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


class LlamaCppLLM(LLMPort):
    """Streaming LLM adapter for llama.cpp server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self._base_url = base_url
        self._session: aiohttp.ClientSession | None = None
        self._current_response: aiohttp.ClientResponse | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def stream(self, messages: list[dict], sampling: dict | None = None,
                     thinking: bool = False) -> AsyncIterator[str]:
        """Stream tokens from llama.cpp /v1/chat/completions.

        Args:
            thinking: Enable model reasoning (default off for speed).

        Yields individual token strings. Caller should concatenate.
        """
        params = {**DEFAULT_SAMPLING, **(sampling or {})}
        session = await self._ensure_session()

        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": thinking},
            **params,
        }

        url = f"{self._base_url}/v1/chat/completions"

        try:
            self._current_response = await session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120, sock_read=60),
            )
            self._current_response.raise_for_status()

            async for line in self._current_response.content:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed SSE: %s", data_str[:100])
        except asyncio.CancelledError:
            logger.debug("LLM stream cancelled")
            raise
        except aiohttp.ClientError as e:
            logger.error("LLM stream error: %s", e)
            raise
        finally:
            if self._current_response:
                self._current_response.close()
                self._current_response = None

    async def cancel(self) -> None:
        """Cancel in-flight generation by closing the HTTP connection.

        The llama.cpp server detects the disconnection and stops generating.
        No explicit /cancel endpoint exists.
        """
        if self._current_response:
            self._current_response.close()
            self._current_response = None
            logger.debug("LLM generation cancelled (connection closed)")

    async def health_check(self) -> bool:
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self._base_url}/health",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("status") == "ok"
        except Exception:
            return False
        return False

    async def tokenize(self, text: str) -> int:
        """Count tokens using llama.cpp's /tokenize endpoint.

        Returns token count. Uses the server's actual tokenizer.
        """
        try:
            session = await self._ensure_session()
            async with session.post(
                f"{self._base_url}/tokenize",
                json={"content": text},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                return len(data.get("tokens", []))
        except Exception as e:
            logger.warning("Tokenize failed, using estimate: %s", e)
            return len(text) // 4

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
