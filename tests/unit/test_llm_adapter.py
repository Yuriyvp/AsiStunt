"""Tests for LlamaCppLLM adapter — unit tests with mocked HTTP.

These tests mock aiohttp to test the adapter logic without a running server.
Live integration tests are done manually via the verification scripts.
"""
import asyncio
import json

import pytest

from voice_assistant.adapters.llamacpp_llm import LlamaCppLLM, DEFAULT_SAMPLING


class MockResponse:
    """Mock aiohttp response for SSE streaming."""

    def __init__(self, status=200, lines=None):
        self.status = status
        self._lines = lines or []
        self.content = self._async_iter()
        self._closed = False

    async def _async_iter(self):
        for line in self._lines:
            yield line.encode("utf-8") + b"\n"

    def raise_for_status(self):
        if self.status >= 400:
            raise Exception(f"HTTP {self.status}")

    def close(self):
        self._closed = True

    async def json(self):
        return {"status": "ok"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockSession:
    """Mock aiohttp.ClientSession."""

    def __init__(self, post_response=None, get_response=None):
        self._post_response = post_response
        self._get_response = get_response or MockResponse(200)
        self.closed = False
        self.last_post_payload = None

    async def post(self, url, json=None, timeout=None):
        self.last_post_payload = json
        return self._post_response

    def get(self, url, timeout=None):
        return self._get_response

    async def close(self):
        self.closed = True


def _make_sse_lines(tokens):
    """Create SSE lines from a list of token strings."""
    lines = []
    for token in tokens:
        data = {"choices": [{"delta": {"content": token}}]}
        lines.append(f"data: {json.dumps(data)}")
    lines.append("data: [DONE]")
    return lines


@pytest.mark.asyncio
async def test_health_check_ok():
    """Health check should return True when server responds ok."""
    llm = LlamaCppLLM()
    llm._session = MockSession(get_response=MockResponse(200))
    result = await llm.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_stream_yields_tokens():
    """Stream should yield individual tokens from SSE."""
    tokens = ["Hello", ", ", "world", "!"]
    sse_lines = _make_sse_lines(tokens)
    mock_resp = MockResponse(200, sse_lines)

    llm = LlamaCppLLM()
    session = MockSession(post_response=mock_resp)
    llm._session = session

    received = []
    async for token in llm.stream([{"role": "user", "content": "Hi"}]):
        received.append(token)

    assert received == tokens


@pytest.mark.asyncio
async def test_stream_sends_correct_payload():
    """Stream should send messages and sampling params."""
    mock_resp = MockResponse(200, ["data: [DONE]"])
    session = MockSession(post_response=mock_resp)

    llm = LlamaCppLLM()
    llm._session = session

    messages = [{"role": "user", "content": "test"}]
    async for _ in llm.stream(messages):
        pass

    assert session.last_post_payload["messages"] == messages
    assert session.last_post_payload["stream"] is True
    assert session.last_post_payload["temperature"] == DEFAULT_SAMPLING["temperature"]


@pytest.mark.asyncio
async def test_cancel_closes_response():
    """Cancel should close the current response."""
    mock_resp = MockResponse(200)
    llm = LlamaCppLLM()
    llm._current_response = mock_resp
    await llm.cancel()
    assert mock_resp._closed is True
    assert llm._current_response is None


@pytest.mark.asyncio
async def test_shutdown_closes_session():
    """Shutdown should close the aiohttp session."""
    session = MockSession()
    llm = LlamaCppLLM()
    llm._session = session
    await llm.shutdown()
    assert session.closed is True
    assert llm._session is None


@pytest.mark.asyncio
async def test_stream_handles_malformed_sse():
    """Should skip malformed SSE lines gracefully."""
    lines = [
        "data: not-json",
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        "data: [DONE]",
    ]
    mock_resp = MockResponse(200, lines)
    session = MockSession(post_response=mock_resp)

    llm = LlamaCppLLM()
    llm._session = session

    received = []
    async for token in llm.stream([{"role": "user", "content": "Hi"}]):
        received.append(token)

    assert received == ["ok"]


@pytest.mark.asyncio
async def test_stream_skips_empty_lines():
    """Should handle empty lines in SSE stream."""
    lines = [
        "",
        'data: {"choices": [{"delta": {"content": "hi"}}]}',
        "",
        "data: [DONE]",
    ]
    mock_resp = MockResponse(200, lines)
    session = MockSession(post_response=mock_resp)

    llm = LlamaCppLLM()
    llm._session = session

    received = []
    async for token in llm.stream([{"role": "user", "content": "Hi"}]):
        received.append(token)

    assert received == ["hi"]
