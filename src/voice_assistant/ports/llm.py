"""Abstract base class for LLM adapters."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMPort(ABC):
    @abstractmethod
    async def stream(self, messages: list[dict], sampling: dict | None = None,
                     thinking: bool = False) -> AsyncIterator[str]:
        """Stream tokens from the LLM. Yields token strings."""
        ...

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel in-flight generation."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        ...
