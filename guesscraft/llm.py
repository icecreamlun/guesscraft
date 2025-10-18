"""LLM client abstractions used by the Guesscraft agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol


@dataclass
class ChatMessage:
    """Represents a message passed to a chat completion model."""

    role: str
    content: str


class LLMClient(Protocol):
    """Protocol for chat style large language model clients."""

    def chat(self, messages: Iterable[ChatMessage]) -> str:
        """Generate the assistant reply for a sequence of messages."""


class OpenAIChatClient:
    """Thin wrapper over the OpenAI responses API.

    The dependency on ``openai`` is optional; the class is only imported when an
    instance is created.  This keeps the package light-weight for environments
    where an API key is not available and allows the unit tests to rely on
    deterministic stub clients.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in user envs
            raise RuntimeError(
                "The 'openai' package is required to use OpenAIChatClient."
            ) from exc

        self._client = OpenAI()
        self._model = model

    def chat(self, messages: Iterable[ChatMessage]) -> str:  # pragma: no cover
        formatted = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        response = self._client.responses.create(model=self._model, input=formatted)
        # The Responses API may return multiple items; we join them into a single
        # string to keep the rest of the code simple.
        parts: List[str] = []
        for item in response.output or []:
            if item.type == "message":
                for content_part in item.message.get("content", []):
                    if content_part.get("type") == "output_text":
                        parts.append(content_part.get("text", ""))
        return "".join(parts)


class SequentialStubClient:
    """A deterministic client that returns pre-scripted replies.

    The client is initialised with an iterable of strings.  Each call to
    :meth:`chat` pops and returns the next reply.  This is useful for unit tests
    and for demonstrating the orchestration layer without relying on networked
    models.
    """

    def __init__(self, replies: Iterable[str]):
        self._replies = list(replies)
        self._index = 0

    def chat(self, messages: Iterable[ChatMessage]) -> str:
        if self._index >= len(self._replies):
            raise RuntimeError("SequentialStubClient has been exhausted.")
        reply = self._replies[self._index]
        self._index += 1
        return reply


__all__ = [
    "ChatMessage",
    "LLMClient",
    "OpenAIChatClient",
    "SequentialStubClient",
]
