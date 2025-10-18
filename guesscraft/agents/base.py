"""Core dataclasses and abstractions shared by Guesscraft agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Literal

from guesscraft.llm import ChatMessage, LLMClient


class GuessOutcome(Enum):
    """Represents the state of the guesser's latest action."""

    UNKNOWN = "unknown"
    CORRECT = "correct"
    INCORRECT = "incorrect"


@dataclass
class AgentTurn:
    """Structured output of the guesser agent."""

    action: Literal["ask", "guess", "finish"]
    content: str
    confidence: float | None = None


@dataclass
class HostReply:
    """Structured response from the host agent."""

    reply: str
    reveal_topic: bool = False
    topic: str | None = None


@dataclass
class ConversationEntry:
    """A single interaction between the guesser and the host."""

    role: Literal["guesser", "host"]
    content: str


@dataclass
class ConversationLog:
    """History of the on-going game."""

    entries: List[ConversationEntry] = field(default_factory=list)

    def append(self, role: Literal["guesser", "host"], content: str) -> None:
        self.entries.append(ConversationEntry(role=role, content=content))

    def as_chat_messages(self) -> List[ChatMessage]:
        mapping = {"guesser": "user", "host": "assistant"}
        return [
            ChatMessage(role=mapping[entry.role], content=entry.content)
            for entry in self.entries
        ]

    def summary(self) -> str:
        """Returns a human readable summary of the conversation."""

        lines = [
            f"{entry.role.capitalize()}: {entry.content.strip()}" for entry in self.entries
        ]
        return "\n".join(lines)


class LLMBackedAgent:
    """Base class that provides convenience methods for LLM prompts."""

    def __init__(self, name: str, system_prompt: str, llm: LLMClient) -> None:
        self.name = name
        self._system_prompt = system_prompt
        self._llm = llm

    def _call_llm(self, messages: Iterable[ChatMessage]) -> str:
        prompt = [ChatMessage(role="system", content=self._system_prompt)]
        prompt.extend(messages)
        return self._llm.chat(prompt)
