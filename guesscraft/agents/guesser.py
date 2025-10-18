"""Guesser agent that interrogates the host to find the topic."""
from __future__ import annotations

import json
from dataclasses import dataclass

from guesscraft.agents.base import AgentTurn, ConversationLog, LLMBackedAgent
from guesscraft.llm import ChatMessage, LLMClient


GUESSER_SYSTEM_PROMPT = """You are playing 20 Questions as the guesser.
You must decide whether to ask another question or make a final guess.
Always respond with JSON in the following format:
{
  "action": "ask" | "guess" | "finish",
  "utterance": "...",   # the natural language text you say out loud
  "confidence": number    # optional but preferred between 0 and 1
}
When you answer with "guess" the utterance must be your best guess for the
secret topic.  Do not reveal the hidden topic unless you are confident.
"""


@dataclass
class GuesserConfig:
    initial_hint: str | None = None


class GuesserAgent(LLMBackedAgent):
    """Guesser agent driven by an LLM."""

    def __init__(self, config: GuesserConfig, llm: LLMClient) -> None:
        super().__init__(name="guesser", system_prompt=GUESSER_SYSTEM_PROMPT, llm=llm)
        self._config = config

    def next_action(self, conversation: ConversationLog) -> AgentTurn:
        """Return the next question or guess based on the conversation so far."""

        hint = self._config.initial_hint or ""
        payload = [
            ChatMessage(
                role="system",
                content=(
                    "You are coordinating with a moderator. "
                    "Always produce valid JSON that can be parsed with json.loads."
                ),
            ),
            *conversation.as_chat_messages(),
        ]
        if hint:
            payload.append(
                ChatMessage(role="user", content=json.dumps({"hint": hint}))
            )
        raw = self._call_llm(payload)
        data = self._parse_json(raw)
        action = data.get("action")
        utterance = data.get("utterance")
        if action not in {"ask", "guess", "finish"}:
            raise ValueError(f"Guesser returned invalid action: {raw!r}")
        if not isinstance(utterance, str):
            raise ValueError(f"Guesser returned invalid utterance: {raw!r}")
        confidence = data.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Guesser returned invalid confidence value: {raw!r}"
                ) from exc
        return AgentTurn(action=action, content=utterance, confidence=confidence)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, object]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Guesser agent returned invalid JSON: {raw!r}") from exc
