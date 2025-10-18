"""Host agent that answers yes/no questions about a hidden topic."""
from __future__ import annotations

import json
from dataclasses import dataclass

from guesscraft.agents.base import ConversationLog, GuessOutcome, HostReply, LLMBackedAgent
from guesscraft.llm import ChatMessage, LLMClient


HOST_SYSTEM_PROMPT = """You are the host in a game of 20 Questions.
You have been given a secret topic and the guesser is trying to figure it out.
Answer questions truthfully with short sentences.  Reply with a JSON object with
these keys:
- "reply": string, the natural language response you say to the guesser
- "reveal_topic": boolean, set to true only when the guesser has guessed
  correctly or the moderator instructs you to reveal the topic
- "topic": string, include the exact topic only when reveal_topic is true
If the moderator message indicates the guess was incorrect you should gently
correct the guesser and encourage them to continue.
"""


@dataclass
class HostConfig:
    topic: str
    topic_description: str | None = None


class HostAgent(LLMBackedAgent):
    """Host backed by an LLM that reasons about the topic."""

    def __init__(self, config: HostConfig, llm: LLMClient) -> None:
        super().__init__(name="host", system_prompt=HOST_SYSTEM_PROMPT, llm=llm)
        self._config = config

    @property
    def topic(self) -> str:
        return self._config.topic

    def answer(
        self,
        conversation: ConversationLog,
        latest_turn: str,
        outcome: GuessOutcome,
    ) -> HostReply:
        """Return a response to the guesser's latest message.

        Parameters
        ----------
        conversation:
            Full conversation log up to and including the guesser's latest turn.
        latest_turn:
            The guesser's latest utterance.
        outcome:
            Whether the latest turn was a correct guess, an incorrect guess, or a
            regular question.
        """

        status_hint = {
            GuessOutcome.UNKNOWN: "The guesser just asked a question.",
            GuessOutcome.CORRECT: "The guesser just correctly guessed the topic.",
            GuessOutcome.INCORRECT: "The guesser just made an incorrect guess.",
        }[outcome]

        context = (
            f"Secret topic: {self._config.topic}."
            + (
                f" Description: {self._config.topic_description}."
                if self._config.topic_description
                else ""
            )
        )

        payload = [
            ChatMessage(
                role="system",
                content="You are coordinating with a moderator. Always respond in JSON.",
            ),
            *conversation.as_chat_messages(),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "moderator": status_hint,
                        "latest_turn": latest_turn,
                        "topic": self._config.topic,
                        "context": context,
                    }
                ),
            ),
        ]
        raw = self._call_llm(payload)
        data = self._parse_json(raw)
        return HostReply(
            reply=data.get("reply", ""),
            reveal_topic=bool(data.get("reveal_topic")),
            topic=data.get("topic"),
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, object]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Host agent returned invalid JSON: {raw!r}") from exc
