"""Game orchestration for 20 Questions self-play."""
from __future__ import annotations

from dataclasses import dataclass

from guesscraft.agents.base import (
    AgentTurn,
    ConversationLog,
    GuessOutcome,
    HostReply,
)
from guesscraft.agents.guesser import GuesserAgent
from guesscraft.agents.host import HostAgent


@dataclass
class GameResult:
    """Summary of a self-play session."""

    success: bool
    turns_taken: int
    transcript: ConversationLog
    final_topic: str


class GameRunner:
    """Coordinates a game between a host and a guesser."""

    def __init__(self, host: HostAgent, guesser: GuesserAgent, max_turns: int = 20):
        self.host = host
        self.guesser = guesser
        self.max_turns = max_turns

    def play(self) -> GameResult:
        conversation = ConversationLog()
        topic_revealed: str | None = None

        turns_taken = 0
        for turn_index in range(1, self.max_turns + 1):
            turns_taken = turn_index
            guesser_turn = self._next_turn(conversation)
            conversation.append("guesser", guesser_turn.content)

            outcome = GuessOutcome.UNKNOWN
            should_end = False
            if guesser_turn.action == "guess":
                outcome = self._evaluate_guess(guesser_turn)
                should_end = outcome is GuessOutcome.CORRECT
            elif guesser_turn.action == "finish":
                should_end = True

            host_reply = self._host_reply(conversation, guesser_turn, outcome)
            conversation.append("host", host_reply.reply)
            if host_reply.reveal_topic and host_reply.topic:
                topic_revealed = host_reply.topic
                return GameResult(
                    success=outcome is GuessOutcome.CORRECT,
                    turns_taken=turn_index,
                    transcript=conversation,
                    final_topic=topic_revealed,
                )

            if should_end:
                topic_revealed = topic_revealed or host_reply.topic or ""
                break

        # If we exit the loop without revealing the topic the host should still do so
        if topic_revealed is None:
            conversation.append("guesser", "I give up.")
            final_reply = self._host_reply(
                conversation,
                AgentTurn(action="finish", content="I give up.", confidence=None),
                GuessOutcome.INCORRECT,
            )
            conversation.append("host", final_reply.reply)
            topic_revealed = final_reply.topic or ""

        return GameResult(
            success=False,
            turns_taken=turns_taken or self.max_turns,
            transcript=conversation,
            final_topic=topic_revealed,
        )

    def _next_turn(self, conversation: ConversationLog) -> AgentTurn:
        return self.guesser.next_action(conversation)

    def _evaluate_guess(self, turn: AgentTurn) -> GuessOutcome:
        topic = self.host.topic
        if turn.content.strip().lower() == topic.strip().lower():
            return GuessOutcome.CORRECT
        return GuessOutcome.INCORRECT

    def _host_reply(
        self, conversation: ConversationLog, turn: AgentTurn, outcome: GuessOutcome
    ) -> HostReply:
        return self.host.answer(conversation, turn.content, outcome)
