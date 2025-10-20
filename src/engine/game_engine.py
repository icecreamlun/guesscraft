from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .types import AgentRole, Event, YesNo


@dataclass
class GameConfig:
    max_turns: int = 20


class GameEngine:
    """Minimal FSM: Guesser asks → Host answers → Update → Decide ask/guess → End."""

    def __init__(
        self,
        ask_fn: Callable[[], Optional[Dict[str, object]]],
        answer_fn: Callable[[str], Dict[str, object]],
        update_fn: Callable[[Dict[str, object]], None],
        should_guess_fn: Callable[[int, int], bool],
        make_guess_fn: Callable[[], Optional[Dict[str, object]]],
        check_guess_fn: Callable[[str], bool],
        config: Optional[GameConfig] = None,
    ):
        self.ask_fn = ask_fn
        self.answer_fn = answer_fn
        self.update_fn = update_fn
        self.should_guess_fn = should_guess_fn
        self.make_guess_fn = make_guess_fn
        self.check_guess_fn = check_guess_fn
        self.config = config or GameConfig()
        self.events: List[Event] = []

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def play(self) -> Dict[str, object]:
        used = 0
        for turn in range(1, self.config.max_turns + 1):
            # Decide ask vs. guess
            if self.should_guess_fn(used, self.config.max_turns):
                guess_payload = self.make_guess_fn()
                if not guess_payload:
                    return {"result": "error", "reason": "no_guess_available"}
                self.events.append(
                    Event(
                        turn=turn,
                        agent=AgentRole.GUESSER,
                        action=guess_payload,
                        timestamp_ms=self._now_ms(),
                        latency_ms=None,
                        tokens_in=None,
                        tokens_out=None,
                    )
                )
                used += 1
                # Check outcome
                oid = (guess_payload.get("object_id") or "") if isinstance(guess_payload, dict) else ""
                name = (guess_payload.get("object_name") or "") if isinstance(guess_payload, dict) else ""
                success = False
                if oid:
                    success = self.check_guess_fn(oid)
                elif name:
                    success = self.check_guess_fn(name)
                return {
                    "result": "win" if success else "lose",
                    "turns_used": used,
                    "guess": guess_payload,
                    "events": [e.__dict__ for e in self.events],
                }

            # Ask a question
            ask_payload = self.ask_fn()
            if not ask_payload:
                # If no question available, force a guess next iteration
                # Attempt immediate guess to avoid wasting a turn
                guess_payload = self.make_guess_fn()
                if guess_payload:
                    self.events.append(
                        Event(
                            turn=turn,
                            agent=AgentRole.GUESSER,
                            action=guess_payload,
                            timestamp_ms=self._now_ms(),
                            latency_ms=None,
                            tokens_in=None,
                            tokens_out=None,
                        )
                    )
                    used += 1
                    oid = (guess_payload.get("object_id") or "") if isinstance(guess_payload, dict) else ""
                    name = (guess_payload.get("object_name") or "") if isinstance(guess_payload, dict) else ""
                    success = False
                    if oid:
                        success = self.check_guess_fn(oid)
                    elif name:
                        success = self.check_guess_fn(name)
                    return {
                        "result": "win" if success else "lose",
                        "turns_used": used,
                        "guess": guess_payload,
                        "events": [e.__dict__ for e in self.events],
                    }
                return {"result": "error", "reason": "no_questions_and_no_guess"}

            self.events.append(
                Event(
                    turn=turn,
                    agent=AgentRole.GUESSER,
                    action=ask_payload,
                    timestamp_ms=self._now_ms(),
                    latency_ms=None,
                    tokens_in=None,
                    tokens_out=None,
                )
            )
            used += 1

            # Host answers
            q_text = str(ask_payload.get("question_text", ""))
            answer_payload = self.answer_fn(q_text)
            self.events.append(
                Event(
                    turn=turn,
                    agent=AgentRole.HOST,
                    action=answer_payload,
                    timestamp_ms=self._now_ms(),
                    latency_ms=None,
                    tokens_in=None,
                    tokens_out=None,
                )
            )

            # Update guesser state
            self.update_fn(answer_payload)

        # If loop ends, no win
        return {
            "result": "lose",
            "turns_used": used,
            "events": [e.__dict__ for e in self.events],
        }


