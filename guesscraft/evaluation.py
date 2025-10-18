"""Evaluation helpers for running multiple self-play games."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

from guesscraft.agents.guesser import GuesserAgent, GuesserConfig
from guesscraft.agents.host import HostAgent, HostConfig
from guesscraft.game import GameResult, GameRunner
from guesscraft.llm import LLMClient


@dataclass
class EvaluationSummary:
    games: Sequence[GameResult]

    @property
    def win_rate(self) -> float:
        if not self.games:
            return 0.0
        return sum(1 for game in self.games if game.success) / len(self.games)

    @property
    def average_turns(self) -> float:
        if not self.games:
            return 0.0
        return mean(game.turns_taken for game in self.games)


def run_benchmark(
    topics: Iterable[HostConfig],
    llm_host: LLMClient,
    llm_guesser: LLMClient,
    guesser_config: GuesserConfig | None = None,
    max_turns: int = 20,
) -> EvaluationSummary:
    """Play multiple games and collect statistics."""

    results: list[GameResult] = []
    for topic in topics:
        host = HostAgent(config=topic, llm=llm_host)
        guesser = GuesserAgent(config=guesser_config or GuesserConfig(), llm=llm_guesser)
        runner = GameRunner(host=host, guesser=guesser, max_turns=max_turns)
        results.append(runner.play())
    return EvaluationSummary(games=results)
