from __future__ import annotations

from typing import Any, Dict, List


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    wins = sum(1 for r in results if r.get("result") == "win")
    turns = [int(r.get("turns_used", 0)) for r in results]
    turns_wins = [int(r.get("turns_used", 0)) for r in results if r.get("result") == "win"]
    mean_turns = (sum(turns) / total) if total else 0.0
    mean_turns_wins = (sum(turns_wins) / len(turns_wins)) if turns_wins else 0.0
    return {
        "games": total,
        "wins": wins,
        "win_rate": (wins / total) if total else 0.0,
        "mean_turns": mean_turns,
        "mean_turns_wins": mean_turns_wins,
    }


