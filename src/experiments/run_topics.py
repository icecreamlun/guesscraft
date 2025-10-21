from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from .runner import run_many


def run_all_topics(topics_dir: str, out_base: str, workers: int = 10) -> Dict[str, Any]:
    os.makedirs(out_base, exist_ok=True)
    config_paths = sorted(glob.glob(os.path.join(topics_dir, "*.yaml")))
    if not config_paths:
        raise FileNotFoundError(f"No topic configs found under {topics_dir}")

    results: List[Tuple[str, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {}
        for cfg in config_paths:
            name = os.path.splitext(os.path.basename(cfg))[0]
            out_dir = os.path.join(out_base, name)
            futures[pool.submit(run_many, cfg, 1, out_dir)] = (name, out_dir)
        for fut in as_completed(futures):
            name, out_dir = futures[fut]
            metrics = fut.result()
            results.append((name, metrics))

    # Aggregate overall
    total_games = 0
    total_wins = 0
    per_topic: Dict[str, Any] = {}
    for name, m in results:
        wins = int(m.get("wins", 0))
        games = int(m.get("games", 0))
        total_wins += wins
        total_games += games
        per_topic[name] = {
            "wins": wins,
            "games": games,
            "win_rate": (wins / games) if games else 0.0,
        }

    summary = {
        "topics": len(results),
        "wins": total_wins,
        "games": total_games,
        "win_rate": (total_wins / total_games) if total_games else 0.0,
        "per_topic": per_topic,
    }

    with open(os.path.join(out_base, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one game per topic concurrently and aggregate accuracy")
    parser.add_argument("--topics_dir", default="experiments/topics", help="Directory of topic YAML configs")
    parser.add_argument("--out_base", default="runs/multi", help="Base output directory")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    args = parser.parse_args()

    summary = run_all_topics(args.topics_dir, args.out_base, args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


