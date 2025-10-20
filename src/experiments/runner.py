from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from ..cli import run_one
from .metrics import aggregate_metrics


def run_many(config_path: str | None, n_games: int, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for i in range(n_games):
        res = run_one(config_path)
        results.append(res)
        with open(os.path.join(out_dir, f"game_{i:04d}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    metrics = aggregate_metrics(results)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

def run_many_concurrent(config_path: str | None, n_games: int, out_dir: str, concurrency: int = 4) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_one, config_path) for _ in range(n_games)]
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            results.append(res)
            with open(os.path.join(out_dir, f"game_{i:04d}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
    metrics = aggregate_metrics(results)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


