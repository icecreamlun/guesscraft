from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from .kb.kb import ObjectKB
from .kb.index import AttributeIndex
from .llm.client import LLMClient
from .agents.attribute_parser import AttributeParser
from .agents.host import HostRuleBased, HostLLM
from .agents.guesser import GuesserEntropy
from .agents.phrasing import GuesserPhraser
from .engine.game_engine import GameEngine, GameConfig


def run_one(config_path: str | None) -> Dict[str, Any]:
    # Load .env early to populate environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    # Load config
    config: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    model = config.get("model", os.getenv("MODEL_NAME", "gpt-4o-mini"))
    topic = config.get("topic", "apple")
    host_mode = config.get("host_mode", "rule_based")  # or "llm"

    # Build components
    kb = ObjectKB.from_seed()
    index = AttributeIndex(kb)
    llm = LLMClient(model=model)
    parser = AttributeParser(kb, llm)
    phraser = GuesserPhraser(llm)
    guesser = GuesserEntropy(kb, index, tau_guess_threshold=0.6, phraser=phraser)

    if host_mode == "llm":
        host = HostLLM(llm=llm, topic_name=topic)
        def answer_fn(q: str) -> Dict[str, Any]:
            return host.answer(q)
        def check_guess_fn(identifier: str) -> bool:
            # For LLM host, accept name match ignoring case
            return identifier.lower() == topic.lower()
    else:
        topic_id = topic if topic in kb.obj_by_id else "apple"
        host = HostRuleBased(kb=kb, attribute_parser=parser, topic_object_id=topic_id)
        def answer_fn(q: str) -> Dict[str, Any]:
            return host.answer(q)
        def check_guess_fn(identifier: str) -> bool:
            return identifier == host.topic_object_id or identifier.lower() == kb.obj_by_id[host.topic_object_id].name.lower()

    engine = GameEngine(
        ask_fn=guesser.next_question,
        answer_fn=answer_fn,
        update_fn=lambda payload: (
            guesser.update_with_answer(
                attribute_id=str(payload.get("justification", "")).split("=")[0] if payload.get("justification") else str(payload.get("attribute_id", "")),
                answer=str(payload.get("answer", "unknown")),
            )
        ),
        should_guess_fn=guesser.should_guess,
        make_guess_fn=guesser.make_guess,
        check_guess_fn=check_guess_fn,
        config=GameConfig(max_turns=int(config.get("max_turns", 20))),
    )

    result = engine.play()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="20Q Self-Play")
    sub = parser.add_subparsers(dest="cmd")
    one = sub.add_parser("one", help="Play one game")
    one.add_argument("--config", required=False, help="Path to YAML config")
    many = sub.add_parser("many", help="Play many games and aggregate metrics")
    many.add_argument("--config", required=False, help="Path to YAML config")
    many.add_argument("--n", type=int, default=10, help="Number of games")
    many.add_argument("--out", required=True, help="Output directory")
    
    manyc = sub.add_parser("many-concurrent", help="Play many games concurrently and aggregate metrics")
    manyc.add_argument("--config", required=False, help="Path to YAML config")
    manyc.add_argument("--n", type=int, default=10, help="Number of games")
    manyc.add_argument("--out", required=True, help="Output directory")
    manyc.add_argument("--concurrency", type=int, default=4, help="Parallel workers")

    args = parser.parse_args()
    if args.cmd == "one":
        result = run_one(args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.cmd == "many":
        from .experiments.runner import run_many
        metrics = run_many(args.config, args.n, args.out)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.cmd == "many-concurrent":
        from .experiments.runner import run_many_concurrent
        metrics = run_many_concurrent(args.config, args.n, args.out, args.concurrency)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


