from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from .llm.client import LLMClient
from .agents.host import HostLLM
from .agents.guesser_llm import GuesserLLM
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

    model = config.get("model", os.getenv("MODEL_NAME", "gpt-5"))
    # Allow separate models per agent
    model_host = config.get("model_host", os.getenv("MODEL_HOST", model))
    model_guesser = config.get("model_guesser", os.getenv("MODEL_GUESSER", model))

    topic = config.get("topic", "chicken")

    # Build components (LLM-only)
    llm_host = LLMClient(model=model_host)
    host = HostLLM(llm=llm_host, topic_name=topic)
    llm_guesser = LLMClient(model=model_guesser)
    guesser = GuesserLLM(llm=llm_guesser)

    def answer_fn(q: str) -> Dict[str, Any]:
        return host.answer(q)

    def _normalize_tokens(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        return tokens

    def _macro_class(token: str) -> str | None:
        # Minimal macro-class mapping to allow hierarchical acceptance (no specific entities)
        vehicle = {
            "vehicle", "car", "truck", "bus", "van", "automobile", "taxi", "suv", "sedan", "minivan", "pickup"
        }
        animal = {"animal", "dog", "cat", "canine", "feline"}
        fruit = {"fruit"}
        structure = {"structure", "building", "tower", "monument"}
        device = {"device", "computer", "laptop", "phone"}
        if token in vehicle:
            return "vehicle"
        if token in animal:
            return "animal"
        if token in fruit:
            return "fruit"
        if token in structure:
            return "structure"
        if token in device:
            return "device"
        return None

    def check_guess_fn(identifier: str) -> bool:
        # Flexible match: exact normalized string; token containment; or macro-class membership
        guess_tokens = _normalize_tokens(identifier)
        topic_tokens = _normalize_tokens(topic)
        norm_guess = " ".join(guess_tokens)
        norm_topic = " ".join(topic_tokens)
        if norm_guess == norm_topic:
            return True
        if len(topic_tokens) == 1:
            if topic_tokens[0] in guess_tokens:
                return True
        else:
            sig = [t for t in topic_tokens if len(t) >= 3]
            if all(t in guess_tokens for t in (sig if sig else topic_tokens)):
                return True
        # Macro-class acceptance: if both guess and topic map to the same macro class
        guess_classes = {c for t in guess_tokens if (c := _macro_class(t))}
        topic_classes = {c for t in topic_tokens if (c := _macro_class(t))}
        if guess_classes and topic_classes and guess_classes.intersection(topic_classes):
            return True
        return False

    engine = GameEngine(
        ask_fn=guesser.next_question,
        answer_fn=answer_fn,
        update_fn=lambda payload: guesser.update_with_answer(payload),
        should_guess_fn=guesser.should_guess,
        make_guess_fn=guesser.make_guess,
        check_guess_fn=check_guess_fn,
        config=GameConfig(max_turns=int(config.get("max_turns", 20))),
    )

    result = engine.play()
    # attach secret topic for console-only visibility
    result["secret_topic"] = topic
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
        # Print final outcome first for quick visibility, without exposing to the guesser agent
        outcome = {
            "result": result.get("result"),
            "turns_used": result.get("turns_used"),
            "answer": result.get("secret_topic"),
        }
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        # Then print the full trace (events) for auditing
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


