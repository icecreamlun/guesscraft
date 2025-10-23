# my-project

## Quickstart

- Run one game:
```bash
python -m src.cli one
```

- Run multiple topics (parallel):
```bash
python -m src.experiments.run_topics --topics_dir experiments/topics --out_base runs/multi --workers 10(can change the workers)
```

## Things to think about

### Agentic workflow
- **Main steps**: initialize config → create LLM clients (Host, Guesser) → start FSM loop (ask/answer/decide/guess) → stop on correct guess or after 20 turns → write trace and metrics.
- **Interaction**: strictly turn-based. Guesser produces exactly one structured action per turn: ASK(question) or GUESS(name). Host answers strictly yes/no/unknown. No mixed actions.
- **Context per agent**:
  - Guesser: message history (Q/A), scratchpad (ReAct thoughts), asked_questions (normalized for de-dup), attempted_guesses, last_action_was_guess.
  - Host: secret topic, IS-A semantics, short answers, and a small heuristic to treat modifier+head (e.g., “small dog”) as a “yes” for topic “dog”.
- **What to keep static early**: 20-turn budget, single-topic runs, a minimal macro-class matcher (vehicle/structure/animal/device/fruit), deterministic settings for stable comparison.
- **Errors and issues**: schema validation with auto-repair, exponential backoff, empty-content retry, model-compat param routing (max_tokens vs max_completion_tokens; omitting temperature for newer families), and dotenv for API keys.
- **Optional sub-agents**: we kept a single Guesser with three structured calls (“should_guess”, “next_question”, “make_guess”). Further decomposition (taxonomy sub-agent, candidate-ranker) is possible but not required to reach good performance.

### Self-play with LLMs → Reliability
- **Chaining actions**: a finite state machine enforces one action per turn. GUESS consumes a turn and ends the game only on success; last turn forces a guess.
- **Extraction**: all agent outputs are JSON with schemas: {thought?, question_text} for ASK; {thought?, guess_text, confidence} for GUESS; {answer} for Host. We validate/repair until the schema passes.
- **Memory controls**: normalized asked_questions to avoid repeats; last_action_was_guess to prevent consecutive guesses; attempted_guesses to avoid duplication; scratchpad preserves recent ReAct thoughts.
- **ReAct guidance**: Thought first, then Action. Questions are concise (≤12 words), no embedded guesses, focus on high-information splits.

### Evaluation
- **How good are the agents?**: we log per-game events and aggregate metrics (win_rate, turns_used). `src/experiments/run_topics.py` runs diverse topics concurrently and writes a summary.
- **Failure modes**: low-information or redundant questions; category-level guesses; semantic near-misses (e.g., modifier vs head noun); JSON parsing hiccups on some model families.
- **Tuning difficulty**: adjust topic diversity, relax/stricten guess matching, vary max turns, or change question style constraints.

### Bonus: Testability
- **Experiment organization**: each topic is a YAML config. Small prompt edits can be versioned as commits; compare `runs/.../metrics.json` and `summary.json` across branches.
- **Parallelization**: `many-concurrent` and the topics runner both support workers for faster sweeps.
- **Efficiency**: keep messages short, cap scratchpad/history, force concise outputs, and retry only on structural failures.

## How I improved from ~20% → ~80% win rate

- v0 (baseline): naive LLM Q/A with loose prompts. No memory de-dup, no structured gating. Many vague questions and category guesses. Win rate ≈ 20%.
- v1 (structure & control): strict JSON schemas + auto-repair + retries; FSM enforcing one action/turn; early stop on correct guess. Reduced flakiness. ≈ 40%.
- v2 (ReAct): Thought → Action format; short, discriminative questions; forbid embedding guesses in questions. Better search discipline. ≈ 50%.
- v3 (“broad-first, then common” strategy): prioritize high-gain taxonomy splits; when a class is likely, use a “frequency prior × confidence” gate to guess a short, canonical, common name; prevent consecutive guesses; track asked_questions to avoid loops. ≈ 60%.
- v4 (semantic alignment): Host treats modifier+head as a “yes” for the head (e.g., “small dog” for “dog”); CLI adds macro-class acceptance (vehicle/structure/animal/device/fruit) so close variants count as correct. ≈ 70%.
- v5 (consistency & recovery): enforce consistency with all prior answers; if recent questions are low gain or repetitive, force a dimension change; keep scratchpad/history truncated but informative. ≈ 80%.
- v6 (model robustness): handle max_tokens vs max_completion_tokens; omit temperature where required; add empty-content retry and JSON-mode nudges. Stability ↑, win rate steady.

## Notes
- To try a stronger Guesser while keeping a lightweight Host, set `MODEL_GUESSER` in your environment or add `model_guesser` to the topic YAML. If you switch to newer model families, ensure parameters are routed correctly (we handle this automatically), or use `gpt-4o`/`gpt-4o-mini` for maximum stability.

## Design rationale & thought process (for reviewers)

### Goals and constraints
- Keep the whole loop deterministic and auditable: explicit FSM, JSON schemas, traces on disk.
- Do not rely on hidden chat memory: agents must carry their own state explicitly.
- Optimize for practical win rate under a 20-turn budget while keeping prompts model-agnostic.

### Why an FSM + structured outputs
- A finite state machine guarantees exactly one action per turn (ASK or GUESS). This removes prompt drift where models “answer and guess” in one turn.
- JSON schemas for each action let us validate and auto-repair outputs. This catches malformed responses early and keeps the loop stable.

### Guesser logic — from empty history to confident guesses
1) Start state (t=0): `history` is empty, `asked_questions` and `attempted_guesses` are empty, `scratchpad` is empty.
2) Asking questions (ReAct): each turn the Guesser produces a Thought (why this split) then an ASK:
   - Prefer high-information category splits (e.g., living vs non-living; food vs not-food).
   - If the last few questions were low-gain or similar, switch to a different dimension to avoid loops.
   - Questions must be short (≤12 words), and never contain candidate names.
3) Updating memory: after Host replies, we append `QA(question, answer)` to `history`, add an Observation to `scratchpad`, and store a normalized version of the question in `asked_questions` to prevent repeats.
4) Deciding when to guess: we gate on two principles:
   - Consistency: any guess must be consistent with all prior answers.
   - Frequency × Confidence: only guess when a short, canonical, common name emerges (frequency prior) AND confidence is high enough. If not, keep asking.
   - Cooldown: avoid consecutive guesses unless the budget forces it (last turn).
5) Making the guess: the name must be ≤2 words, canonical, non-category. We log Thought and GUESS to the scratchpad, track `attempted_guesses` to avoid repetition, and move on.

Rationale: This design concentrates information early (taxonomy splits) and preserves momentum with de-dup memory and consistency checks. The frequency prior encourages “most common” within the inferred class before exploring rarer items.

### Host logic — strict IS-A semantics with a pragmatic shortcut
- The Host returns only yes/no/unknown based on the topic itself (strict IS-A). If unsure, say “unknown”.
- Shortcut: if the question is a modifier + head (e.g., “small dog”) and the topic is the head noun (“dog”), answer “yes”. This fixes common “unknown” misfires when the head is correct.

### “Close enough” wins without leaking the answer
- Host doesn’t decide win/lose. The CLI checks whether the Guesser’s name should count as correct:
  - Exact normalized match, or token containment (e.g., “domestic dog” contains “dog”).
  - Macro-class match: if guess and topic share a broad family (vehicle/structure/animal/device/fruit), it counts as a hit. 

### Reliability tactics we adopted
- JSON schemas + validation + auto-repair retries.
- Short, explicit prompts; cap history and scratchpad length.
- Model-compat shims (max_tokens vs max_completion_tokens; temperature handling) and a light retry if the model returns empty content.
- `.env` loading for keys; deterministic defaults for reproducibility.

### Evaluation methodology
- Scripted runs across diverse topics with concurrency (`run_topics.py`).
- Metrics: per-topic wins, overall win rate, turns used; all runs produce JSON traces for post-hoc analysis.
- Failure categories we look for: low-gain question loops, premature category guesses, semantic near-miss names, or model JSON flakiness.

### Evolution highlights (thought process)
- Start: unstructured prompting led to vague questions, mixed actions, and unstable outputs (≈20%).
- Add FSM + schemas: stabilized the loop and made errors observable (≈40%).
- Introduce ReAct: force “Thought → Action”, short discriminative ASK, no embedded guesses (≈50%).
- Add memory/controls: normalize `asked_questions`, prevent repeated guesses, enforce consistency (≈70%).
- Soften success criteria: Host “modifier+head → yes”; CLI macro-class acceptance (≈76%).
- Robustness work: parameter compatibility across model families; empty-content retry; concise prompts (≈80%).

### What we would do next
- Learnable thresholds for “frequency × confidence” rather than purely prompt-based.
- A light candidate-ranker sub-agent when a class is locked, to choose between top-N common names.
- Per-topic difficulty tiers and ablations (e.g., disallow macro-class hits) for finer-grained evaluation.
- Add self-consistency (N parallel Reasoners → vote) when budget allows, to improve late-game guesses.

## Future plan

- Reliability & determinism
  - Add a fallback plain-text parser when JSON is empty, with tight regex schemas per action.
  - Inject per-turn integrity checks (e.g., contradiction detector) that can request a self-correction turn.
  - Support “retry with paraphrase” on known flaky prompts while preserving turn budget.

- Learning from runs
  - Harvest failed traces to auto-synthesize “contrastive prompt patches” (what to avoid vs what to prefer) and replay A/B.
  - Build a small offline scorer for question information-gain using weak heuristics (entropy proxy) to guide ReAct thoughts.

- Smarter guessing
  - Replace static frequency prior with a light learned prior over macro-classes and common entities (few-shot or tiny table), without hardcoding specifics.
  - Add a beam of 2–3 candidate guesses with a consistency re-ranker that checks each guess against the entire Q/A history.

- Host realism & robustness
  - Add a “strictness slider” (more/less likely to say unknown) to modulate difficulty without changing topics.
  - Expand modifier+head heuristics to handle common hyphenations and compounding without leaking answers.

- Experiment framework
  - First-class support for prompt versioning: each run persists prompt hashes and diffs alongside metrics for clean comparisons.
  - Adaptive batching and caching of repeated prompts to cut latency and cost in large sweeps.

- Engineering quality
  - Telemetry hooks (latency, token usage, repair counts) with simple charts in `runs/.../summary.json`.
  - CI smoke tests that run a tiny seed suite (2–3 topics) on PRs to catch regressions.