# Guesscraft

Guesscraft is a light-weight playground for running games of 20 Questions
between language model agents without relying on third-party agent frameworks.
The repository focuses on structuring the orchestration loop, handling
structured outputs from the agents, and making it easy to run experiments with
self-play.

## Project structure

- `guesscraft/llm.py` – minimal chat client abstractions, including a stubbed
  client for offline testing and a thin wrapper for the OpenAI Responses API.
- `guesscraft/agents/` – host and guesser agent implementations that enforce a
  JSON protocol to keep the game reliable.
- `guesscraft/game.py` – orchestrates a single game, making sure the guesser is
  limited to 20 turns and asking the host to reveal the topic when appropriate.
- `guesscraft/evaluation.py` – helper utilities to run multiple games and collect
  simple statistics such as win rate and average number of turns.

## Running a game

The package is framework-free and only requires an `LLMClient` implementation.
The snippet below demonstrates how to hook up OpenAI models (set
`OPENAI_API_KEY` in your environment first):

```python
from guesscraft.agents.host import HostAgent, HostConfig
from guesscraft.agents.guesser import GuesserAgent, GuesserConfig
from guesscraft.game import GameRunner
from guesscraft.llm import OpenAIChatClient

host_llm = OpenAIChatClient(model="gpt-4o-mini")
guesser_llm = OpenAIChatClient(model="gpt-4o-mini")

host = HostAgent(HostConfig(topic="koala"), llm=host_llm)
guesser = GuesserAgent(GuesserConfig(), llm=guesser_llm)

result = GameRunner(host, guesser).play()
print(result.success, result.turns_taken)
print(result.transcript.summary())
```

## Evaluating agents

`guesscraft.evaluation.run_benchmark` makes it easy to pit agents against a list
of topics and gather aggregate metrics.  Because the project ships with a
`SequentialStubClient` you can unit test prompt tweaks by scripting the model
outputs without making network calls.

## Development

Install dependencies (only `pytest` is required for the test suite):

```bash
pip install -r requirements-dev.txt
```

Run the tests with:

```bash
pytest
```
