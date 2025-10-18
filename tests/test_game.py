from guesscraft.agents.guesser import GuesserAgent, GuesserConfig
from guesscraft.agents.host import HostAgent, HostConfig
from guesscraft.game import GameRunner
from guesscraft.llm import SequentialStubClient


def test_game_runner_success_with_stub_llms():
    guesser_stub = SequentialStubClient(
        [
            '{"action": "ask", "utterance": "Is it furry?"}',
            '{"action": "guess", "utterance": "koala"}',
        ]
    )
    host_stub = SequentialStubClient(
        [
            '{"reply": "Yes, it is.", "reveal_topic": false}',
            '{"reply": "Correct!", "reveal_topic": true, "topic": "koala"}',
        ]
    )

    host = HostAgent(HostConfig(topic="koala"), llm=host_stub)
    guesser = GuesserAgent(GuesserConfig(), llm=guesser_stub)

    result = GameRunner(host, guesser, max_turns=5).play()

    assert result.success is True
    assert result.final_topic == "koala"
    # 2 guesser turns and 2 host replies
    assert len(result.transcript.entries) == 4


def test_game_runner_failure_reveals_topic():
    guesser_stub = SequentialStubClient(
        [
            '{"action": "guess", "utterance": "dog"}',
        ]
    )
    host_stub = SequentialStubClient(
        [
            '{"reply": "No, that is not it.", "reveal_topic": false}',
            '{"reply": "The topic was koala.", "reveal_topic": true, "topic": "koala"}',
        ]
    )

    host = HostAgent(HostConfig(topic="koala"), llm=host_stub)
    guesser = GuesserAgent(GuesserConfig(), llm=guesser_stub)

    result = GameRunner(host, guesser, max_turns=1).play()

    assert result.success is False
    assert result.final_topic == "koala"
    # first guess + host reply + final give up + final reveal
    assert len(result.transcript.entries) == 4
