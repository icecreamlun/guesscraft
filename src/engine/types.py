from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


class AgentRole(str, Enum):
	HOST = "host"
	GUESSER = "guesser"


class GuesserActionType(str, Enum):
	ASK_QUESTION = "ask_question"
	MAKE_GUESS = "make_guess"


class HostActionType(str, Enum):
	ANSWER_YES_NO = "answer_yes_no"


YesNo = Literal["yes", "no", "unknown"]


@dataclass
class GuesserAskQuestion:
	type: Literal["ask_question"]
	question_text: str
	attribute_id: Optional[str]
	confidence: float


@dataclass
class GuesserMakeGuess:
	type: Literal["make_guess"]
	object_id: Optional[str]
	object_name: Optional[str]
	confidence: float


GuesserAction = Union[GuesserAskQuestion, GuesserMakeGuess]


@dataclass
class HostAnswerYesNo:
	type: Literal["answer_yes_no"]
	answer: YesNo
	justification: Optional[str]
	consistency_score: float


HostAction = HostAnswerYesNo


@dataclass
class Event:
	turn: int
	agent: AgentRole
	action: Dict[str, Any]
	timestamp_ms: int
	latency_ms: Optional[int]
	tokens_in: Optional[int]
	tokens_out: Optional[int]


