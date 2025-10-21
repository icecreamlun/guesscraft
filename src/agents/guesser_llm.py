from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient
from ..llm.schema import object_schema, string_schema, boolean_schema, number_schema


@dataclass
class QA:
    question: str
    answer: str  # "yes" | "no" | "unknown"


@dataclass
class GuesserLLMState:
    history: List[QA] = field(default_factory=list)
    last_question_text: Optional[str] = None
    attempted_guesses: List[str] = field(default_factory=list)
    scratchpad: List[str] = field(default_factory=list)  # ReAct trace: Thought / Action / Observation


class GuesserLLM:
    """LLM-only guesser that maintains a QA history and decides whether to ask or guess.

    - should_guess: ask the model (structured) whether to guess now given history and remaining turns
    - next_question: ask the model (structured) for the next yes/no question
    - make_guess: ask the model (structured) for the best current guess
    - update_with_answer: append last Q with host's yes/no/unknown
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.state = GuesserLLMState()

        self._schema_should = object_schema(
            "ShouldGuess",
            {
                "should_guess": boolean_schema("should_guess"),
                "confidence": number_schema("confidence", 0.0, 1.0),
            },
            ["should_guess", "confidence"],
        )
        self._schema_question = object_schema(
            "NextQuestion",
            {
                "thought": {"anyOf": [{"type": "null"}, string_schema("thought", 0, 400)]},
                "question_text": string_schema("question_text", 3, 160),
            },
            ["question_text"],
        )
        self._schema_guess = object_schema(
            "MakeGuess",
            {
                "thought": {"anyOf": [{"type": "null"}, string_schema("thought", 0, 400)]},
                "guess_text": string_schema("guess_text", 1, 80),
                "confidence": number_schema("confidence", 0.0, 1.0),
            },
            ["guess_text", "confidence"],
        )

    def _history_text(self) -> str:
        if not self.state.history:
            return "(no questions asked yet)"
        lines = []
        for qa in self.state.history[-12:]:  # cap context
            lines.append(f"Q: {qa.question}\nA: {qa.answer}")
        return "\n".join(lines)

    def should_guess(self, questions_used: int, max_questions: int) -> bool:
        remaining = max(0, max_questions - questions_used)
        system = (
            "You decide whether to GUESS now in a 20 Questions game.\n"
            "Return true only if you can name a single concrete topic with high confidence.\n"
            "Never return true for broad categories (e.g., 'ornamental tree').\n"
            "Prefer to ask more questions if your best candidate is a category or long descriptive phrase."
        )
        prev = ", ".join(self.state.attempted_guesses[-5:]) or "(none)"
        user = (
            "History (latest first):\n" + self._history_text() + "\n\n" +
            "Scratchpad so far:\n" + "\n".join(self.state.scratchpad[-12:]) + "\n\n" +
            "Decision rules:\n"
            "- Only GUESS when a specific proper/common noun stands out (e.g., 'apple', 'Eiffel Tower').\n"
            "- Do NOT guess categories (e.g., 'ornamental tree').\n"
            f"Turns used: {questions_used}; Remaining: {remaining}\n"
            f"Do not repeat previous guesses: {prev}"
        )
        payload, _meta = self.llm.structured_call(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=self._schema_should,
            temperature=0.0,
            max_output_tokens=40,
            max_repairs=2,
        )
        return bool(payload.get("should_guess", False))

    def next_question(self) -> Optional[Dict[str, object]]:
        system = (
            "You are the Guesser using a ReAct loop in a 20 Questions game.\n"
            "First write a brief Thought explaining the next discriminative test, then output the ASK action.\n"
            "Guidelines:\n"
            "- Aim for high-information binary splits (balanced). Prefer taxonomy/category membership first (animal/plant/mineral, food/not food, fruit/not fruit).\n"
            "- Avoid repeats and near-paraphrases.\n"
            "- Do NOT embed guesses in the question; no candidate names.\n"
            "- Keep the question short (<= 12 words) and end with '?'."
        )
        user = (
            "Ask the next yes/no question.\n" + self._history_text() + "\n\n" +
            "Scratchpad so far:\n" + "\n".join(self.state.scratchpad[-12:])
        )
        payload, _meta = self.llm.structured_call(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=self._schema_question,
            temperature=0.0,
            max_output_tokens=60,
            max_repairs=2,
        )
        thought = str(payload.get("thought") or "").strip()
        q = str(payload.get("question_text", "")).strip()
        if not q:
            return None
        if thought:
            self.state.scratchpad.append(f"Thought: {thought}")
        self.state.scratchpad.append(f"Action: ASK(\"{q}\")")
        self.state.last_question_text = q
        return {
            "type": "ask_question",
            "question_text": q,
            "attribute_id": None,
            "confidence": 1.0,
        }

    def make_guess(self) -> Optional[Dict[str, object]]:
        system = (
            "You are the Guesser using a ReAct loop.\n"
            "First write a brief Thought explaining elimination and candidate choice, then output the GUESS action.\n"
            "Return a short, canonical name only (<= 2 words), no sentences, no adjectives.\n"
            "If your best candidate is only a broad category, do NOT guess (unless forced by budget).\n"
            "Output JSON only with thought (optional), guess_text and confidence. Do not repeat prior guesses."
        )
        prev = ", ".join(self.state.attempted_guesses[-8:]) or "(none)"
        user = (
            "Make your best guess now based on the history.\n" + self._history_text() + "\n\n" +
            "Scratchpad so far:\n" + "\n".join(self.state.scratchpad[-12:]) + "\n" +
            f"Previously attempted guesses (do not repeat): {prev}"
        )
        payload, _meta = self.llm.structured_call(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=self._schema_guess,
            temperature=0.0,
            max_output_tokens=60,
            max_repairs=2,
        )
        thought = str(payload.get("thought") or "").strip()
        guess = str(payload.get("guess_text", "")).strip()
        if not guess:
            return None
        conf = float(payload.get("confidence", 0.0))
        # Track attempted guesses to avoid repetition later
        self.state.attempted_guesses.append(guess.lower())
        if thought:
            self.state.scratchpad.append(f"Thought: {thought}")
        self.state.scratchpad.append(f"Action: GUESS(\"{guess}\")")
        return {
            "type": "make_guess",
            "object_id": None,
            "object_name": guess,
            "confidence": conf,
        }

    def update_with_answer(self, host_payload: Dict[str, Any]) -> None:
        ans = str(host_payload.get("answer", "unknown")).lower()
        q = self.state.last_question_text or ""
        self.state.history.append(QA(question=q, answer=ans))
        self.state.scratchpad.append(f"Observation: {ans.upper()}")
        self.state.last_question_text = None


