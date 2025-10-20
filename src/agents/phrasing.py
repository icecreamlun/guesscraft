from __future__ import annotations

from typing import Dict, List

from ..llm.client import LLMClient
from ..llm.schema import object_schema, string_schema


class GuesserPhraser:
    """LLM phrasing shim: turn an attribute into a crisp yes/no question."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._schema = object_schema(
            "PhrasedQuestion",
            {
                "question_text": string_schema("question_text", 3, 160),
            },
            ["question_text"],
        )

    def phrase(self, attribute_name: str, asked_questions: List[str]) -> str:
        system = (
            "You write a single, concise yes/no question for a 20 Questions game. "
            "Avoid repeating past questions; no extra words; no punctuation beyond '?' at the end."
        )
        user = (
            "Attribute to ask about: "
            + attribute_name
            + "\nPreviously asked questions: "
            + " | ".join(asked_questions[-5:])
        )
        payload, _meta = self.llm.structured_call(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=self._schema,
            temperature=0.0,
            max_output_tokens=60,
            max_repairs=2,
        )
        return str(payload.get("question_text", "")).strip()


