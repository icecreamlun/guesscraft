from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..kb.kb import ObjectKB
from ..llm.client import LLMClient
from ..engine.validator import host_action_schema
from .attribute_parser import AttributeParser


@dataclass
class HostRuleBased:
    kb: ObjectKB
    attribute_parser: AttributeParser
    topic_object_id: str

    def answer(self, question_text: str) -> Dict[str, object]:
        classification = self.attribute_parser.classify(question_text, n=2, temperature=0.0)
        attr_id = classification.attribute_id
        if attr_id is None:
            return {
                "type": "answer_yes_no",
                "answer": "unknown",
                "justification": "unmapped_question",
                "consistency_score": 0.5,
            }
        attr = self.kb.attr_by_id.get(attr_id)
        if not attr or attr.kind != "boolean":
            return {
                "type": "answer_yes_no",
                "answer": "unknown",
                "justification": "non_boolean_attribute",
                "consistency_score": 0.5,
            }
        topic = self.kb.obj_by_id[self.topic_object_id]
        value = topic.attributes.get(attr_id)
        if isinstance(value, bool):
            return {
                "type": "answer_yes_no",
                "answer": "yes" if value else "no",
                "justification": f"{attr_id}={value}",
                "consistency_score": 1.0,
            }
        return {
            "type": "answer_yes_no",
            "answer": "unknown",
            "justification": "missing_value",
            "consistency_score": 0.5,
        }


@dataclass
class HostLLM:
    llm: LLMClient
    topic_name: str

    def answer(self, question_text: str) -> Dict[str, object]:
        # Heuristic: if the question appears to directly name a specialization of the topic
        # with the topic as the head noun (e.g., "small dog" when topic is "dog"), answer yes.
        # This avoids returning unknown for modifier+topic forms.
        def _normalize_tokens(text: str) -> list[str]:
            import re as _re
            return _re.findall(r"[a-z0-9]+", text.lower())

        q_tokens = _normalize_tokens(question_text)
        t_tokens = _normalize_tokens(self.topic_name)
        if t_tokens:
            topic_last = t_tokens[-1]
            if q_tokens:
                q_last = q_tokens[-1]
                # Allow simple plural tolerance on the question head (e.g., dogs -> dog)
                if q_last.endswith('s') and q_last[:-1] == topic_last:
                    q_last = topic_last
                if q_last == topic_last:
                    return {
                        "type": "answer_yes_no",
                        "answer": "yes",
                        "justification": None,
                        "consistency_score": 1.0,
                    }
        system = (
            "You are the Host in a 20 Questions game. "
            "You secretly know the topic and must answer questions strictly with yes/no/unknown. "
            "Return ONLY a compact JSON object matching the given schema. "
            "If the question is a direct guess of the topic (same meaning/name), implicitly treat it as a guess and answer 'yes'. "
            "Keep answers minimal: 'yes' or 'no' or 'unknown' only; do not restate the topic. "
            "Answer only about the topic ITSELF (strict IS-A semantics), not things it grows on/produces/relates to. "
            "When unsure, prefer 'unknown' rather than guessing. "
            "Examples (Topic='apple'):\n"
            "- 'Is it a fruit?' -> yes\n"
            "- 'Is it food?' -> yes\n"
            "- 'Is it a plant?' -> no\n"
            "- 'Is it a tree?' -> no\n"
            "- 'Is it a fruit tree?' -> no\n"
            "- 'Is it a type of fruit tree?' -> no\n"
            "- 'Is it a berry?' -> no\n"
            "- 'Is it a citrus?' -> no\n"
            "- 'Is it a pome?' -> yes"
        )
        user = f"Topic: {self.topic_name}. Question: {question_text.strip()}"
        schema = host_action_schema()
        payload, _meta = self.llm.structured_call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            schema=schema,
            temperature=0.0,
            max_output_tokens=64,
            max_repairs=2,
        )
        return payload


