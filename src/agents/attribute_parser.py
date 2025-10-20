from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..kb.kb import ObjectKB
from ..llm.client import LLMClient
from ..llm.schema import object_schema, enum_schema, string_schema, pretty_compact_schema


@dataclass
class ClassificationResult:
	attribute_id: Optional[str]
	confidence: float
	justification: Optional[str]


class AttributeParser:
	"""Maps free-form yes/no question text to a known attribute label.

	Uses an LLM constrained to a small schema returning an `attribute_id` from the
	provided label set (or null if none applies), plus a confidence score.
	"""

	def __init__(self, kb: ObjectKB, llm: LLMClient):
		self.kb = kb
		self.llm = llm
		self._attribute_ids = self.kb.attribute_ids()
		self._schema = object_schema(
			"AttributeClassification",
			{
				"attribute_id": {"anyOf": [{"type": "null"}, {"type": "string", "enum": self._attribute_ids}]},
				"confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
				"justification": {"anyOf": [{"type": "null"}, string_schema("justification", 0, 300)]},
			},
			["confidence", "attribute_id"],
		)

	def classify(self, question_text: str, n: int = 2, temperature: float = 0.0) -> ClassificationResult:
		messages = [
			{
				"role": "system",
				"content": (
					"You map user yes/no questions in a 20-questions game to a known attribute label. "
					"Return ONLY JSON with fields: attribute_id (or null), confidence [0,1], justification."
				),
			},
			{
				"role": "user",
				"content": (
					"Known attribute ids: " + ", ".join(self._attribute_ids) + "\n"
					"Question: " + question_text.strip()
				),
			},
		]

		votes: List[ClassificationResult] = []
		for _ in range(max(1, n)):
			parsed, _meta = self.llm.structured_call(messages, self._schema, temperature=temperature, max_output_tokens=200, max_repairs=2)
			votes.append(
				ClassificationResult(
					attribute_id=parsed.get("attribute_id"),
					confidence=float(parsed.get("confidence", 0.0)),
					justification=parsed.get("justification"),
				)
			)

		# majority vote by attribute_id; tie-break by average confidence
		from collections import defaultdict
		count: Dict[Optional[str], int] = defaultdict(int)
		conf: Dict[Optional[str], float] = defaultdict(float)
		for v in votes:
			count[v.attribute_id] += 1
			conf[v.attribute_id] += v.confidence
		best_attr = None
		best = (-1, -1.0)  # (count, avg_conf)
		for attr_id, c in count.items():
			avg = conf[attr_id] / max(1, c)
			key = (c, avg)
			if key > best:
				best = key
				best_attr = attr_id
		return ClassificationResult(attribute_id=best_attr, confidence=conf[best_attr] / max(1, count[best_attr]), justification=None)


