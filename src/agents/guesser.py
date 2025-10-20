from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..engine.types import YesNo
from ..kb.kb import ObjectKB
from ..kb.index import AttributeIndex
from .phrasing import GuesserPhraser


@dataclass
class GuesserEntropyState:
    candidate_ids: List[str]
    asked_attribute_ids: Set[str] = field(default_factory=set)
    constraints: Dict[str, bool] = field(default_factory=dict)


class GuesserEntropy:
    """Entropy-based question selector over a closed-world KB.

    - Maintains candidate set of object ids
    - Selects the boolean attribute that maximizes answer entropy (yes/no/unknown)
    - Updates candidate set from host answers (unknown leaves set unchanged)
    - Decides when to guess based on candidate set size or threshold
    """

    def __init__(
        self,
        kb: ObjectKB,
        index: AttributeIndex,
        tau_guess_threshold: float = 0.6,
        phraser: GuesserPhraser | None = None,
    ):
        self.kb = kb
        self.index = index
        self.state = GuesserEntropyState(candidate_ids=[o.id for o in kb.objects])
        self._tau = tau_guess_threshold
        self._phraser = phraser

    def _boolean_attribute_ids(self) -> List[str]:
        return [a.id for a in self.kb.attributes if a.kind == "boolean"]

    def select_next_attribute(self) -> Optional[str]:
        candidates = [
            a_id
            for a_id in self._boolean_attribute_ids()
            if a_id not in self.state.asked_attribute_ids
        ]
        if not candidates:
            return None
        # Pick attribute with maximum answer entropy on current candidate set
        best_attr = None
        best_score = -1.0
        for a_id in candidates:
            score = self.index.expected_entropy_yes_no(self.state.candidate_ids, a_id)
            if score > best_score:
                best_score = score
                best_attr = a_id
        return best_attr

    def next_question(self) -> Optional[Dict[str, object]]:
        attr_id = self.select_next_attribute()
        if attr_id is None:
            return None
        self.state.asked_attribute_ids.add(attr_id)
        attr = self.kb.attr_by_id[attr_id]
        asked = []
        # Build a short history of the last questions
        # (Expect the outer engine to maintain a log; for now we pass empty if unknown.)
        if self._phraser is not None:
            question_text = self._phraser.phrase(attr.name, asked)
        else:
            question_text = f"Is it {attr.name.lower()}?"
        return {
            "type": "ask_question",
            "question_text": question_text,
            "attribute_id": attr_id,
            "confidence": 1.0,
        }

    def update_with_answer(self, attribute_id: str, answer: YesNo) -> None:
        # Only boolean attributes can be filtered with yes/no directly
        attr = self.kb.attr_by_id.get(attribute_id)
        if not attr or attr.kind != "boolean":
            return
        if answer == "unknown":
            return
        expected = True if answer == "yes" else False
        self.state.constraints[attribute_id] = expected
        # Filter candidates
        new_candidates: List[str] = []
        for oid in self.state.candidate_ids:
            obj = self.kb.obj_by_id[oid]
            if obj.attributes.get(attribute_id) == expected:
                new_candidates.append(oid)
        self.state.candidate_ids = new_candidates

    def should_guess(self, questions_used: int, max_questions: int) -> bool:
        if len(self.state.candidate_ids) <= 1:
            return True
        # Uniform posterior over candidates
        p_map = 1.0 / max(1, len(self.state.candidate_ids))
        if p_map >= self._tau:
            return True
        # Near budget end, be decisive
        if max_questions - questions_used <= 2 and len(self.state.candidate_ids) <= 3:
            return True
        return False

    def make_guess(self) -> Optional[Dict[str, object]]:
        if not self.state.candidate_ids:
            return None
        oid = self.state.candidate_ids[0]
        obj = self.kb.obj_by_id[oid]
        conf = 1.0 / len(self.state.candidate_ids)
        return {
            "type": "make_guess",
            "object_id": obj.id,
            "object_name": obj.name,
            "confidence": conf,
        }


