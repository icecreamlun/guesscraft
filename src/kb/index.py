from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .kb import ObjectKB, ObjectItem


class AttributeIndex:
	"""Precompute partitions for each attribute value to support entropy and filtering."""

	def __init__(self, kb: ObjectKB):
		self.kb = kb
		self.index: Dict[str, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
		for obj in kb.objects:
			for attr_id, value in obj.attributes.items():
				self.index[attr_id][value].append(obj.id)

	def partition_counts(self, candidate_ids: List[str], attr_id: str) -> Dict[Any, int]:
		value_to_count: Dict[Any, int] = defaultdict(int)
		for value, obj_ids in self.index.get(attr_id, {}).items():
			# count overlap with candidates
			cand = set(candidate_ids)
			value_to_count[value] = sum(1 for oid in obj_ids if oid in cand)
		return dict(value_to_count)

	def yes_no_partition_counts(self, candidate_ids: List[str], attr_id: str, yes_value: Any = True, no_value: Any = False) -> Tuple[int, int, int]:
		counts = self.partition_counts(candidate_ids, attr_id)
		yes = counts.get(yes_value, 0)
		no = counts.get(no_value, 0)
		unknown = max(0, len(candidate_ids) - yes - no)
		return yes, no, unknown

	@staticmethod
	def entropy_from_counts(counts: List[int]) -> float:
		import math
		total = sum(counts)
		if total == 0:
			return 0.0
		entropy = 0.0
		for c in counts:
			if c <= 0:
				continue
			p = c / total
			entropy -= p * math.log2(p)
		return entropy

	def expected_entropy_yes_no(self, candidate_ids: List[str], attr_id: str) -> float:
		yes, no, unknown = self.yes_no_partition_counts(candidate_ids, attr_id)
		return self.entropy_from_counts([yes, no, unknown])


