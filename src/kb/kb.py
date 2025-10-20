from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Attribute:
	id: str
	name: str
	description: str
	# kind: boolean or enum (string options)
	kind: str
	options: Optional[List[str]] = None


@dataclass
class ObjectItem:
	id: str
	name: str
	# attributes: mapping from attribute.id to value (bool or enum string)
	attributes: Dict[str, Any]


class ObjectKB:
	"""In-memory KB with attributes and objects."""

	def __init__(self, attributes: List[Attribute], objects: List[ObjectItem]):
		self.attributes = attributes
		self.objects = objects
		self.attr_by_id = {a.id: a for a in attributes}
		self.obj_by_id = {o.id: o for o in objects}

	@classmethod
	def from_seed(cls) -> "ObjectKB":
		# Minimal seed: a few boolean attributes and objects for early testing
		attributes = [
			Attribute(id="is_animal", name="Is an animal", description="The object is a living animal", kind="boolean"),
			Attribute(id="is_fruit", name="Is a fruit", description="The object is a fruit", kind="boolean"),
			Attribute(id="is_electronic", name="Is electronic", description="The object is an electronic device", kind="boolean"),
			Attribute(id="size", name="Size", description="Coarse size bucket", kind="enum", options=["small", "medium", "large"]),
			Attribute(id="habitat", name="Habitat", description="Where it typically exists", kind="enum", options=["land", "air", "water"]),
		]
		objects = [
			ObjectItem(
				id="apple",
				name="Apple",
				attributes={
					"is_animal": False,
					"is_fruit": True,
					"is_electronic": False,
					"size": "small",
					"habitat": "land",
				},
			),
			ObjectItem(
				id="eagle",
				name="Eagle",
				attributes={
					"is_animal": True,
					"is_fruit": False,
					"is_electronic": False,
					"size": "medium",
					"habitat": "air",
				},
			),
			ObjectItem(
				id="laptop",
				name="Laptop",
				attributes={
					"is_animal": False,
					"is_fruit": False,
					"is_electronic": True,
					"size": "medium",
					"habitat": "land",
				},
			),
			ObjectItem(
				id="whale",
				name="Whale",
				attributes={
					"is_animal": True,
					"is_fruit": False,
					"is_electronic": False,
					"size": "large",
					"habitat": "water",
				},
			),
		]
		return cls(attributes=attributes, objects=objects)

	def attribute_ids(self) -> List[str]:
		return [a.id for a in self.attributes]

	def filter_objects(self, constraints: Dict[str, Any]) -> List[ObjectItem]:
		"""Filter objects by attribute constraints (exact match)."""
		result: List[ObjectItem] = []
		for obj in self.objects:
			ok = True
			for attr_id, expected in constraints.items():
				if obj.attributes.get(attr_id) != expected:
					ok = False
					break
			if ok:
				result.append(obj)
		return result


