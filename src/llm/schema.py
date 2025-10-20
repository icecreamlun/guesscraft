from __future__ import annotations

import json
from typing import Any, Dict, List

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def validate_json(instance: Any, schema: Dict[str, Any]) -> None:
	"""Validate an instance against a JSON Schema, raising on error."""
	validator = Draft202012Validator(schema)
	errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
	if errors:
		raise ValidationError("; ".join(str(e.message) for e in errors))


def enum_schema(title: str, values: List[str]) -> Dict[str, Any]:
	return {
		"title": title,
		"type": "string",
		"enum": values,
	}


def number_schema(title: str, minimum: float = 0.0, maximum: float = 1.0) -> Dict[str, Any]:
	return {
		"title": title,
		"type": "number",
		"minimum": minimum,
		"maximum": maximum,
	}


def boolean_schema(title: str) -> Dict[str, Any]:
	return {
		"title": title,
		"type": "boolean",
	}


def object_schema(title: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
	return {
		"title": title,
		"type": "object",
		"properties": properties,
		"required": required,
		"additionalProperties": False,
	}


def string_schema(title: str, min_length: int = 1, max_length: int | None = None) -> Dict[str, Any]:
	schema: Dict[str, Any] = {"title": title, "type": "string", "minLength": min_length}
	if max_length is not None:
		schema["maxLength"] = max_length
	return schema


def pretty_compact_schema(schema: Dict[str, Any]) -> str:
	"""Compact schema to a short JSON string for prompt inclusion."""
	return json.dumps(schema, separators=(",", ":"))

