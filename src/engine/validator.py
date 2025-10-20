from __future__ import annotations

from typing import Any, Dict, List

from jsonschema.exceptions import ValidationError

from ..llm.schema import (
	boolean_schema,
	enum_schema,
	object_schema,
	number_schema,
	string_schema,
	validate_json,
)


def guesser_action_schema(attribute_ids: List[str]) -> Dict[str, Any]:
	return {
		"oneOf": [
			object_schema(
				"GuesserAskQuestion",
				{
					"type": enum_schema("type", ["ask_question"]),
					"question_text": string_schema("question_text", min_length=3, max_length=256),
					"attribute_id": {
						"anyOf": [
							{"type": "null"},
							{"type": "string", "enum": attribute_ids},
						]
					},
					"confidence": number_schema("confidence", 0.0, 1.0),
				},
				["type", "question_text", "confidence"],
			),
			object_schema(
				"GuesserMakeGuess",
				{
					"type": enum_schema("type", ["make_guess"]),
					"object_id": {"anyOf": [{"type": "null"}, string_schema("object_id", 1, 128)]},
					"object_name": {"anyOf": [{"type": "null"}, string_schema("object_name", 1, 128)]},
					"confidence": number_schema("confidence", 0.0, 1.0),
				},
				["type", "confidence"],
			),
		],
		"unevaluatedProperties": False,
	}


def host_action_schema() -> Dict[str, Any]:
	return object_schema(
		"HostAnswerYesNo",
		{
			"type": enum_schema("type", ["answer_yes_no"]),
			"answer": enum_schema("answer", ["yes", "no", "unknown"]),
			"justification": {"anyOf": [{"type": "null"}, string_schema("justification", 0, 400)]},
			"consistency_score": number_schema("consistency_score", 0.0, 1.0),
		},
		["type", "answer", "consistency_score"],
	)


def validate_guesser_action(payload: Dict[str, Any], attribute_ids: List[str]) -> None:
	validate_json(payload, guesser_action_schema(attribute_ids))


def validate_host_action(payload: Dict[str, Any]) -> None:
	validate_json(payload, host_action_schema())


