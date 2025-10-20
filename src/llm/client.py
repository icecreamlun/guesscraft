from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .retry import retry_call
from .schema import validate_json, pretty_compact_schema


class LLMClientError(RuntimeError):
	pass


@dataclass
class LLMResponseMeta:
	model: str
	temperature: float
	usage_prompt_tokens: Optional[int]
	usage_completion_tokens: Optional[int]
	usage_total_tokens: Optional[int]
	attempts: int
	raw_text: str


class LLMClient:
	"""Thin wrapper for JSON-mode structured outputs with validation and auto-repair."""

	def __init__(
		self,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		model: Optional[str] = None,
		organization: Optional[str] = None,
		request_timeout_seconds: float = 30.0,
		max_retries: int = 2,
		seed: Optional[int] = None,
	):
		self.api_key = api_key or os.getenv("OPENAI_API_KEY")
		self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
		self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
		self.organization = organization or os.getenv("OPENAI_ORG")
		self.request_timeout_seconds = float(
			os.getenv("REQUEST_TIMEOUT_SECONDS", str(request_timeout_seconds))
		)
		self.max_retries = int(os.getenv("MAX_RETRIES", str(max_retries)))
		self.seed = seed

		try:
			from openai import OpenAI  # type: ignore
		except Exception as exc:
			raise LLMClientError(
				"openai package not installed. Add it to requirements and install."
			) from exc

		self._OpenAI = OpenAI
		self._client = self._OpenAI(
			api_key=self.api_key,
			base_url=self.base_url,
			organization=self.organization,
		)

	def _chat_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> Any:
		return self._client.chat.completions.create(
			model=self.model,
			messages=messages,
			temperature=temperature,
			max_tokens=max_tokens,
			response_format={"type": "json_object"},
			seed=self.seed,
			timeout=self.request_timeout_seconds,
		)

	def structured_call(
		self,
		messages: List[Dict[str, Any]],
		schema: Dict[str, Any],
		temperature: float = 0.0,
		max_output_tokens: int = 256,
		max_repairs: int = 2,
	) -> Tuple[Dict[str, Any], LLMResponseMeta]:
		"""Call model in JSON mode, validate against schema, auto-repair on failure.

		Returns (parsed_json, meta).
		"""

		attempts = 0
		last_text = ""

		# Ensure at least one message explicitly mentions 'json' in lowercase
		# to satisfy providers that require explicit user/system consent for JSON mode.
		def _contains_json_word(msgs: List[Dict[str, Any]]) -> bool:
			for m in msgs:
				c = (m.get("content") or "").lower()
				if "json" in c:
					return True
			return False

		messages_effective = list(messages)
		if not _contains_json_word(messages_effective):
			messages_effective.append({
				"role": "system",
				"content": "Respond in json only. Return a single minified json object strictly matching the provided schema.",
			})

		def _once() -> Any:
			return self._chat_completion(messages_effective, temperature, max_output_tokens)

		resp = retry_call(
			_once,
			max_retries=self.max_retries,
		)
		attempts += 1
		choice = resp.choices[0]
		last_text = (choice.message.content or "").strip()

		for repair_idx in range(max_repairs + 1):
			try:
				parsed = json.loads(last_text)
				validate_json(parsed, schema)
				meta = LLMResponseMeta(
					model=self.model,
					temperature=temperature,
					usage_prompt_tokens=getattr(resp, "usage", None).prompt_tokens if getattr(resp, "usage", None) else None,
					usage_completion_tokens=getattr(resp, "usage", None).completion_tokens if getattr(resp, "usage", None) else None,
					usage_total_tokens=getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else None,
					attempts=attempts,
					raw_text=last_text,
				)
				return parsed, meta
			except Exception as err:
				if repair_idx >= max_repairs:
					raise LLMClientError(f"Structured output failed after repairs: {err}") from err
				# Attempt repair: append a system nudge with error and schema
				repair_msg = {
					"role": "system",
					"content": (
						"Your last output could not be parsed/validated. "
						"Fix the issues below and return ONLY a compact json object that strictly matches the schema.\n"
						f"Error: {str(err)}\n"
						f"Schema: {pretty_compact_schema(schema)}"
					),
				}
				messages_repaired = [*messages_effective, repair_msg]
				resp = retry_call(
					lambda: self._chat_completion(messages_repaired, temperature=temperature, max_tokens=max_output_tokens),
					max_retries=self.max_retries,
				)
				attempts += 1
				choice = resp.choices[0]
				last_text = (choice.message.content or "").strip()

	def json_action_prompt(self, system_instructions: str, user_task: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Create a standard JSON-mode message set for structured outputs."""
		return [
			{"role": "system", "content": system_instructions.strip()},
			{
				"role": "user",
				"content": (
					"Return ONLY a minified JSON object with no prose.\n"
					"It must strictly satisfy this JSON Schema: "
					f"{pretty_compact_schema(schema)}\n"
					"Task: "
					f"{user_task.strip()}"
				),
			},
		]


