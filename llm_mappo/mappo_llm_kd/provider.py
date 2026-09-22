"""Secret-safe OpenAI-compatible Chat Completions provider boundary."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import re
from time import perf_counter
from typing import Callable, Mapping
from urllib.error import HTTPError
from urllib import request
from urllib.parse import urlsplit

from .prompt import LabelPrompt


Transport = Callable[[str, bytes, dict[str, str], float], Mapping[str, object]]
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_SENSITIVE_KEYS = {
    "apikey",
    "xapikey",
    "authorization",
    "secret",
    "password",
    "credential",
    "accesstoken",
    "bearertoken",
}


class ProviderCallError(RuntimeError):
    """Redacted provider failure safe for receipts and logs."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    requested_model: str
    base_url: str
    endpoint_path: str
    api_key_environment_variable: str
    temperature: float
    seed: int | None
    timeout_seconds: float
    retry_limit: int
    max_completion_tokens: int
    max_prompt_characters: int
    response_format: str
    extra_body: dict[str, object]

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_generation_settings()

    def _validate_identity(self) -> None:
        strings = (
            self.provider_id,
            self.requested_model,
            self.base_url,
            self.endpoint_path,
            self.api_key_environment_variable,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("Provider identity fields must be non-empty strings.")
        if not self.base_url.startswith("https://"):
            raise ValueError("Provider base_url must use HTTPS.")
        parts = urlsplit(self.base_url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Provider base_url must contain no embedded credential.")
        if not self.endpoint_path.startswith("/"):
            raise ValueError("Provider endpoint_path must start with '/'.")
        if _ENVIRONMENT_NAME.fullmatch(self.api_key_environment_variable) is None:
            raise ValueError("API key setting must be an environment variable name.")

    def _validate_generation_settings(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Provider temperature is outside [0, 2].")
        if self.timeout_seconds <= 0 or self.retry_limit < 0:
            raise ValueError("Provider timeout/retry values are invalid.")
        if self.max_completion_tokens < 1 or self.max_prompt_characters < 1:
            raise ValueError("Provider token and prompt caps must be positive.")
        if self.response_format != "json_object":
            raise ValueError("Phase 3 requires the json_object response format.")
        if contains_credential(self.extra_body):
            raise ValueError("extra_body must contain no embedded credential.")

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + self.endpoint_path

    def public_identity(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "requested_model": self.requested_model,
            "base_url_sha256": sha256(self.base_url.encode("utf-8")).hexdigest(),
            "endpoint_path": self.endpoint_path,
            "api_key_environment_variable": self.api_key_environment_variable,
            "temperature": self.temperature,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "retry_limit": self.retry_limit,
            "max_completion_tokens": self.max_completion_tokens,
            "max_prompt_characters": self.max_prompt_characters,
            "response_format": self.response_format,
            "extra_body": self.extra_body,
        }


@dataclass(frozen=True)
class ProviderCompletion:
    content: str
    receipt: dict[str, object]


class OpenAICompatibleProvider:
    """Make one redacted provider attempt; retry policy belongs to labeling."""

    def __init__(self, config: ProviderConfig, *, transport: Transport | None = None):
        self.config = config
        self._transport = transport or _urlopen_transport

    def complete(self, prompt: LabelPrompt) -> ProviderCompletion:
        secret = os.environ.get(self.config.api_key_environment_variable)
        if not secret or not secret.strip():
            raise RuntimeError(
                f"{self.config.api_key_environment_variable} is required in the "
                "process environment."
            )
        if len(prompt.user_text) > self.config.max_prompt_characters:
            raise ProviderCallError("prompt_size_limit_exceeded")
        body = self._request_body(prompt)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
        }
        started = perf_counter()
        try:
            decoded = self._transport(
                self.config.endpoint,
                json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers,
                self.config.timeout_seconds,
            )
            content, returned_model, response_id, usage = _decode_completion(decoded)
        except Exception as error:
            code = (
                f"provider_http_{error.code}"
                if isinstance(error, HTTPError)
                else "provider_request_failed"
            )
            raise ProviderCallError(code) from None
        latency = perf_counter() - started
        response_hash = sha256(
            json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        receipt = {
            **self.config.public_identity(),
            "returned_model": returned_model,
            "response_id": response_id,
            "prompt_version": prompt.prompt_version,
            "semantic_view_schema": "temporal-semantic-view-v1",
            "system_prompt_sha256": prompt.system_sha256,
            "user_prompt_sha256": prompt.user_sha256,
            "semantic_view_sha256": prompt.semantic_view_sha256,
            "usage": usage,
            "latency_seconds": latency,
            "response_sha256": response_hash,
        }
        return ProviderCompletion(content=content, receipt=receipt)

    def _request_body(self, prompt: LabelPrompt) -> dict[str, object]:
        protected = {
            "model",
            "messages",
            "stream",
            "temperature",
            "max_tokens",
            "response_format",
            "seed",
        }
        if protected.intersection(self.config.extra_body):
            raise ValueError("extra_body cannot override protected request fields.")
        body = {
            "model": self.config.requested_model,
            "messages": [
                {"role": "system", "content": prompt.system_text},
                {"role": "user", "content": prompt.user_text},
            ],
            "stream": False,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_completion_tokens,
            "response_format": {"type": self.config.response_format},
            **self.config.extra_body,
        }
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        return body


def _decode_completion(
    decoded: Mapping[str, object],
) -> tuple[str, str, str | None, dict[str, int]]:
    try:
        choices = decoded["choices"]
        first = choices[0]
        content = first["message"]["content"]
        returned_model = decoded["model"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Provider returned an invalid chat completion.") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Provider returned empty completion content.")
    if not isinstance(returned_model, str) or not returned_model:
        raise ValueError("Provider did not return a model identity.")
    response_id = decoded.get("id")
    if response_id is not None and not isinstance(response_id, str):
        response_id = None
    raw_usage = decoded.get("usage", {})
    usage = {}
    if isinstance(raw_usage, Mapping):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw_usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = value
    return content, returned_model, response_id, usage


def _urlopen_transport(
    endpoint: str, body: bytes, headers: dict[str, str], timeout_seconds: float
) -> Mapping[str, object]:
    query = request.Request(endpoint, data=body, headers=headers, method="POST")
    with request.urlopen(query, timeout=timeout_seconds) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("Provider response must be a JSON object.")
    return decoded


def contains_credential(value: object) -> bool:
    """Return whether an arbitrary public payload embeds credential material."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in _SENSITIVE_KEYS or contains_credential(child):
                return True
        return False
    if isinstance(value, list):
        return any(contains_credential(item) for item in value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered.startswith("sk-") or lowered.startswith("bearer ")
    return False
