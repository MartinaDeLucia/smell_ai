from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from time import monotonic
from typing import Any, Callable, Optional

from llm_detection.types import LLMGenerationResult

def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping-like Ollama object or a typed response."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _serialize_provider_response(
    value: Any,
) -> str:
    """
    Best-effort serialization of the complete provider response.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (bytes, bytearray)):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    # Typed provider objects may expose model_dump() or dict().
    for method_name in (
        "model_dump",
        "dict",
    ):
        method = getattr(
            value,
            method_name,
            None,
        )

        if callable(method):
            try:
                value = method()
                break
            except Exception:
                pass

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)

class LLMProvider(ABC):
    """Bridge implementor for text generation.

    ``generate`` is intentionally kept for backward compatibility.
    ``generate_with_metadata`` can additionally expose provider-native reasoning
    (when available) without forcing every provider to support it.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMGenerationResult:
        return LLMGenerationResult(response=self.generate(prompt))


@dataclass
class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests."""

    fixed_response: Optional[str] = None
    response_factory: Optional[Callable[[str], str]] = None
    native_reasoning: str = ""

    def generate(self, prompt: str) -> str:
        if self.response_factory is not None:
            return self.response_factory(prompt)
        if self.fixed_response is not None:
            return self.fixed_response
        raise ValueError(
            "MockLLMProvider requires fixed_response or response_factory"
        )

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMGenerationResult:
        return LLMGenerationResult(
            response=self.generate(prompt),
            native_reasoning=self.native_reasoning,
        )


class LocalLLMProvider(LLMProvider):
    """
    Ollama-backed local provider.

    Thinking is provider/model specific and is kept separate from the
    user-facing CodeSmile reasoning.
    """

    def __init__(
        self,
        model_name: str,
        host: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        response_format: Optional[Any] = None,
        think: Optional[bool | str] = None,
    ):
        self.model_name = model_name
        self.host = host
        self.options = options
        self.response_format = response_format
        self.think = think

    def _client(self):
        try:
            import ollama
        except Exception as e:
            raise RuntimeError(
                "ollama is not available; "
                "cannot use LocalLLMProvider"
            ) from e

        return (
            ollama.Client(host=self.host)
            if self.host
            else ollama
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self.generate_with_metadata(
            prompt
        ).response

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        response_format: Optional[
            dict[str, Any]
        ] = None,
    ) -> LLMGenerationResult:

        client = self._client()

        messages: list[
            dict[str, str]
        ] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": self.options,
            "stream": False,
        }

        if response_format is not None:
            kwargs["format"] = response_format

        elif self.response_format is not None:
            kwargs["format"] = (
                self.response_format
            )

        if self.think is not None:
            kwargs["think"] = self.think

        try:
            response = client.chat(
                **kwargs
            )

        except Exception as e:
            host_hint = (
                f" ({self.host})"
                if self.host
                else ""
            )

            raise RuntimeError(
                "Failed to chat with Ollama"
                + host_hint
                + ". Ensure Ollama is installed/running "
                "and the model is available locally."
            ) from e

        message = _value(
            response,
            "message",
            {},
        )

        final_response = str(
            _value(
                message,
                "content",
                "",
            )
            or ""
        )

        native_reasoning = str(
            _value(
                message,
                "thinking",
                "",
            )
            or ""
        )

        metadata_keys = (
            "model",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_cached_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )

        metadata = {
            key: _value(
                response,
                key,
            )
            for key in metadata_keys
            if _value(
                response,
                key,
            )
            is not None
        }

        metadata["provider"] = "ollama"

        prompt_tokens = _value(
            response,
            "prompt_eval_count",
        )

        output_tokens = _value(
            response,
            "eval_count",
        )

        if (
            prompt_tokens is not None
            or output_tokens is not None
        ):
            usage: dict[str, Any] = {}

            if prompt_tokens is not None:
                usage["input_tokens"] = (
                    prompt_tokens
                )

            if output_tokens is not None:
                usage["output_tokens"] = (
                    output_tokens
                )

            if (
                isinstance(
                    prompt_tokens,
                    int,
                )
                and isinstance(
                    output_tokens,
                    int,
                )
            ):
                usage["total_tokens"] = (
                    prompt_tokens
                    + output_tokens
                )

            metadata["usage"] = usage

        if (
            not final_response.strip()
            and native_reasoning.strip()
        ):
            metadata[
                "empty_content_with_thinking"
            ] = True

        return LLMGenerationResult(
            response=final_response,
            native_reasoning=native_reasoning,
            metadata=metadata,

            # This lets us retain the complete Ollama response too,
            # maintaining the same trace model as remote APIs.
            raw_provider_response=(
                _serialize_provider_response(
                    response
                )
            ),
        )

class ApiLLMProvider(LLMProvider):
    """
    HTTP-backed provider for remote LLM APIs.

    Supported protocols:

    generic_http
        Backward compatibility with the old CodeSmile stub:
        POST <base_url>/generate

    openai_compatible
        Chat-completions compatible API:
        POST <base_url>/chat/completions

    The API key is injected at runtime and never persisted here.
    """

    GENERIC_HTTP = "generic_http"
    OPENAI_COMPATIBLE = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
        *,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        protocol: str = GENERIC_HTTP,
        options: Optional[
            dict[str, Any]
        ] = None,
        supports_structured_output: bool = False,
    ):
        base_url = (
            base_url
            or ""
        ).strip().rstrip("/")

        if not base_url:
            raise ValueError(
                "base_url must not be empty"
            )

        self.base_url = base_url
        self.timeout_s = float(
            timeout_s
        )

        self.model_name = (
            (model_name or "").strip()
            or None
        )

        self.api_key = (
            (api_key or "").strip()
            or None
        )

        self.protocol = (
            protocol
            or self.GENERIC_HTTP
        ).strip()

        self.options = dict(
            options or {}
        )

        self.supports_structured_output = bool(
            supports_structured_output
        )

        # Exposed for GenerationTrace.
        self.response_format = (
            "json_schema"
            if self.supports_structured_output
            else None
        )

        self.think = None

        if self.protocol not in {
            self.GENERIC_HTTP,
            self.OPENAI_COMPATIBLE,
        }:
            raise ValueError(
                "Unsupported API protocol: "
                f"{self.protocol}"
            )

        if (
            self.protocol
            == self.OPENAI_COMPATIBLE
            and not self.model_name
        ):
            raise ValueError(
                "model_name is required for "
                "openai_compatible API providers"
            )

    def _httpx(self):
        try:
            import httpx
        except Exception as exc:
            raise RuntimeError(
                "httpx is not available; "
                "cannot use ApiLLMProvider"
            ) from exc

        return httpx

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self.generate_with_metadata(
            prompt
        ).response

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        response_format: Optional[
            dict[str, Any]
        ] = None,
    ) -> LLMGenerationResult:

        if (
            self.protocol
            == self.OPENAI_COMPATIBLE
        ):
            return (
                self._generate_openai_compatible(
                    prompt,
                    system_prompt=system_prompt,
                    response_format=response_format,
                )
            )

        return self._generate_generic(
            prompt
        )

    # ---------------------------------------------------------
    # Legacy generic API
    # ---------------------------------------------------------

    def _generate_generic(
        self,
        prompt: str,
    ) -> LLMGenerationResult:

        httpx = self._httpx()

        url = (
            f"{self.base_url}/generate"
        )

        started = monotonic()

        with httpx.Client(
            timeout=self.timeout_s
        ) as client:

            resp = client.post(
                url,
                json={
                    "prompt": prompt
                },
            )

            resp.raise_for_status()

            elapsed_ms = round(
                (
                    monotonic()
                    - started
                )
                * 1000,
                3,
            )

            raw_provider_response = (
                resp.text
            )

            is_json = (
                "application/json"
                in resp.headers.get(
                    "content-type",
                    "",
                )
            )

            data = (
                resp.json()
                if is_json
                else None
            )

        if isinstance(
            data,
            dict,
        ):

            if (
                "response" not in data
                and "content" not in data
            ):
                return LLMGenerationResult(
                    response=resp.text,
                    metadata={
                        "provider":
                            self.GENERIC_HTTP,
                        "http_status":
                            getattr(
                                resp,
                                "status_code",
                                None,
                            ),
                        "latency_ms":
                            elapsed_ms,
                    },
                    raw_provider_response=(
                        raw_provider_response
                    ),
                )

            response_text = str(
                data.get(
                    "response",
                    data.get(
                        "content",
                        "",
                    ),
                )
                or ""
            )

            native_reasoning = str(
                data.get(
                    "reasoning",
                    data.get(
                        "thinking",
                        "",
                    ),
                )
                or ""
            )

            metadata = {
                key: value
                for key, value
                in data.items()
                if key not in {
                    "response",
                    "content",
                    "reasoning",
                    "thinking",
                }
            }

            metadata.update(
                {
                    "provider":
                        self.GENERIC_HTTP,
                    "http_status":
                        getattr(
                            resp,
                            "status_code",
                            None,
                        ),
                    "latency_ms":
                        elapsed_ms,
                }
            )

            return LLMGenerationResult(
                response=response_text,
                native_reasoning=(
                    native_reasoning
                ),
                metadata=metadata,
                raw_provider_response=(
                    raw_provider_response
                ),
            )

        return LLMGenerationResult(
            response=resp.text,
            metadata={
                "provider":
                    self.GENERIC_HTTP,
                "http_status":
                    getattr(
                        resp,
                        "status_code",
                        None,
                    ),
                "latency_ms":
                    elapsed_ms,
            },
            raw_provider_response=(
                raw_provider_response
            ),
        )

    # ---------------------------------------------------------
    # OpenAI-compatible chat API
    # ---------------------------------------------------------

    def _generate_openai_compatible(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str],
        response_format: Optional[
            dict[str, Any]
        ],
    ) -> LLMGenerationResult:

        httpx = self._httpx()

        url = (
            f"{self.base_url}"
            "/chat/completions"
        )

        messages: list[
            dict[str, str]
        ] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content":
                        system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload: dict[
            str,
            Any,
        ] = {
            "model":
                self.model_name,
            "messages":
                messages,
        }

        payload.update(
            self.options
        )

        # Structured output remains optional because not every
        # OpenAI-compatible endpoint implements JSON Schema.
        if (
            response_format is not None
            and self.supports_structured_output
        ):
            payload[
                "response_format"
            ] = {
                "type":
                    "json_schema",
                "json_schema": {
                    "name":
                        "codesmile_detection",
                    "strict":
                        True,
                    "schema":
                        response_format,
                },
            }

        headers = {
            "Content-Type":
                "application/json"
        }

        if self.api_key:
            headers[
                "Authorization"
            ] = (
                f"Bearer "
                f"{self.api_key}"
            )

        started = monotonic()

        try:
            with httpx.Client(
                timeout=self.timeout_s
            ) as client:

                resp = client.post(
                    url,
                    json=payload,
                    headers=headers,
                )

                resp.raise_for_status()

                elapsed_ms = round(
                    (
                        monotonic()
                        - started
                    )
                    * 1000,
                    3,
                )

                # Keep the COMPLETE body returned by the API.
                raw_provider_response = (
                    resp.text
                )

                data = resp.json()

        except Exception as exc:
            raise RuntimeError(
                "Failed to call API provider "
                f"at {self.base_url}. "
                "Check the base URL, model name, "
                "API key and provider availability."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "API response is not "
                "a JSON object"
            )

        choices = data.get(
            "choices"
        )

        if (
            not isinstance(
                choices,
                list,
            )
            or not choices
        ):
            raise RuntimeError(
                "API response does not "
                "contain choices[0]"
            )

        choice = choices[0]

        if not isinstance(
            choice,
            dict,
        ):
            raise RuntimeError(
                "API response choices[0] "
                "is invalid"
            )

        message = choice.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):
            raise RuntimeError(
                "API response does not "
                "contain a valid message"
            )

        final_response = (
            self._content_to_text(
                message.get(
                    "content"
                )
            )
        )

        # Some providers expose reasoning under different
        # names. Keep it diagnostic-only.
        native_reasoning = str(
            message.get(
                "reasoning",
                message.get(
                    "reasoning_content",
                    message.get(
                        "thinking",
                        "",
                    ),
                ),
            )
            or ""
        )

        metadata: dict[
            str,
            Any,
        ] = {
            "provider":
                self.OPENAI_COMPATIBLE,
            "http_status":
                getattr(
                    resp,
                    "status_code",
                    None,
                ),
            "latency_ms":
                elapsed_ms,
        }

        for key in (
            "id",
            "model",
            "created",
            "system_fingerprint",
        ):
            if data.get(
                key
            ) is not None:
                metadata[key] = (
                    data.get(key)
                )

        if (
            choice.get(
                "finish_reason"
            )
            is not None
        ):
            metadata[
                "finish_reason"
            ] = choice.get(
                "finish_reason"
            )

        # Normalize token information while preserving the
        # provider's original usage object too.
        usage = data.get(
            "usage"
        )

        if isinstance(
            usage,
            dict,
        ):

            input_tokens = usage.get(
                "prompt_tokens",
                usage.get(
                    "input_tokens"
                ),
            )

            output_tokens = usage.get(
                "completion_tokens",
                usage.get(
                    "output_tokens"
                ),
            )

            total_tokens = usage.get(
                "total_tokens"
            )

            if (
                total_tokens is None
                and isinstance(
                    input_tokens,
                    int,
                )
                and isinstance(
                    output_tokens,
                    int,
                )
            ):
                total_tokens = (
                    input_tokens
                    + output_tokens
                )

            normalized_usage: dict[
                str,
                Any,
            ] = {
                "provider_usage":
                    dict(usage)
            }

            if input_tokens is not None:
                normalized_usage[
                    "input_tokens"
                ] = input_tokens

            if output_tokens is not None:
                normalized_usage[
                    "output_tokens"
                ] = output_tokens

            if total_tokens is not None:
                normalized_usage[
                    "total_tokens"
                ] = total_tokens

            metadata[
                "usage"
            ] = normalized_usage

        return LLMGenerationResult(
            # The actual assistant answer.
            response=final_response,

            # Provider-specific optional reasoning.
            native_reasoning=native_reasoning,

            # Usage / timing / identifiers.
            metadata=metadata,

            # COMPLETE HTTP response body.
            raw_provider_response=(
                raw_provider_response
            ),
        )

    @staticmethod
    def _content_to_text(
        content: Any,
    ) -> str:
        """
        Convert common chat-completions content formats
        into plain text.
        """

        if content is None:
            return ""

        if isinstance(
            content,
            str,
        ):
            return content

        if isinstance(
            content,
            list,
        ):
            parts: list[str] = []

            for item in content:
                if isinstance(
                    item,
                    str,
                ):
                    parts.append(
                        item
                    )

                elif isinstance(
                    item,
                    dict,
                ):
                    text = (
                        item.get("text")
                        or item.get(
                            "content"
                        )
                    )

                    if text is not None:
                        parts.append(
                            str(text)
                        )

            return "".join(
                parts
            )

        return str(
            content
        )

    def test_connection(
        self,
    ) -> LLMGenerationResult:
        """
        Perform a minimal real generation.

        NOTE:
        for a paid API this may consume a very small number
        of billable tokens.
        """

        if (
            self.protocol
            == self.GENERIC_HTTP
        ):
            return (
                self.generate_with_metadata(
                    "Reply with OK."
                )
            )

        original_options = dict(
            self.options
        )

        test_options = dict(
            self.options
        )

        test_options.setdefault(
            "temperature",
            0,
        )

        if (
            "max_tokens"
            not in test_options
            and
            "max_completion_tokens"
            not in test_options
        ):
            test_options[
                "max_tokens"
            ] = 8

        self.options = (
            test_options
        )

        try:
            return self.generate_with_metadata(
                "Reply with exactly OK.",
                system_prompt=(
                    "You are testing an "
                    "API connection."
                ),
                response_format=None,
            )

        finally:
            self.options = (
                original_options
            )

class OllamaModelManager:
    """Small service used by the GUI to manage local Ollama models."""

    def __init__(self, host: Optional[str] = None):
        self.host = host

    def _client(self):
        try:
            import ollama
        except Exception as e:
            raise RuntimeError(
                "ollama is not available. "
                "Install the Python package and Ollama first."
            ) from e

        return ollama.Client(host=self.host) if self.host else ollama

    def list_installed_models(self) -> list[dict[str, Any]]:
        response = self._client().list()
        models = _value(response, "models", []) or []

        out: list[dict[str, Any]] = []

        for model in models:
            details = _value(model, "details")

            out.append(
                {
                    "model": str(
                        _value(
                            model,
                            "model",
                            _value(model, "name", ""),
                        )
                        or ""
                    ),
                    "size": _value(model, "size"),
                    "parameter_size": _value(
                        details,
                        "parameter_size",
                    ),
                    "quantization_level": _value(
                        details,
                        "quantization_level",
                    ),
                    "family": _value(details, "family"),
                }
            )

        return out

    def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[
            Callable[[str, Optional[int], Optional[int]], None]
        ] = None,
    ) -> None:
        model_name = (model_name or "").strip()

        if not model_name:
            raise ValueError("Model name must not be empty")

        stream = self._client().pull(
            model_name,
            stream=True,
        )

        for progress in stream:
            status = str(
                _value(progress, "status", "") or ""
            )
            completed = _value(progress, "completed")
            total = _value(progress, "total")

            if progress_callback:
                progress_callback(
                    status,
                    completed,
                    total,
                )

    def show_model(self, model_name: str) -> Any:
        return self._client().show(model_name)

    def supports_native_thinking(
        self,
        model_name: str,
    ) -> bool:
        """Best-effort detection using Ollama /api/show metadata."""
        try:
            response = self.show_model(model_name)
        except Exception:
            return False

        thinking = _value(response, "thinking")

        if not thinking:
            return False

        values = _value(thinking, "values", []) or []

        # [False] explicitly means thinking is unsupported.
        return any(value is not False for value in values)

    def delete_model(self, model_name: str) -> None:
        self._client().delete(model_name)