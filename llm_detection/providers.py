from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from llm_detection.types import LLMGenerationResult


def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a mapping-like Ollama object or a typed response."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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
    """Ollama-backed local provider.

    ``think`` is optional. When the selected Ollama model supports thinking,
    Ollama returns it separately from the final response. The orchestrator does
    not depend on this trace for detection: the portable per-finding rationale
    is requested in the JSON output itself.
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
            import ollama  # lazy import
        except Exception as e:
            raise RuntimeError(
                "ollama is not available; cannot use LocalLLMProvider"
            ) from e

        return ollama.Client(host=self.host) if self.host else ollama

    def generate(self, prompt: str) -> str:
        return self.generate_with_metadata(prompt).response

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMGenerationResult:

        client = self._client()

        # Keep system instructions separate from the user analysis input.
        messages: list[dict[str, str]] = []

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

        # A response format supplied for this specific generation takes
        # precedence over the provider-level default format.
        if response_format is not None:
            kwargs["format"] = response_format
        elif self.response_format is not None:
            kwargs["format"] = self.response_format

        if self.think is not None:
            kwargs["think"] = self.think

        try:
            response = client.chat(**kwargs)
        except Exception as e:
            host_hint = f" ({self.host})" if self.host else ""
            raise RuntimeError(
                "Failed to chat with Ollama"
                + host_hint
                + ". Ensure Ollama is installed/running "
                "and the model is available locally."
            ) from e

        message = _value(response, "message", {})

        final_response = str(_value(message, "content", "") or "")
        native_reasoning = str(_value(message, "thinking", "") or "")

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
            key: _value(response, key)
            for key in metadata_keys
            if _value(response, key) is not None
        }

        # Diagnostic only. Keep response and provider-native reasoning separate.
        if not final_response.strip() and native_reasoning.strip():
            metadata["empty_content_with_thinking"] = True

        return LLMGenerationResult(
            response=final_response,
            native_reasoning=native_reasoning,
            metadata=metadata,
        )


class ApiLLMProvider(LLMProvider):
    """Generic HTTP provider.

    Expected endpoint: POST /generate {"prompt": ...}. If a JSON response also
    contains ``reasoning`` or ``thinking``, it is preserved as provider-native
    reasoning metadata.

    ``system_prompt`` and ``response_format`` are accepted for interface
    compatibility, but are not sent because the generic /generate contract
    does not define support for them.
    """

    def __init__(self, base_url: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        return self.generate_with_metadata(prompt).response

    def generate_with_metadata(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMGenerationResult:
        try:
            import httpx  # lazy import
        except Exception as e:
            raise RuntimeError(
                "httpx is not available; cannot use ApiLLMProvider"
            ) from e

        url = f"{self.base_url}/generate"

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                url,
                json={"prompt": prompt},
            )
            resp.raise_for_status()

            is_json = (
                "application/json"
                in resp.headers.get("content-type", "")
            )
            data = resp.json() if is_json else None

        if isinstance(data, dict):
            response_text = str(
                data.get(
                    "response",
                    data.get("content", ""),
                )
            )

            native_reasoning = str(
                data.get(
                    "reasoning",
                    data.get("thinking", ""),
                )
                or ""
            )

            return LLMGenerationResult(
                response=response_text,
                native_reasoning=native_reasoning,
                metadata={
                    k: v
                    for k, v in data.items()
                    if k
                    not in {
                        "response",
                        "content",
                        "reasoning",
                        "thinking",
                    }
                },
            )

        return LLMGenerationResult(response=resp.text)


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