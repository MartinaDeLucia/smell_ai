from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ProviderKind(str, Enum):
    LOCAL = "local"
    API = "api"


class PromptMode(str, Enum):
    DEFAULT = "default"
    DRAFT = "draft"
    DRAFT_IF_AVAILABLE = "draft_if_available"


class NormalizationMode(str, Enum):
    """Controls how tolerant the orchestrator is when normalizing LLM output.

    - STRICT: accept ONLY the declared schema ({"findings": [...]}); on any mismatch return no findings.
    - SALVAGE: best-effort recovery (fences/prose tolerated; alternate top-level keys can be salvaged).
    """

    STRICT = "strict"
    SALVAGE = "salvage"

class DetectionStatus(str, Enum):
    SUCCESS = "success"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class LLMGenerationResult:
    """Provider output with optional provider-native reasoning metadata.

    ``response`` is the final answer consumed by the orchestrator.
    ``native_reasoning`` stores a provider-native thinking trace when the
    selected model exposes one (for example Ollama thinking-capable models).
    The tool's user-facing per-finding rationale is kept separately in
    ``LLMSmellFinding.reasoning`` so it works with every provider.
    """

    response: str
    native_reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMProviderDefinition:
    """Persisted configuration for a provider selectable from UI."""

    provider_id: str
    kind: ProviderKind
    display_name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMSmellDefinition:
    """A smell that can be detected via LLM prompts."""

    smell_id: str
    display_name: str
    description: str
    default_prompt: str
    draft_prompt: Optional[str] = None
    created_by_user: bool = True
    # In CR01 use cases, a smell becomes usable for detection once a default prompt
    # has been saved (prompt engineering). Keeping this explicit allows toggling.
    enabled: bool = False

    def is_ready_for_detection(self) -> bool:
        return self.enabled and bool(self.default_prompt.strip())

    def get_prompt(self, prompt_mode: PromptMode) -> str:
        if prompt_mode == PromptMode.DEFAULT:
            if not self.default_prompt.strip():
                raise ValueError(
                    f"Default prompt is empty for smell_id='{self.smell_id}'"
                )
            return self.default_prompt
        if prompt_mode == PromptMode.DRAFT:
            if self.draft_prompt is None or not self.draft_prompt.strip():
                raise ValueError(
                    f"No draft prompt available for smell_id='{self.smell_id}'"
                )
            return self.draft_prompt
        if prompt_mode == PromptMode.DRAFT_IF_AVAILABLE:
            if self.draft_prompt is not None and self.draft_prompt.strip():
                return self.draft_prompt
            if not self.default_prompt.strip():
                raise ValueError(
                    f"No default prompt available for smell_id='{self.smell_id}'"
                )
            return self.default_prompt
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    def save_draft_as_default(self) -> None:
        """Promote the current draft prompt to default and enable detection."""
        if self.draft_prompt is None or not self.draft_prompt.strip():
            raise ValueError(
                f"Cannot promote empty draft to default for smell_id='{self.smell_id}'"
            )
        self.default_prompt = self.draft_prompt
        self.enabled = True


@dataclass
class LLMCatalog:
    """Root persisted object for CR01 (smells + prompts + providers)."""

    schema_version: int = 1
    smells: list[LLMSmellDefinition] = field(default_factory=list)
    providers: list[LLMProviderDefinition] = field(default_factory=list)

    def get_smell(self, smell_id: str) -> LLMSmellDefinition:
        for smell in self.smells:
            if smell.smell_id == smell_id:
                return smell
        raise KeyError(f"Unknown smell_id: {smell_id}")

    def upsert_smell(self, smell: LLMSmellDefinition) -> None:
        for index, existing in enumerate(self.smells):
            if existing.smell_id == smell.smell_id:
                self.smells[index] = smell
                return
        self.smells.append(smell)

    def get_provider(self, provider_id: str) -> LLMProviderDefinition:
        for provider in self.providers:
            if provider.provider_id == provider_id:
                return provider
        raise KeyError(f"Unknown provider_id: {provider_id}")

    def upsert_provider(self, provider: LLMProviderDefinition) -> None:
        for index, existing in enumerate(self.providers):
            if existing.provider_id == provider.provider_id:
                self.providers[index] = provider
                return
        self.providers.append(provider)


@dataclass(frozen=True)
class DetectionTarget:
    filename: str
    code: str


@dataclass(frozen=True)
class LLMSmellFinding:
    """Normalized LLM output, designed to be convertible to the existing CSV schema."""

    filename: str
    function_name: str
    smell_name: str
    line: int
    description: str
    reasoning: str = ""
    additional_info: str = ""
    smell_id: Optional[str] = None
    confidence: Optional[float] = None
    raw_response: Optional[str] = None

    def to_overview_row(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "function_name": self.function_name,
            "smell_name": self.smell_name,
            "line": self.line,
            "description": self.description,
            "reasoning": self.reasoning,
            "additional_info": self.additional_info,
        }


@dataclass(frozen=True)
class DetectionResult:
    filename: str
    smell_id: str
    status: DetectionStatus
    findings: tuple[LLMSmellFinding, ...] = ()
    raw_response: Optional[str] = None
    error: Optional[str] = None