from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Iterable, Optional

from utils.file_utils import FileUtils

from llm_detection.catalog_store import LLMCatalogStore
from llm_detection.types import (
    DetectionTarget,
    LLMCatalog,
    LLMSmellDefinition,
    LLMProviderDefinition,
    PromptMode,
    ProviderKind,
)


class CatalogValidationError(ValueError):
    pass


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-_]", "", text)
    return text or "smell"


class LLMCatalogService:
    """High-level API for UC03/UC02.

    - UC03 (Manage Code Smells): add/remove/update description.
      Creation requires only name + description. Prompts are handled elsewhere.

    - UC02 (Prompt Engineering): save draft prompt, promote to default.

    Catalog is persisted via LLMCatalogStore.
    """

    def __init__(self, store: Optional[LLMCatalogStore] = None):
        self.store = store or LLMCatalogStore()

    def load(self) -> LLMCatalog:
        return self.store.ensure_exists()

    def save(self, catalog: LLMCatalog) -> None:
        self.store.save(catalog)

    # -------- UC03: Manage smells --------

    def add_smell(self, name: str, description: str) -> str:
        """Create a new user-defined smell with name + description.

        Prompts are initialized empty and smell is NOT ready for detection.
        Returns the new smell_id.
        """

        name = (name or "").strip()
        description = (description or "").strip()

        if not name:
            raise CatalogValidationError("Smell name must not be empty")
        if not description:
            raise CatalogValidationError("Smell description must not be empty")

        catalog = self.load()

        # Enforce unique display_name (case-insensitive)
        for existing in catalog.smells:
            if existing.display_name.strip().lower() == name.lower():
                raise CatalogValidationError(
                    f"A smell named '{name}' already exists"
                )
        base_id = _slugify(name)
        smell_id = self._next_available_smell_id(catalog, base_id)

        smell = LLMSmellDefinition(
            smell_id=smell_id,
            display_name=name,
            description=description,
            default_prompt="",
            draft_prompt="",
            created_by_user=True,
            enabled=False,
        )

        catalog.upsert_smell(smell)
        self.save(catalog)
        return smell_id

    def remove_smell(self, smell_id: str) -> None:
        catalog = self.load()
        if not any(s.smell_id == smell_id for s in catalog.smells):
            raise KeyError(f"Unknown smell_id: {smell_id}")

        catalog.smells = [s for s in catalog.smells if s.smell_id != smell_id]
        self.save(catalog)

    def update_smell_description(self, smell_id: str, description: str) -> None:
        description = (description or "").strip()
        if not description:
            raise CatalogValidationError("Smell description must not be empty")

        catalog = self.load()
        smell = catalog.get_smell(smell_id)
        updated = replace(smell, description=description)
        catalog.upsert_smell(updated)
        self.save(catalog)

    def list_smells(self) -> list[LLMSmellDefinition]:
        return list(self.load().smells)

    def list_detectable_smells(self) -> list[LLMSmellDefinition]:
        return [s for s in self.load().smells if s.is_ready_for_detection()]

    # -------- UC02: Prompt engineering --------

    def save_draft_prompt(self, smell_id: str, prompt_text: str) -> None:
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            raise CatalogValidationError("Draft prompt must not be empty")

        catalog = self.load()
        smell = catalog.get_smell(smell_id)
        updated = replace(smell, draft_prompt=prompt_text)
        catalog.upsert_smell(updated)
        self.save(catalog)

    def promote_draft_to_default(self, smell_id: str) -> None:
        catalog = self.load()
        smell = catalog.get_smell(smell_id)
        # mutate via method for consistency
        smell.save_draft_as_default()
        catalog.upsert_smell(smell)
        self.save(catalog)

    def get_prompt(self, smell_id: str, mode: PromptMode) -> str:
        catalog = self.load()
        return catalog.get_smell(smell_id).get_prompt(mode)

    # -------- Providers --------

    def list_providers(self) -> list[LLMProviderDefinition]:
        return list(self.load().providers)

    def get_provider(self, provider_id: str) -> LLMProviderDefinition:
        return self.load().get_provider(provider_id)

    def add_or_update_local_ollama_provider(
        self,
        model_name: str,
        *,
        host: Optional[str] = "http://localhost:11434",
        display_name: Optional[str] = None,
        options: Optional[dict] = None,
        response_format: object = "json",
        think: Optional[bool | str] = None,
    ) -> str:
        """Persist an Ollama model as a selectable local provider.

        If a provider for the same model/host already exists, it is updated
        instead of duplicated. Returns the provider_id.
        """
        model_name = (model_name or "").strip()
        if not model_name:
            raise CatalogValidationError("Model name must not be empty")

        normalized_host = (host or "").strip() or None
        catalog = self.load()

        existing = next(
            (
                p
                for p in catalog.providers
                if p.kind == ProviderKind.LOCAL
                and str(p.config.get("model_name") or "").strip() == model_name
                and ((str(p.config.get("host") or "").strip() or None) == normalized_host)
            ),
            None,
        )

        provider_id = (
            existing.provider_id
            if existing is not None
            else self._next_available_provider_id(
                catalog, f"local-ollama-{_slugify(model_name)}"
            )
        )

        config = dict(existing.config) if existing is not None else {}
        config["model_name"] = model_name
        if normalized_host:
            config["host"] = normalized_host
        else:
            config.pop("host", None)
        if options is not None:
            config["options"] = dict(options)
        if response_format is not None:
            config["format"] = response_format
        if think is not None:
            config["think"] = think
        else:
            config.pop("think", None)

        provider = LLMProviderDefinition(
            provider_id=provider_id,
            kind=ProviderKind.LOCAL,
            display_name=(display_name or f"Ollama - {model_name}").strip(),
            config=config,
        )
        catalog.upsert_provider(provider)
        self.save(catalog)
        return provider_id

    def add_or_update_api_provider(
            self,
            *,
            display_name: str,
            base_url: str,
            model_name: str,
            protocol: str = "openai_compatible",
            timeout_s: float = 60.0,
            options: Optional[dict] = None,
            supports_structured_output: bool = True,
            existing_provider_id: Optional[str] = None,
    ) -> str:
        """
        Persist a remote API provider.

        API keys are deliberately NOT accepted here,
        therefore they can never end up inside llm_catalog.json.
        """

        display_name = (
                display_name or ""
        ).strip()

        base_url = (
                base_url or ""
        ).strip().rstrip("/")

        model_name = (
                model_name or ""
        ).strip()

        protocol = (
                protocol or ""
        ).strip()

        if not display_name:
            raise CatalogValidationError(
                "Display name must not be empty"
            )

        if not base_url:
            raise CatalogValidationError(
                "Base URL must not be empty"
            )

        if not model_name:
            raise CatalogValidationError(
                "Model name must not be empty"
            )

        if protocol not in {
            "openai_compatible",
            "generic_http",
        }:
            raise CatalogValidationError(
                f"Unsupported API protocol: "
                f"{protocol}"
            )

        if timeout_s <= 0:
            raise CatalogValidationError(
                "Timeout must be greater "
                "than zero"
            )

        catalog = self.load()

        existing = None

        if existing_provider_id:
            existing = catalog.get_provider(
                existing_provider_id
            )

            if (
                    existing.kind
                    != ProviderKind.API
            ):
                raise CatalogValidationError(
                    f"Provider "
                    f"'{existing_provider_id}' "
                    "is not an API provider"
                )

        else:
            # Avoid duplicates with the same
            # protocol + URL + model.
            existing = next(
                (
                    provider
                    for provider
                    in catalog.providers
                    if (
                        provider.kind
                        == ProviderKind.API
                        and str(
                    provider.config.get(
                        "protocol"
                    )
                    or "generic_http"
                )
                        == protocol
                        and str(
                    provider.config.get(
                        "base_url"
                    )
                    or ""
                ).rstrip("/")
                        == base_url
                        and str(
                    provider.config.get(
                        "model_name"
                    )
                    or ""
                )
                        == model_name
                )
                ),
                None,
            )

        provider_id = (
            existing.provider_id
            if existing is not None
            else self._next_available_provider_id(
                catalog,
                f"api-{_slugify(display_name)}",
            )
        )

        config = (
            dict(existing.config)
            if existing is not None
            else {}
        )

        config.update(
            {
                "protocol":
                    protocol,

                "base_url":
                    base_url,

                "model_name":
                    model_name,

                "timeout_s":
                    float(timeout_s),

                "options":
                    dict(options or {}),

                "supports_structured_output":
                    bool(
                        supports_structured_output
                    ),
            }
        )

        provider = LLMProviderDefinition(
            provider_id=provider_id,
            kind=ProviderKind.API,
            display_name=display_name,
            config=config,
        )

        catalog.upsert_provider(
            provider
        )

        self.save(
            catalog
        )

        return provider_id

    def remove_provider(
            self,
            provider_id: str,
    ) -> None:

        catalog = self.load()

        if not any(
                provider.provider_id
                == provider_id
                for provider
                in catalog.providers
        ):
            raise KeyError(
                f"Unknown provider_id: "
                f"{provider_id}"
            )

        catalog.providers = [
            provider
            for provider
            in catalog.providers
            if provider.provider_id
               != provider_id
        ]

        self.save(
            catalog
        )

    # -------- Targets / path handling --------

    @staticmethod
    def build_targets_from_input_path(input_path: str) -> list[DetectionTarget]:
        """Build detection targets from an input path.

        Supports:
        - a single project folder containing one or more .py files (recursive)
        - a folder containing multiple projects (recursive)

        Returns one DetectionTarget per Python file.
        """

        filenames = FileUtils.get_python_files(input_path)
        if not filenames:
            raise CatalogValidationError(
                "Input path contains no Python files (.py)"
            )

        targets: list[DetectionTarget] = []
        for filename in filenames:
            with open(filename, "r", encoding="utf-8") as f:
                targets.append(DetectionTarget(filename=filename, code=f.read()))
        return targets

    @staticmethod
    def validate_prompt_engineering_input_path(input_path: str) -> None:
        """Validation tailored for Prompt Engineering.

        We recommend a single project folder to keep prompt experiments bounded.
        This is a strict validation that helps UI be robust.
        """

        input_path = (input_path or "").strip()
        if not input_path:
            raise CatalogValidationError("Input path must not be empty")
        if not os.path.isdir(input_path):
            raise CatalogValidationError("Input path must be a directory")

        # Must contain at least 1 python file
        if not FileUtils.get_python_files(input_path):
            raise CatalogValidationError(
                "Input path must contain at least one .py file"
            )

        # Heuristic: if there are multiple immediate subdirectories each containing
        # python files and there are no python files at root, treat as multi-project.
        root_py = [
            f
            for f in os.listdir(input_path)
            if f.endswith(".py") and os.path.isfile(os.path.join(input_path, f))
        ]
        if root_py:
            return

        project_like_dirs = 0
        for name in os.listdir(input_path):
            p = os.path.join(input_path, name)
            if not os.path.isdir(p):
                continue
            if FileUtils.get_python_files(p):
                project_like_dirs += 1
                if project_like_dirs > 1:
                    raise CatalogValidationError(
                        "Prompt engineering expects a single project folder; "
                        "input path looks like it contains multiple projects"
                    )

    # -------- Internal helpers --------

    @staticmethod
    def _next_available_smell_id(catalog: LLMCatalog, base_id: str) -> str:
        existing = {s.smell_id for s in catalog.smells}
        if base_id not in existing:
            return base_id
        i = 2
        while f"{base_id}-{i}" in existing:
            i += 1
        return f"{base_id}-{i}"

    @staticmethod
    def _next_available_provider_id(catalog: LLMCatalog, base_id: str) -> str:
        existing = {p.provider_id for p in catalog.providers}
        if base_id not in existing:
            return base_id
        i = 2
        while f"{base_id}-{i}" in existing:
            i += 1
        return f"{base_id}-{i}"
