from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from typing import Any, Optional

from llm_detection.types import (
    LLMCatalog,
    LLMSmellDefinition,
    LLMProviderDefinition,
    ProviderKind,
)


def _smell_from_dict(d: dict[str, Any]) -> LLMSmellDefinition:
    return LLMSmellDefinition(
        smell_id=d["smell_id"],
        display_name=d.get("display_name", d["smell_id"]),
        description=d.get("description", ""),
        default_prompt=d.get("default_prompt", ""),
        draft_prompt=d.get("draft_prompt"),
        created_by_user=bool(d.get("created_by_user", False)),
        enabled=bool(d.get("enabled", True)),
    )


def _provider_from_dict(d: dict[str, Any]) -> LLMProviderDefinition:
    return LLMProviderDefinition(
        provider_id=d["provider_id"],
        kind=ProviderKind(d["kind"]),
        display_name=d.get("display_name", d["provider_id"]),
        config=dict(d.get("config", {})),
    )


class LLMCatalogStore:
    """Loads/saves the CR01 catalog (smells/prompts/providers) from JSON."""

    def __init__(self, file_path: str = os.path.join("config", "llm_catalog.json")):
        self.file_path = file_path

    def exists(self) -> bool:
        return os.path.exists(self.file_path)

    def load(self) -> LLMCatalog:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        smells = [_smell_from_dict(x) for x in data.get("smells", [])]
        providers = [_provider_from_dict(x) for x in data.get("providers", [])]

        return LLMCatalog(
            schema_version=int(data.get("schema_version", 1)),
            smells=smells,
            providers=providers,
        )

    def save(self, catalog: LLMCatalog) -> None:
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)

        payload = asdict(catalog)
        # Enums are stored as their value (json-serializable)
        for p in payload.get("providers", []):
            if isinstance(p.get("kind"), ProviderKind):
                p["kind"] = p["kind"].value

        json_text = json.dumps(payload, ensure_ascii=False, indent=2)

        # Atomic-ish write on Windows: write to temp file then replace.
        dir_name = os.path.dirname(self.file_path) or "."
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=dir_name, suffix=".tmp"
        ) as tmp:
            tmp.write(json_text)
            tmp_path = tmp.name

        os.replace(tmp_path, self.file_path)

    def ensure_exists(self, seed: Optional[LLMCatalog] = None) -> LLMCatalog:
        if self.exists():
            return self.load()

        # Do not seed a phantom local model: a fresh CodeSmile installation may
        # have Ollama but no downloaded LLMs. Models are discovered/installed from
        # the Model Hub and registered here only after the user chooses one.
        catalog = seed or LLMCatalog(
            schema_version=1,
            smells=[],
            providers=[],
        )
        self.save(catalog)
        return catalog
