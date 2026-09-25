from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class OllamaModelVariant:
    tag: str
    parameters: str
    download: str
    download_bytes: int
    context: str
    profile: str


@dataclass(frozen=True)
class OllamaCatalogModel:
    model_id: str
    name: str
    publisher: str
    summary: str
    categories: tuple[str, ...] = field(default_factory=tuple)
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    official_url: str = ""
    default_tag: str = ""
    variants: tuple[OllamaModelVariant, ...] = field(default_factory=tuple)

    def variant_by_tag(self, tag: str) -> Optional[OllamaModelVariant]:
        return next((variant for variant in self.variants if variant.tag == tag), None)

    @property
    def smallest_variant(self) -> Optional[OllamaModelVariant]:
        return min(self.variants, key=lambda variant: variant.download_bytes, default=None)


class OllamaCuratedCatalog:
    """Curated discovery catalog shipped with CodeSmile.

    The local Ollama API can reliably list models already installed on the machine,
    but it does not provide a documented search endpoint for the whole public model
    library. CodeSmile therefore ships a small, explicit catalog of useful model
    families and keeps the advanced free-form Ollama tag path available.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else self.default_path()
        self.schema_version = 1
        self.verified_on = ""
        self.source = ""
        self.models: list[OllamaCatalogModel] = []
        self.reload()

    @staticmethod
    def default_path() -> Path:
        return Path(__file__).resolve().parents[1] / "config" / "ollama_model_catalog.json"

    def reload(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.schema_version = int(payload.get("schema_version", 1))
        self.verified_on = str(payload.get("verified_on", ""))
        self.source = str(payload.get("source", ""))

        models: list[OllamaCatalogModel] = []
        for item in payload.get("models", []):
            variants = tuple(
                OllamaModelVariant(
                    tag=str(variant["tag"]),
                    parameters=str(variant.get("parameters", "")),
                    download=str(variant.get("download", "")),
                    download_bytes=int(variant.get("download_bytes", 0) or 0),
                    context=str(variant.get("context", "")),
                    profile=str(variant.get("profile", "")),
                )
                for variant in item.get("variants", [])
            )
            models.append(
                OllamaCatalogModel(
                    model_id=str(item["model_id"]),
                    name=str(item.get("name", item["model_id"])),
                    publisher=str(item.get("publisher", "")),
                    summary=str(item.get("summary", "")),
                    categories=tuple(str(value).lower() for value in item.get("categories", [])),
                    capabilities=tuple(str(value) for value in item.get("capabilities", [])),
                    official_url=str(item.get("official_url", "")),
                    default_tag=str(item.get("default_tag", "")),
                    variants=variants,
                )
            )
        self.models = models

    def get(self, model_id: str) -> OllamaCatalogModel:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(model_id)

    def search(self, query: str = "", category: str = "all") -> list[OllamaCatalogModel]:
        needle = (query or "").strip().lower()
        category = (category or "all").strip().lower()

        def matches(model: OllamaCatalogModel) -> bool:
            if category not in {"", "all"}:
                if category == "lightweight":
                    has_light_variant = any(
                        variant.profile.lower() in {"tiny", "lightweight"}
                        for variant in model.variants
                    )
                    if "lightweight" not in model.categories and not has_light_variant:
                        return False
                elif category not in model.categories:
                    return False
            if not needle:
                return True
            searchable = " ".join(
                [
                    model.name,
                    model.publisher,
                    model.summary,
                    " ".join(model.categories),
                    " ".join(model.capabilities),
                    " ".join(variant.tag for variant in model.variants),
                ]
            ).lower()
            return needle in searchable

        return sorted((model for model in self.models if matches(model)), key=lambda model: model.name.lower())

    @staticmethod
    def installed_tags(installed_models: Iterable[dict]) -> set[str]:
        return {
            str(item.get("model") or item.get("name") or "").strip()
            for item in installed_models
            if str(item.get("model") or item.get("name") or "").strip()
        }
