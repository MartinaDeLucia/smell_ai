from __future__ import annotations

from typing import Optional

from llm_detection.credential_store import ApiCredentialStore
from llm_detection.providers import (
    ApiLLMProvider,
    LLMProvider,
    LocalLLMProvider,
)
from llm_detection.types import (
    LLMProviderDefinition,
    ProviderKind,
)


class LLMProviderFactory:
    """
    Creates runtime provider instances from catalog definitions.

    GUI and orchestrator do not need to know how each provider
    implementation is constructed.
    """

    @staticmethod
    def create(
        provider_def: LLMProviderDefinition,
        *,
        credential_store: Optional[ApiCredentialStore] = None,
    ) -> LLMProvider:

        config = dict(provider_def.config or {})

        # ---------------------------------------------------------
        # LOCAL / OLLAMA
        # ---------------------------------------------------------

        if provider_def.kind == ProviderKind.LOCAL:
            model_name = str(
                config.get("model_name") or ""
            ).strip()

            if not model_name:
                raise RuntimeError(
                    f"Local provider '{provider_def.provider_id}' "
                    "has no model_name."
                )

            host = (
                config.get("host")
                or config.get("base_url")
            )

            options = config.get("options")

            return LocalLLMProvider(
                model_name=model_name,
                host=str(host) if host else None,
                options=(
                    dict(options)
                    if isinstance(options, dict)
                    else None
                ),
                response_format=(
                    config.get("format")
                    or config.get("response_format")
                ),
                think=config.get("think"),
            )

        # ---------------------------------------------------------
        # REMOTE API
        # ---------------------------------------------------------

        if provider_def.kind == ProviderKind.API:
            protocol = str(
                config.get("protocol")
                or ApiLLMProvider.GENERIC_HTTP
            ).strip()

            base_url = str(
                config.get("base_url") or ""
            ).strip()

            if not base_url:
                raise RuntimeError(
                    f"API provider '{provider_def.provider_id}' "
                    "has no base_url."
                )

            api_key = None

            if protocol == ApiLLMProvider.OPENAI_COMPATIBLE:
                store = (
                    credential_store
                    or ApiCredentialStore()
                )

                api_key = store.get_api_key(
                    provider_def.provider_id
                )

                if not api_key:
                    raise RuntimeError(
                        f"No API key is stored for "
                        f"'{provider_def.display_name}'. "
                        "Open Manage API Providers and "
                        "save the credentials again."
                    )

            options = config.get("options")

            return ApiLLMProvider(
                base_url=base_url,
                model_name=(
                    str(
                        config.get("model_name")
                        or ""
                    ).strip()
                    or None
                ),
                api_key=api_key,
                protocol=protocol,
                timeout_s=float(
                    config.get("timeout_s", 60.0)
                ),
                options=(
                    dict(options)
                    if isinstance(options, dict)
                    else None
                ),
                supports_structured_output=bool(
                    config.get(
                        "supports_structured_output",
                        False,
                    )
                ),
            )

        raise RuntimeError(
            f"Unsupported provider kind: "
            f"{provider_def.kind}"
        )