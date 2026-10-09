from __future__ import annotations

from typing import Optional


class ApiCredentialStore:
    """
    Stores API keys outside CodeSmile's JSON catalog.

    llm_catalog.json contains only non-sensitive provider configuration.
    API keys are persisted through the operating-system credential store
    using the `keyring` package.
    """

    SERVICE_NAME = "CodeSmile"

    def _keyring(self):
        try:
            import keyring
        except Exception as exc:
            raise RuntimeError(
                "The 'keyring' package is required to store API credentials "
                "securely. Install project requirements and try again."
            ) from exc

        return keyring

    def set_api_key(
        self,
        provider_id: str,
        api_key: str,
    ) -> None:
        provider_id = (provider_id or "").strip()
        api_key = (api_key or "").strip()

        if not provider_id:
            raise ValueError("provider_id must not be empty")

        if not api_key:
            raise ValueError("API key must not be empty")

        self._keyring().set_password(
            self.SERVICE_NAME,
            provider_id,
            api_key,
        )

    def get_api_key(
        self,
        provider_id: str,
    ) -> Optional[str]:
        provider_id = (provider_id or "").strip()

        if not provider_id:
            return None

        value = self._keyring().get_password(
            self.SERVICE_NAME,
            provider_id,
        )

        return str(value).strip() if value else None

    def delete_api_key(
        self,
        provider_id: str,
    ) -> None:
        provider_id = (provider_id or "").strip()

        if not provider_id:
            return

        keyring = self._keyring()

        try:
            keyring.delete_password(
                self.SERVICE_NAME,
                provider_id,
            )
        except Exception:
            # Removing a provider must not fail only because
            # a credential was not stored.
            pass