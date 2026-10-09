from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from llm_detection.catalog_service import (
    CatalogValidationError,
    LLMCatalogService,
)
from llm_detection.credential_store import (
    ApiCredentialStore,
)
from llm_detection.providers import (
    ApiLLMProvider,
)
from llm_detection.types import (
    ProviderKind,
)


class ApiProviderManagerDialog(
    tk.Toplevel
):
    """
    Add, edit, test and remove remote API LLM providers.

    The initial supported protocol is OpenAI-compatible
    Chat Completions.

    Non-secret configuration:
        llm_catalog.json

    API key:
        OS credential store
    """

    PROTOCOL_LABELS = {
        "OpenAI-compatible":
            ApiLLMProvider.OPENAI_COMPATIBLE,
    }

    def __init__(
        self,
        parent,
        catalog_service: Optional[
            LLMCatalogService
        ] = None,
        credential_store: Optional[
            ApiCredentialStore
        ] = None,
        on_catalog_changed: Optional[
            Callable[[], None]
        ] = None,
        on_provider_ready: Optional[
            Callable[[str], None]
        ] = None,
    ):
        super().__init__(
            parent
        )

        self.catalog_service = (
            catalog_service
            or LLMCatalogService()
        )

        self.credential_store = (
            credential_store
            or ApiCredentialStore()
        )

        self.on_catalog_changed = (
            on_catalog_changed
        )

        self.on_provider_ready = (
            on_provider_ready
        )

        self.title(
            "CodeSmile · API Provider Manager"
        )

        self.geometry(
            "900x600"
        )

        self.transient(
            parent
        )

        self._selected_provider_id = (
            None
        )

        self._event_queue = (
            queue.Queue()
        )

        self._display_name_var = (
            tk.StringVar()
        )

        self._base_url_var = (
            tk.StringVar()
        )

        self._model_name_var = (
            tk.StringVar()
        )

        self._api_key_var = (
            tk.StringVar()
        )

        self._temperature_var = (
            tk.StringVar(
                value="0"
            )
        )

        self._max_tokens_var = (
            tk.StringVar(
                value="1024"
            )
        )

        self._timeout_var = (
            tk.StringVar(
                value="60"
            )
        )

        self._structured_output_var = (
            tk.BooleanVar(
                value=True
            )
        )

        self._status_var = (
            tk.StringVar(
                value="Ready"
            )
        )

        self._build_ui()
        self._reload_provider_list()

        self.after(
            100,
            self._drain_events,
        )

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------

    def _build_ui(
        self,
    ) -> None:

        self.grid_columnconfigure(
            0,
            weight=1,
        )

        self.grid_rowconfigure(
            1,
            weight=1,
        )

        header = ttk.Frame(
            self,
            padding=12,
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        ttk.Label(
            header,
            text="API Provider Manager",
            font=(
                "Segoe UI",
                16,
                "bold",
            ),
        ).pack(
            anchor="w"
        )

        ttk.Label(
            header,
            text=(
                "Configure remote LLMs "
                "without editing source code. "
                "API keys are stored separately."
            ),
        ).pack(
            anchor="w",
            pady=(4, 0),
        )

        body = ttk.Panedwindow(
            self,
            orient="horizontal",
        )

        body.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=12,
            pady=8,
        )

        left = ttk.Frame(
            body,
            padding=8,
        )

        right = ttk.Frame(
            body,
            padding=8,
        )

        body.add(
            left,
            weight=2,
        )

        body.add(
            right,
            weight=3,
        )

        # ---------------- LEFT ----------------

        left.grid_columnconfigure(
            0,
            weight=1,
        )

        left.grid_rowconfigure(
            1,
            weight=1,
        )

        ttk.Label(
            left,
            text="Configured providers",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 6),
        )

        columns = (
            "name",
            "model",
            "url",
        )

        self._tree = ttk.Treeview(
            left,
            columns=columns,
            show="headings",
        )

        self._tree.heading(
            "name",
            text="Name",
        )

        self._tree.heading(
            "model",
            text="Model",
        )

        self._tree.heading(
            "url",
            text="Base URL",
        )

        self._tree.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        self._tree.bind(
            "<<TreeviewSelect>>",
            lambda _event:
                self._load_selected_provider(),
        )

        actions = ttk.Frame(
            left
        )

        actions.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        ttk.Button(
            actions,
            text="New",
            command=self._clear_form,
        ).pack(
            side="left"
        )

        self._delete_btn = ttk.Button(
            actions,
            text="Remove",
            command=(
                self._remove_selected_provider
            ),
        )

        self._delete_btn.pack(
            side="right"
        )

        # ---------------- RIGHT ----------------

        right.grid_columnconfigure(
            1,
            weight=1,
        )

        self._add_entry(
            right,
            0,
            "Display name:",
            self._display_name_var,
        )

        self._add_entry(
            right,
            1,
            "Base URL:",
            self._base_url_var,
        )

        self._add_entry(
            right,
            2,
            "Model ID:",
            self._model_name_var,
        )

        ttk.Label(
            right,
            text="API key:",
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=5,
        )

        ttk.Entry(
            right,
            textvariable=(
                self._api_key_var
            ),
            show="•",
        ).grid(
            row=3,
            column=1,
            sticky="ew",
            pady=5,
        )

        ttk.Label(
            right,
            text=(
                "When editing, leave the "
                "field empty to keep the "
                "stored key."
            ),
        ).grid(
            row=4,
            column=1,
            sticky="w",
        )

        options = ttk.LabelFrame(
            right,
            text="Generation options",
            padding=8,
        )

        options.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=10,
        )

        options.grid_columnconfigure(
            1,
            weight=1,
        )

        self._add_entry(
            options,
            0,
            "Temperature:",
            self._temperature_var,
        )

        self._add_entry(
            options,
            1,
            "Max output tokens:",
            self._max_tokens_var,
        )

        self._add_entry(
            options,
            2,
            "Timeout (s):",
            self._timeout_var,
        )

        ttk.Checkbutton(
            options,
            text=(
                "Use JSON Schema "
                "structured output"
            ),
            variable=(
                self._structured_output_var
            ),
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )

        buttons = ttk.Frame(
            right
        )

        buttons.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=12,
        )

        self._test_btn = ttk.Button(
            buttons,
            text="Test Connection",
            command=(
                self._test_connection
            ),
        )

        self._test_btn.pack(
            side="left"
        )

        self._save_btn = ttk.Button(
            buttons,
            text="Save & Use",
            command=(
                self._save_and_use
            ),
        )

        self._save_btn.pack(
            side="right"
        )

        footer = ttk.Frame(
            self,
            padding=12,
        )

        footer.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        ttk.Label(
            footer,
            textvariable=(
                self._status_var
            ),
        ).pack(
            side="left"
        )

        ttk.Button(
            footer,
            text="Close",
            command=self.destroy,
        ).pack(
            side="right"
        )

    @staticmethod
    def _add_entry(
        parent,
        row: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:

        ttk.Label(
            parent,
            text=label,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            pady=5,
            padx=(0, 8),
        )

        ttk.Entry(
            parent,
            textvariable=variable,
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=5,
        )

    # ------------------------------------------------------
    # CATALOG
    # ------------------------------------------------------

    def _api_providers(
        self,
    ):
        return [
            provider
            for provider
            in self.catalog_service.load().providers

            if (
                provider.kind
                == ProviderKind.API

                and str(
                    provider.config.get(
                        "protocol"
                    )
                    or ""
                )
                == (
                    ApiLLMProvider
                    .OPENAI_COMPATIBLE
                )
            )
        ]

    def _reload_provider_list(
        self,
        select_provider_id=None,
    ) -> None:

        for item in (
            self._tree.get_children()
        ):
            self._tree.delete(
                item
            )

        for provider in (
            self._api_providers()
        ):

            self._tree.insert(
                "",
                "end",
                iid=(
                    provider.provider_id
                ),
                values=(
                    provider.display_name,
                    provider.config.get(
                        "model_name",
                        "",
                    ),
                    provider.config.get(
                        "base_url",
                        "",
                    ),
                ),
            )

            if (
                select_provider_id
                == provider.provider_id
            ):
                self._tree.selection_set(
                    provider.provider_id
                )

        if select_provider_id:
            self._load_selected_provider()

    def _clear_form(
        self,
    ) -> None:

        self._selected_provider_id = (
            None
        )

        self._display_name_var.set("")
        self._base_url_var.set("")
        self._model_name_var.set("")
        self._api_key_var.set("")
        self._temperature_var.set("0")
        self._max_tokens_var.set("1024")
        self._timeout_var.set("60")
        self._structured_output_var.set(
            True
        )

        self._status_var.set(
            "New provider"
        )

    def _load_selected_provider(
        self,
    ) -> None:

        selection = (
            self._tree.selection()
        )

        if not selection:
            return

        provider_id = selection[0]

        provider = (
            self.catalog_service
            .get_provider(
                provider_id
            )
        )

        config = (
            provider.config
        )

        self._selected_provider_id = (
            provider_id
        )

        self._display_name_var.set(
            provider.display_name
        )

        self._base_url_var.set(
            str(
                config.get(
                    "base_url"
                )
                or ""
            )
        )

        self._model_name_var.set(
            str(
                config.get(
                    "model_name"
                )
                or ""
            )
        )

        self._api_key_var.set(
            ""
        )

        options = (
            config.get(
                "options"
            )
        )

        if not isinstance(
            options,
            dict,
        ):
            options = {}

        self._temperature_var.set(
            str(
                options.get(
                    "temperature",
                    0,
                )
            )
        )

        self._max_tokens_var.set(
            str(
                options.get(
                    "max_tokens",
                    "",
                )
            )
        )

        self._timeout_var.set(
            str(
                config.get(
                    "timeout_s",
                    60,
                )
            )
        )

        self._structured_output_var.set(
            bool(
                config.get(
                    "supports_structured_output",
                    True,
                )
            )
        )

    def _form_values(
        self,
    ) -> dict:

        display_name = (
            self._display_name_var
            .get()
            .strip()
        )

        base_url = (
            self._base_url_var
            .get()
            .strip()
            .rstrip("/")
        )

        model_name = (
            self._model_name_var
            .get()
            .strip()
        )

        api_key = (
            self._api_key_var
            .get()
            .strip()
        )

        if not display_name:
            raise CatalogValidationError(
                "Display name is required"
            )

        if not base_url:
            raise CatalogValidationError(
                "Base URL is required"
            )

        if not model_name:
            raise CatalogValidationError(
                "Model ID is required"
            )

        if not (
            base_url.startswith(
                "http://"
            )
            or base_url.startswith(
                "https://"
            )
        ):
            raise CatalogValidationError(
                "Base URL must start "
                "with http:// or https://"
            )

        try:
            timeout_s = float(
                self._timeout_var.get()
                or "60"
            )

        except ValueError as exc:
            raise CatalogValidationError(
                "Timeout must be numeric"
            ) from exc

        options = {}

        temperature = (
            self._temperature_var
            .get()
            .strip()
        )

        if temperature:
            options[
                "temperature"
            ] = float(
                temperature
            )

        max_tokens = (
            self._max_tokens_var
            .get()
            .strip()
        )

        if max_tokens:
            options[
                "max_tokens"
            ] = int(
                max_tokens
            )

        return {
            "display_name":
                display_name,

            "base_url":
                base_url,

            "model_name":
                model_name,

            "api_key":
                api_key,

            "protocol":
                ApiLLMProvider
                .OPENAI_COMPATIBLE,

            "timeout_s":
                timeout_s,

            "options":
                options,

            "supports_structured_output":
                bool(
                    self._structured_output_var
                    .get()
                ),
        }

    def _resolve_api_key(
        self,
        values: dict,
    ) -> str:

        if values[
            "api_key"
        ]:
            return values[
                "api_key"
            ]

        if self._selected_provider_id:
            stored = (
                self.credential_store
                .get_api_key(
                    self._selected_provider_id
                )
            )

            if stored:
                return stored

        raise CatalogValidationError(
            "API key is required"
        )

    # ------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------

    def _test_connection(
        self,
    ) -> None:

        try:
            values = (
                self._form_values()
            )

            api_key = (
                self._resolve_api_key(
                    values
                )
            )

        except Exception as exc:
            messagebox.showerror(
                "Invalid configuration",
                str(exc),
                parent=self,
            )
            return

        provider = ApiLLMProvider(
            base_url=(
                values["base_url"]
            ),
            model_name=(
                values["model_name"]
            ),
            api_key=api_key,
            protocol=(
                values["protocol"]
            ),
            timeout_s=(
                values["timeout_s"]
            ),
            options=(
                values["options"]
            ),
            supports_structured_output=(
                values[
                    "supports_structured_output"
                ]
            ),
        )

        self._set_busy(
            True
        )

        self._status_var.set(
            "Testing API connection..."
        )

        def worker():
            try:
                result = (
                    provider
                    .test_connection()
                )

                self._event_queue.put(
                    (
                        "success",
                        result,
                    )
                )

            except Exception as exc:
                self._event_queue.put(
                    (
                        "error",
                        str(exc),
                    )
                )

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    def _save_and_use(
        self,
    ) -> None:

        try:
            values = (
                self._form_values()
            )

            api_key = (
                self._resolve_api_key(
                    values
                )
            )

            provider_id = (
                self.catalog_service
                .add_or_update_api_provider(
                    display_name=(
                        values[
                            "display_name"
                        ]
                    ),
                    base_url=(
                        values[
                            "base_url"
                        ]
                    ),
                    model_name=(
                        values[
                            "model_name"
                        ]
                    ),
                    protocol=(
                        values[
                            "protocol"
                        ]
                    ),
                    timeout_s=(
                        values[
                            "timeout_s"
                        ]
                    ),
                    options=(
                        values[
                            "options"
                        ]
                    ),
                    supports_structured_output=(
                        values[
                            "supports_structured_output"
                        ]
                    ),
                    existing_provider_id=(
                        self._selected_provider_id
                    ),
                )
            )

            self.credential_store.set_api_key(
                provider_id,
                api_key,
            )

        except Exception as exc:
            messagebox.showerror(
                "Could not save provider",
                str(exc),
                parent=self,
            )
            return

        self._selected_provider_id = (
            provider_id
        )

        self._api_key_var.set(
            ""
        )

        self._reload_provider_list(
            select_provider_id=(
                provider_id
            )
        )

        self._status_var.set(
            "Provider saved and ready"
        )

        if self.on_catalog_changed:
            self.on_catalog_changed()

        if self.on_provider_ready:
            self.on_provider_ready(
                provider_id
            )

    def _remove_selected_provider(
        self,
    ) -> None:

        selection = (
            self._tree.selection()
        )

        if not selection:
            return

        provider_id = (
            selection[0]
        )

        provider = (
            self.catalog_service
            .get_provider(
                provider_id
            )
        )

        if not messagebox.askyesno(
            "Remove provider",
            f"Remove "
            f"'{provider.display_name}'?",
            parent=self,
        ):
            return

        try:
            self.catalog_service.remove_provider(
                provider_id
            )

            self.credential_store.delete_api_key(
                provider_id
            )

        except Exception as exc:
            messagebox.showerror(
                "Error",
                str(exc),
                parent=self,
            )
            return

        self._clear_form()
        self._reload_provider_list()

        if self.on_catalog_changed:
            self.on_catalog_changed()

    # ------------------------------------------------------
    # THREAD EVENTS
    # ------------------------------------------------------

    def _set_busy(
        self,
        busy: bool,
    ) -> None:

        state = (
            "disabled"
            if busy
            else "normal"
        )

        self._test_btn.configure(
            state=state
        )

        self._save_btn.configure(
            state=state
        )

        self._delete_btn.configure(
            state=state
        )

    def _drain_events(
        self,
    ) -> None:

        try:
            while True:

                event = (
                    self._event_queue
                    .get_nowait()
                )

                if event[0] == "success":

                    result = event[1]

                    self._set_busy(
                        False
                    )

                    self._status_var.set(
                        "Connection successful"
                    )

                    messagebox.showinfo(
                        "Connection successful",
                        (
                            "The API responded "
                            "successfully.\n\n"
                            f"Response: "
                            f"{result.response[:200]}"
                        ),
                        parent=self,
                    )

                elif event[0] == "error":

                    self._set_busy(
                        False
                    )

                    self._status_var.set(
                        "Connection failed"
                    )

                    messagebox.showerror(
                        "API connection failed",
                        event[1],
                        parent=self,
                    )

        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(
                100,
                self._drain_events,
            )