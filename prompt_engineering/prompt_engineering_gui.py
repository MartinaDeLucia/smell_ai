from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from datetime import datetime
from time import monotonic
from typing import Any, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from gui.api_provider_manager_gui import ApiProviderManagerDialog
from gui.folder_utils import open_folder
from gui.manage_code_smells_gui import AddSmellDialog
from gui.ollama_model_manager_gui import OllamaModelManagerDialog
from llm_detection.catalog_service import (
    CatalogValidationError,
    LLMCatalogService,
)
from llm_detection.orchestrator import LLMOrchestrator
from llm_detection.provider_factory import LLMProviderFactory
from llm_detection.types import (
    DetectionTarget,
    PromptMode,
    ProviderKind,
)
from utils.file_utils import FileUtils


class PromptEngineeringGUI:
    def __init__(
        self,
        master: tk.Tk,
        catalog_service: Optional[LLMCatalogService] = None,
    ):
        self.master = master
        self.catalog_service = catalog_service or LLMCatalogService()

        # ---------------------------------------------------------
        # STATE
        # ---------------------------------------------------------

        self._current_smell_id: Optional[str] = None
        self._draft_dirty: bool = False
        self._ui_disabled_no_smells: bool = False

        self._mode_var = tk.StringVar(
            value=PromptMode.DRAFT.value
        )

        self._smell_display_to_id: dict[str, str] = {}

        # Provider selection is generic:
        # Local Ollama OR remote API.
        self._provider_type_var = tk.StringVar(
            value=ProviderKind.LOCAL.value
        )

        self._provider_display_to_id: dict[str, str] = {}

        self._selected_provider_id: Optional[str] = None

        self._input_path_value: str = ""
        self._output_path_value: str = ""

        # Background execution state.
        self._cancel_event = threading.Event()

        self._run_started_at: Optional[float] = None
        self._heartbeat_after_id: Optional[str] = None

        self._running_total: int = 0
        self._running_index: int = 0
        self._running_filename: str = ""

        # ---------------------------------------------------------
        # UI
        # ---------------------------------------------------------

        self._build_ui()

        self._load_smells_into_dropdown()
        self._load_providers_into_dropdown()

        self._sync_test_button_state()

        self.master.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self) -> None:
        self.master.title("Prompt Engineering")
        self.master.geometry("900x700")

        style = ttk.Style(self.master)

        try:
            style.configure(
                "Calm.Horizontal.TProgressbar",
                thickness=8,
            )
        except Exception:
            pass

        # ---------------------------------------------------------
        # TOP: smell selector
        # ---------------------------------------------------------

        top = ttk.Frame(self.master)

        top.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=10,
            pady=10,
        )

        self.master.grid_columnconfigure(
            0,
            weight=1,
        )

        self.master.grid_rowconfigure(
            0,
            weight=0,
        )

        self.master.grid_rowconfigure(
            1,
            weight=1,
        )

        self.master.grid_rowconfigure(
            2,
            weight=0,
        )

        self.master.grid_rowconfigure(
            3,
            weight=0,
        )

        self.master.grid_rowconfigure(
            4,
            weight=3,
        )

        top.grid_columnconfigure(
            0,
            weight=1,
        )

        top.grid_columnconfigure(
            1,
            weight=0,
        )

        top.grid_columnconfigure(
            2,
            weight=0,
        )

        ttk.Label(
            top,
            text="Code smell:",
        ).grid(
            row=0,
            column=0,
            sticky="e",
            padx=(0, 6),
        )

        self._smell_combo = ttk.Combobox(
            top,
            state="readonly",
            width=50,
        )

        self._smell_combo.grid(
            row=0,
            column=1,
            sticky="w",
        )

        self._smell_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._on_smell_selected(),
        )

        self._add_smell_btn = ttk.Button(
            top,
            text="+",
            width=3,
            command=self._on_add_smell,
        )

        self._add_smell_btn.grid(
            row=0,
            column=2,
            sticky="w",
            padx=(6, 0),
        )

        # ---------------------------------------------------------
        # PROMPT
        # ---------------------------------------------------------

        mode_frame = ttk.LabelFrame(
            self.master,
            text="Prompt",
        )

        mode_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=10,
        )

        mode_frame.grid_columnconfigure(
            0,
            weight=1,
        )

        mode_frame.grid_rowconfigure(
            2,
            weight=1,
        )

        best = ttk.Label(
            mode_frame,
            text=(
                "Best practices: in the smell prompt write ONLY "
                "the smell definition and smell-specific detection rules. "
                "Do not include the JSON schema or output contracts: "
                "those are already enforced by the orchestrator."
            ),
            foreground="#444",
            wraplength=820,
        )

        best.grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(8, 0),
        )

        radio_row = ttk.Frame(
            mode_frame
        )

        radio_row.grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=(8, 4),
        )

        self._draft_radio = ttk.Radiobutton(
            radio_row,
            text="Temporary (editable)",
            value=PromptMode.DRAFT.value,
            variable=self._mode_var,
            command=self._on_prompt_mode_changed,
        )

        self._draft_radio.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 12),
        )

        self._default_radio = ttk.Radiobutton(
            radio_row,
            text="Default (read-only)",
            value=PromptMode.DEFAULT.value,
            variable=self._mode_var,
            command=self._on_prompt_mode_changed,
        )

        self._default_radio.grid(
            row=0,
            column=1,
            sticky="w",
        )

        self._prompt_text = ScrolledText(
            mode_frame,
            height=8,
            wrap="word",
        )

        self._prompt_text.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=10,
            pady=(4, 10),
        )

        self._prompt_text.bind(
            "<KeyRelease>",
            self._on_prompt_edited,
        )

        # ---------------------------------------------------------
        # PATHS / PROVIDER
        # ---------------------------------------------------------

        paths = ttk.LabelFrame(
            self.master,
            text="Paths / Provider",
        )

        paths.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=10,
            pady=10,
        )

        paths.grid_columnconfigure(
            1,
            weight=1,
        )

        paths.grid_columnconfigure(
            2,
            weight=0,
        )

        # ---------------- Input path ----------------

        ttk.Label(
            paths,
            text="Input path:",
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=6,
        )

        self._input_path_label = ttk.Label(
            paths,
            text="No path selected",
        )

        self._input_path_label.grid(
            row=0,
            column=1,
            sticky="ew",
            pady=6,
        )

        self._choose_input_btn = ttk.Button(
            paths,
            text="Choose Input Folder",
            command=self._choose_input_path,
        )

        self._choose_input_btn.grid(
            row=0,
            column=2,
            sticky="e",
            padx=10,
            pady=6,
        )

        # ---------------- Output path ----------------

        ttk.Label(
            paths,
            text="Output path:",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=6,
        )

        self._output_path_label = ttk.Label(
            paths,
            text="No path selected",
        )

        self._output_path_label.grid(
            row=1,
            column=1,
            sticky="ew",
            pady=6,
        )

        output_buttons_frame = ttk.Frame(
            paths
        )

        output_buttons_frame.grid(
            row=1,
            column=2,
            sticky="e",
            padx=10,
            pady=6,
        )

        self._choose_output_btn = ttk.Button(
            output_buttons_frame,
            text="Choose Folder",
            command=self._choose_output_path,
        )

        self._choose_output_btn.pack(
            side="left",
            padx=(0, 5),
        )

        self._open_output_btn = ttk.Button(
            output_buttons_frame,
            text="Open Results Folder",
            command=self._open_output_folder,
            state="disabled",
        )

        self._open_output_btn.pack(
            side="left"
        )

        # ---------------- Provider type ----------------

        ttk.Label(
            paths,
            text="Provider type:",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=10,
            pady=(6, 4),
        )

        provider_type_frame = ttk.Frame(
            paths
        )

        provider_type_frame.grid(
            row=2,
            column=1,
            sticky="w",
            pady=(6, 4),
        )

        self._local_provider_radio = ttk.Radiobutton(
            provider_type_frame,
            text="Local",
            value=ProviderKind.LOCAL.value,
            variable=self._provider_type_var,
            command=self._on_provider_type_changed,
        )

        self._local_provider_radio.pack(
            side="left",
            padx=(0, 12),
        )

        self._api_provider_radio = ttk.Radiobutton(
            provider_type_frame,
            text="API",
            value=ProviderKind.API.value,
            variable=self._provider_type_var,
            command=self._on_provider_type_changed,
        )

        self._api_provider_radio.pack(
            side="left",
        )

        # ---------------- Provider selector ----------------

        ttk.Label(
            paths,
            text="LLM provider:",
        ).grid(
            row=3,
            column=0,
            sticky="w",
            padx=10,
            pady=(4, 10),
        )

        self._provider_combo = ttk.Combobox(
            paths,
            state="readonly",
            width=50,
        )

        self._provider_combo.grid(
            row=3,
            column=1,
            sticky="w",
            pady=(4, 10),
        )

        self._provider_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._on_provider_selected(),
        )

        self._manage_provider_btn = ttk.Button(
            paths,
            text="Browse / Download Models",
            command=self._open_provider_manager,
        )

        self._manage_provider_btn.grid(
            row=3,
            column=2,
            sticky="e",
            padx=10,
            pady=(4, 10),
        )

        # ---------------------------------------------------------
        # ACTIONS
        # ---------------------------------------------------------

        actions = ttk.Frame(
            self.master
        )

        actions.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=10,
        )

        actions.grid_columnconfigure(
            0,
            weight=1,
        )

        self._test_btn = ttk.Button(
            actions,
            text="Test with LLM",
            command=self._on_test_clicked,
        )

        self._test_btn.grid(
            row=0,
            column=0,
            sticky="w",
        )

        self._cancel_btn = ttk.Button(
            actions,
            text="Cancel",
            command=self._on_cancel_clicked,
        )

        self._cancel_btn.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(160, 0),
        )

        self._cancel_btn.configure(
            state="disabled"
        )

        self._save_default_btn = ttk.Button(
            actions,
            text="Save temporary as default",
            command=self._on_save_default_clicked,
        )

        self._save_default_btn.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(10, 0),
        )

        self._exit_btn = ttk.Button(
            actions,
            text="Exit",
            command=self._on_close,
        )

        self._exit_btn.grid(
            row=0,
            column=2,
            sticky="e",
        )

        self._status_var = tk.StringVar(
            value="Idle"
        )

        self._status_label = ttk.Label(
            actions,
            textvariable=self._status_var,
        )

        self._status_label.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

        self._progress = ttk.Progressbar(
            actions,
            mode="determinate",
            style="Calm.Horizontal.TProgressbar",
        )

        self._progress.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(4, 0),
        )

        # ---------------------------------------------------------
        # OUTPUT / LOG
        # ---------------------------------------------------------

        out = ttk.LabelFrame(
            self.master,
            text="Results / Log",
        )

        out.grid(
            row=4,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10,
        )

        out.grid_columnconfigure(
            0,
            weight=1,
        )

        out.grid_rowconfigure(
            0,
            weight=1,
        )

        self._output_text = ScrolledText(
            out,
            height=12,
            wrap="word",
            state="disabled",
        )

        self._output_text.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10,
        )

    # ============================================================
    # DATA LOADING
    # ============================================================

    def _load_smells_into_dropdown(self) -> None:
        catalog = self.catalog_service.load()

        self._smell_display_to_id.clear()

        values: list[str] = []

        for smell in sorted(
            catalog.smells,
            key=lambda s: s.display_name.lower(),
        ):
            label = (
                f"{smell.display_name}  "
                f"({smell.smell_id})"
            )

            self._smell_display_to_id[
                label
            ] = smell.smell_id

            values.append(
                label
            )

        self._smell_combo[
            "values"
        ] = values

        if not values:
            self._ui_disabled_no_smells = True

            self._disable_all_controls_no_smells()

            self._append_output(
                "The smell catalog is empty: add at least one smell with '+' to continue.\n"
            )

            return

        self._ui_disabled_no_smells = False

        self._smell_combo.configure(
            state="readonly"
        )

        self._draft_radio.configure(
            state="normal"
        )

        self._default_radio.configure(
            state="normal"
        )

        self._add_smell_btn.configure(
            state="normal"
        )

        self._local_provider_radio.configure(
            state="normal"
        )

        self._api_provider_radio.configure(
            state="normal"
        )

        self._manage_provider_btn.configure(
            state="normal"
        )

        self._smell_combo.current(
            0
        )

        self._on_smell_selected()

    def _load_providers_into_dropdown(
        self,
    ) -> None:
        """
        Load either Local or API providers according to the
        currently selected provider type.
        """

        catalog = self.catalog_service.load()

        kind = ProviderKind(
            self._provider_type_var.get()
        )

        self._provider_display_to_id.clear()

        values: list[str] = []

        for provider in catalog.providers:
            if provider.kind != kind:
                continue

            model = provider.config.get(
                "model_name"
            )

            model_part = (
                f" | model: {model}"
                if model
                else ""
            )

            label = (
                f"{provider.display_name} "
                f"({provider.provider_id})"
                f"{model_part}"
            )

            self._provider_display_to_id[
                label
            ] = provider.provider_id

            values.append(
                label
            )

        self._provider_combo[
            "values"
        ] = values

        self._update_manage_provider_button()

        if self._ui_disabled_no_smells:
            self._provider_combo.set(
                ""
            )

            self._provider_combo.configure(
                state="disabled"
            )

            self._selected_provider_id = None

            self._sync_test_button_state()

            return

        if not values:
            self._provider_combo.set(
                ""
            )

            self._provider_combo.configure(
                state="disabled"
            )

            self._selected_provider_id = None

            self._sync_test_button_state()

            return

        self._provider_combo.configure(
            state="readonly"
        )

        self._provider_combo.current(
            0
        )

        self._on_provider_selected()

    def _on_provider_type_changed(
        self,
    ) -> None:
        self._selected_provider_id = None

        self._load_providers_into_dropdown()

    def _update_manage_provider_button(
        self,
    ) -> None:
        kind = ProviderKind(
            self._provider_type_var.get()
        )

        if kind == ProviderKind.LOCAL:
            self._manage_provider_btn.configure(
                text="Browse / Download Models",
                command=self._open_model_manager,
            )
        else:
            self._manage_provider_btn.configure(
                text="Manage API Providers",
                command=self._open_api_provider_manager,
            )

    def _open_provider_manager(
        self,
    ) -> None:
        kind = ProviderKind(
            self._provider_type_var.get()
        )

        if kind == ProviderKind.LOCAL:
            self._open_model_manager()
        else:
            self._open_api_provider_manager()

    def _open_model_manager(
        self,
    ) -> None:
        OllamaModelManagerDialog(
            self.master,
            catalog_service=self.catalog_service,
            on_catalog_changed=self._load_providers_into_dropdown,
            on_model_ready=self._select_local_provider_by_id,
        )

    def _open_api_provider_manager(
        self,
    ) -> None:
        ApiProviderManagerDialog(
            self.master,
            catalog_service=self.catalog_service,
            on_catalog_changed=self._load_providers_into_dropdown,
            on_provider_ready=self._select_api_provider_by_id,
        )

    def _select_local_provider_by_id(
        self,
        provider_id: str,
    ) -> None:
        self._select_provider_by_id(
            provider_id,
            ProviderKind.LOCAL,
        )

    def _select_api_provider_by_id(
        self,
        provider_id: str,
    ) -> None:
        self._select_provider_by_id(
            provider_id,
            ProviderKind.API,
        )

    def _select_provider_by_id(
        self,
        provider_id: str,
        kind: ProviderKind,
    ) -> None:
        self._provider_type_var.set(
            kind.value
        )

        self._load_providers_into_dropdown()

        for display, pid in (
            self._provider_display_to_id.items()
        ):
            if pid == provider_id:
                self._provider_combo.set(
                    display
                )

                self._on_provider_selected()

                break

    def _on_provider_selected(
        self,
    ) -> None:
        selected = self._provider_combo.get()

        self._selected_provider_id = (
            self._provider_display_to_id.get(
                selected
            )
        )

        self._sync_test_button_state()

    def _sync_test_button_state(
        self,
    ) -> None:
        """
        Enable Test only when the essential prerequisites exist.
        """

        if self._ui_disabled_no_smells:
            self._test_btn.configure(
                state="disabled"
            )
            return

        if not self._current_smell_id:
            self._test_btn.configure(
                state="disabled"
            )
            return

        if not (
            self._selected_provider_id
            or ""
        ).strip():
            self._test_btn.configure(
                state="disabled"
            )
            return

        if self._run_started_at is not None:
            self._test_btn.configure(
                state="disabled"
            )
            return

        self._test_btn.configure(
            state="normal"
        )

    def _disable_all_controls_no_smells(
        self,
    ) -> None:
        self._smell_combo.set("")

        self._smell_combo.configure(
            state="disabled"
        )

        self._add_smell_btn.configure(
            state="normal"
        )

        self._draft_radio.configure(
            state="disabled"
        )

        self._default_radio.configure(
            state="disabled"
        )

        self._prompt_text.configure(
            state="disabled"
        )

        self._test_btn.configure(
            state="disabled"
        )

        self._save_default_btn.configure(
            state="disabled"
        )

        self._provider_combo.configure(
            state="disabled"
        )

        self._manage_provider_btn.configure(
            state="disabled"
        )

        self._local_provider_radio.configure(
            state="disabled"
        )

        self._api_provider_radio.configure(
            state="disabled"
        )

    # ============================================================
    # UI EVENTS
    # ============================================================

    def _on_add_smell(
        self,
    ) -> None:
        def on_success(
            new_smell_id: str,
        ) -> None:
            self._load_smells_into_dropdown()
            self._load_providers_into_dropdown()

            for display, sid in (
                self._smell_display_to_id.items()
            ):
                if sid == new_smell_id:
                    self._smell_combo.set(
                        display
                    )

                    self._on_smell_selected()

                    break

        AddSmellDialog(
            self.master,
            self.catalog_service,
            on_success,
        )

    def _on_smell_selected(
        self,
    ) -> None:
        if not self._confirm_discard_unsaved_draft_if_needed(
            context="switching smell"
        ):
            self._restore_combo_to_current_smell()
            return

        selected = self._smell_combo.get()

        self._current_smell_id = (
            self._smell_display_to_id.get(
                selected
            )
        )

        self._draft_dirty = False

        self._refresh_prompt_view()

        self._sync_test_button_state()

    def _restore_combo_to_current_smell(
        self,
    ) -> None:
        if not self._current_smell_id:
            return

        for i, display in enumerate(
            self._smell_combo["values"]
        ):
            if (
                self._smell_display_to_id.get(
                    display
                )
                == self._current_smell_id
            ):
                self._smell_combo.current(i)
                return

    def _on_prompt_mode_changed(
        self,
    ) -> None:
        if (
            self._mode_var.get()
            == PromptMode.DEFAULT.value
        ):
            if not self._confirm_discard_unsaved_draft_if_needed(
                context="switching to the default prompt"
            ):
                self._mode_var.set(
                    PromptMode.DRAFT.value
                )
                return

        self._draft_dirty = False
        self._refresh_prompt_view()

    def _refresh_prompt_view(
        self,
    ) -> None:
        if not self._current_smell_id:
            self._set_prompt_text(
                "",
                editable=False,
            )
            return

        mode = PromptMode(
            self._mode_var.get()
        )

        catalog = self.catalog_service.load()
        smell = catalog.get_smell(
            self._current_smell_id
        )

        if mode == PromptMode.DRAFT:
            self._set_prompt_text(
                smell.draft_prompt or "",
                editable=True,
            )
        else:
            self._set_prompt_text(
                smell.default_prompt or "",
                editable=False,
            )

    def _set_prompt_text(
        self,
        text: str,
        editable: bool,
    ) -> None:
        self._prompt_text.configure(
            state="normal"
        )

        self._prompt_text.delete(
            "1.0",
            "end",
        )

        self._prompt_text.insert(
            "1.0",
            text,
        )

        self._prompt_text.configure(
            state="normal" if editable else "disabled"
        )

    def _on_prompt_edited(
        self,
        _event,
    ) -> None:
        if (
            self._mode_var.get()
            != PromptMode.DRAFT.value
        ):
            return

        self._draft_dirty = True

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    def _choose_input_path(
        self,
    ) -> None:
        path = filedialog.askdirectory()

        if path:
            self._input_path_value = path
            self._input_path_label.configure(
                text=path
            )

    def _choose_output_path(
        self,
    ) -> None:
        path = filedialog.askdirectory()

        if path:
            self._output_path_value = path
            self._output_path_label.configure(
                text=path
            )
            self._open_output_btn.configure(
                state="normal"
            )

    def _open_output_folder(
        self,
    ) -> None:
        """
        Open the Prompt Engineering result directory.

        Results are stored inside <selected_path>/output.
        Before the first run, the selected parent directory is opened.
        """

        base_path = (
            self._output_path_value
            or ""
        ).strip()

        if not base_path:
            messagebox.showwarning(
                "Output folder",
                "Select an output folder first.",
            )
            return

        results_path = os.path.join(
            base_path,
            "output",
        )

        path_to_open = (
            results_path
            if os.path.isdir(results_path)
            else base_path
        )

        try:
            open_folder(path_to_open)
        except Exception as exc:
            messagebox.showerror(
                "Output folder",
                f"Could not open the output folder:\n{exc}",
            )

    # ------------------------------------------------------------
    # Run test
    # ------------------------------------------------------------

    def _on_test_clicked(
        self,
    ) -> None:
        smell_id = self._current_smell_id

        if not smell_id:
            messagebox.showerror(
                "Error",
                "Select a smell.",
            )
            return

        provider_id = (
            self._selected_provider_id
            or ""
        ).strip()

        if not provider_id:
            messagebox.showerror(
                "Error",
                "No LLM provider selected.",
            )
            return

        mode = PromptMode(
            self._mode_var.get()
        )

        input_path = (
            self._input_path_value
            or ""
        ).strip()

        output_path = (
            self._output_path_value
            or ""
        ).strip()

        try:
            self.catalog_service.validate_prompt_engineering_input_path(
                input_path
            )
        except CatalogValidationError as exc:
            messagebox.showerror(
                "Invalid input path",
                str(exc),
            )
            return

        if not output_path:
            messagebox.showerror(
                "Error",
                "Missing output path.",
            )
            return

        prompt_text = (
            self._get_current_prompt_text()
            .strip()
        )

        if not prompt_text:
            messagebox.showerror(
                "Error",
                "The prompt is empty.",
            )
            return

        if mode == PromptMode.DRAFT:
            try:
                self.catalog_service.save_draft_prompt(
                    smell_id,
                    prompt_text,
                )
                self._draft_dirty = False
            except CatalogValidationError as exc:
                messagebox.showerror(
                    "Error",
                    str(exc),
                )
                return

        python_files = FileUtils.get_python_files(
            input_path
        )

        if not python_files:
            messagebox.showerror(
                "Error",
                "The input path contains no Python files (.py).",
            )
            return

        if len(python_files) > 15:
            ok = messagebox.askyesno(
                "Confirmation",
                (
                    f"The input contains {len(python_files)} .py files.\n"
                    "The test may take a long time or consume API credits/tokens.\n\n"
                    "Continue?"
                ),
            )

            if not ok:
                return

        os.makedirs(
            output_path,
            exist_ok=True,
        )

        self._cancel_event.clear()

        self._running_total = len(
            python_files
        )
        self._running_index = 0
        self._running_filename = ""

        self._set_running_state(True)

        self._progress.configure(
            mode="indeterminate"
        )
        self._progress.start(120)

        self._run_started_at = monotonic()

        self._status_var.set(
            f"Running: 0/{len(python_files)} (starting...)"
        )

        self._schedule_heartbeat()

        provider_kind = self._provider_type_var.get()

        self._append_output(
            f"Input Path: {input_path}\n"
        )
        self._append_output(
            f"Output Path: {output_path}\n"
        )
        self._append_output(
            f"Smell ID: {smell_id}\n"
        )
        self._append_output(
            f"Prompt mode: {mode.value}\n"
        )
        self._append_output(
            f"Provider type: {provider_kind}\n"
        )
        self._append_output(
            f"Provider: {provider_id}\n"
        )
        self._append_output(
            "Analyzing file(s)...\n"
        )

        run_prompt_text = prompt_text

        thread = threading.Thread(
            target=self._run_test_thread,
            args=(
                smell_id,
                mode,
                python_files,
                output_path,
                provider_id,
                run_prompt_text,
            ),
            daemon=True,
        )

        thread.start()

    def _on_cancel_clicked(
        self,
    ) -> None:
        if self._cancel_event.is_set():
            return

        self._cancel_event.set()

        self._append_output(
            "Cancellation requested: the test will stop after the current file.\n"
        )

    # ------------------------------------------------------------
    # Save prompt
    # ------------------------------------------------------------

    def _on_save_default_clicked(
        self,
    ) -> None:
        smell_id = self._current_smell_id

        if not smell_id:
            messagebox.showerror(
                "Error",
                "Select a smell.",
            )
            return

        if (
            self._mode_var.get()
            == PromptMode.DRAFT.value
        ):
            prompt_text = (
                self._get_current_prompt_text()
                .strip()
            )

            if not prompt_text:
                messagebox.showerror(
                    "Error",
                    "The temporary prompt is empty.",
                )
                return

            try:
                self.catalog_service.save_draft_prompt(
                    smell_id,
                    prompt_text,
                )
                self._draft_dirty = False
            except CatalogValidationError as exc:
                messagebox.showerror(
                    "Error",
                    str(exc),
                )
                return

        ok = messagebox.askyesno(
            "Confirmation",
            (
                "Save the temporary prompt as the default prompt?\n"
                "(The smell will become selectable for LLM detection.)"
            ),
        )

        if not ok:
            return

        try:
            self.catalog_service.promote_draft_to_default(
                smell_id
            )
        except Exception as exc:
            messagebox.showerror(
                "Error",
                str(exc),
            )
            return

        self._append_output(
            "Prompt saved as default.\n"
        )

        self._mode_var.set(
            PromptMode.DEFAULT.value
        )
        self._refresh_prompt_view()

    def _get_current_prompt_text(
        self,
    ) -> str:
        return self._prompt_text.get(
            "1.0",
            "end",
        )

    def _confirm_discard_unsaved_draft_if_needed(
        self,
        context: str,
    ) -> bool:
        if not self._draft_dirty:
            return True

        return messagebox.askyesno(
            "Unsaved changes",
            (
                "There are unsaved changes in the temporary prompt "
                "(you have not run the test yet).\n"
                f"If you continue, they will be lost ({context}).\n\n"
                "Continue?"
            ),
        )

    def _on_close(
        self,
    ) -> None:
        if self._draft_dirty:
            ok = messagebox.askyesno(
                "Exit",
                (
                    "There are unsaved changes in the temporary prompt.\n"
                    "Exit anyway?"
                ),
            )

            if not ok:
                return

        self.master.destroy()

    # ============================================================
    # BACKGROUND TEST
    # ============================================================

    def _run_test_thread(
        self,
        smell_id: str,
        mode: PromptMode,
        python_files: list[str],
        output_path: str,
        provider_id: str,
        prompt_text: str,
    ) -> None:
        try:
            catalog = self.catalog_service.load()

            smell = catalog.get_smell(smell_id)

            if mode == PromptMode.DRAFT:
                smell = replace(
                    smell,
                    draft_prompt=prompt_text.strip(),
                )
                catalog.upsert_smell(smell)

            provider_def = catalog.get_provider(
                provider_id
            )

            provider = LLMProviderFactory.create(
                provider_def
            )

            orchestrator = LLMOrchestrator(
                provider=provider,
                catalog=catalog,
            )

            all_findings = []
            prompts_sent = 0
            total = len(python_files)

            raw_records: list[dict[str, Any]] = []

            for idx, filename in enumerate(
                python_files,
                start=1,
            ):
                if self._cancel_event.is_set():
                    break

                with open(
                    filename,
                    "r",
                    encoding="utf-8",
                ) as file:
                    code = file.read()

                target = DetectionTarget(
                    filename=filename,
                    code=code,
                )

                def _ui_start_file(
                    i: int = idx,
                    n: int = total,
                    fn: str = filename,
                    chars: int = len(code),
                ) -> None:
                    self._running_index = i
                    self._running_total = n
                    self._running_filename = os.path.basename(fn)

                    self._append_output(
                        f"[{i}/{n}] Starting analysis: "
                        f"{os.path.basename(fn)} "
                        f"(chars: {chars})\n"
                    )

                self.master.after(
                    0,
                    _ui_start_file,
                )

                findings, stats, raw_by_file = (
                    orchestrator.detect_for_prompt_engineering_with_raw(
                        targets=[target],
                        smell_id=smell_id,
                        prompt_mode=mode,
                    )
                )

                all_findings.extend(findings)
                prompts_sent += stats.prompts_sent

                trace = (
                    orchestrator.last_traces[-1]
                    if orchestrator.last_traces
                    else None
                )

                result = (
                    orchestrator.last_results[-1]
                    if orchestrator.last_results
                    else None
                )

                raw_model_response = raw_by_file.get(
                    filename,
                    "",
                )

                raw_records.append(
                    {
                        "filename": filename,
                        "smell_id": smell_id,
                        "prompt_mode": mode.value,
                        "provider_id": provider_id,
                        "status": (
                            result.status.value
                            if result
                            else "unknown"
                        ),
                        "assessment": (
                            result.assessment
                            if result
                            else ""
                        ),
                        "error": (
                            result.error
                            if result
                            else None
                        ),
                        "raw_response": raw_model_response,
                        "raw_model_response": raw_model_response,
                        "raw_provider_response": (
                            trace.raw_provider_response
                            if trace
                            else None
                        ),
                        "native_reasoning": (
                            trace.native_reasoning
                            if trace
                            else ""
                        ),
                        "system_prompt": (
                            trace.system_prompt
                            if trace
                            else ""
                        ),
                        "user_prompt": (
                            trace.user_prompt
                            if trace
                            else ""
                        ),
                        "response_schema": (
                            trace.response_format
                            if trace
                            else {}
                        ),
                        "provider_name": (
                            trace.provider_name
                            if trace
                            else ""
                        ),
                        "model_name": (
                            trace.model_name
                            if trace
                            else None
                        ),
                        "provider_options": (
                            trace.provider_options
                            if trace
                            else {}
                        ),
                        "provider_response_format": (
                            trace.provider_response_format
                            if trace
                            else None
                        ),
                        "provider_think": (
                            trace.provider_think
                            if trace
                            else None
                        ),
                        "generation_metadata": (
                            trace.metadata
                            if trace
                            else {}
                        ),
                    }
                )

            valid_findings = [
                finding
                for finding in all_findings
                if getattr(finding, "line", -1) > 0
            ]

            invalid_response_count = sum(
                1
                for record in raw_records
                if record.get("status") == "invalid_response"
            )

            provider_error_count = sum(
                1
                for record in raw_records
                if record.get("status") == "provider_error"
            )

            parse_error_count = (
                invalid_response_count
                + provider_error_count
            )

            df = orchestrator.findings_to_dataframe(
                valid_findings
            )

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output_dir = os.path.join(
                output_path,
                "output",
            )

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            out_file = os.path.join(
                output_dir,
                f"prompt_engineering_{smell_id}_{timestamp}.csv",
            )

            df.to_csv(
                out_file,
                index=False,
            )

            csv_rows = int(len(df.index))

            try:
                csv_size = int(
                    os.path.getsize(out_file)
                )
            except Exception:
                csv_size = -1

            raw_file = os.path.join(
                output_dir,
                f"prompt_engineering_{smell_id}_{timestamp}_raw.jsonl",
            )

            with open(
                raw_file,
                "w",
                encoding="utf-8",
            ) as file:
                for record in raw_records:
                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )

            try:
                raw_size = int(
                    os.path.getsize(raw_file)
                )
            except Exception:
                raw_size = -1

            def _ui_done() -> None:
                self._stop_heartbeat()

                self._progress.stop()
                self._progress.configure(
                    mode="determinate",
                    maximum=max(1, total),
                    value=min(total, len(raw_records)),
                )

                self._append_output(
                    (
                        "Test completed. "
                        f"Prompts sent: {prompts_sent} | "
                        f"Targets: {total} | "
                        f"Findings: {len(all_findings)} "
                        f"(valid: {len(valid_findings)} | "
                        f"invalid responses: {invalid_response_count} | "
                        f"provider errors: {provider_error_count})\n"
                    )
                )

                self._append_output(
                    (
                        "Analysis completed. "
                        "Total code smells found: "
                        f"{len(valid_findings)}\n"
                    )
                )

                self._append_output(
                    f"Output folder: {output_dir}\n"
                )

                self._append_output(
                    f"Results saved to: {out_file}\n"
                )

                self._append_output(
                    f"Raw responses saved to: {raw_file}\n"
                )

                self._append_output(
                    f"CSV rows: {csv_rows} | CSV bytes: {csv_size}\n"
                )

                self._append_output(
                    f"Raw bytes: {raw_size}\n"
                )

                assessments = [
                    record
                    for record in raw_records
                    if (
                        record.get("assessment")
                        or ""
                    ).strip()
                ]

                if assessments:
                    self._append_output(
                        "\nDeveloper assessment preview:\n"
                    )

                    for record in assessments[:10]:
                        assessment = (
                            record.get("assessment")
                            or ""
                        ).strip()

                        status = record.get(
                            "status",
                            "unknown",
                        )

                        self._append_output(
                            f"- {os.path.basename(record['filename'])} "
                            f"[{status}]: {assessment}\n"
                        )

                    if len(assessments) > 10:
                        self._append_output(
                            f"... {len(assessments) - 10} more assessments in the raw JSONL.\n"
                        )

                if df.empty:
                    if parse_error_count > 0:
                        self._append_output(
                            "No valid findings extracted (invalid response/provider error).\n"
                        )
                    else:
                        self._append_output(
                            "No findings returned by the LLM.\n"
                        )
                else:
                    self._append_output(
                        "Valid findings were generated and saved to CSV.\n"
                    )

                    self._append_output(
                        "\nReasoning preview:\n"
                    )

                    for finding in valid_findings[:20]:
                        rationale = (
                            getattr(
                                finding,
                                "reasoning",
                                "",
                            )
                            or ""
                        ).strip()

                        self._append_output(
                            f"- {os.path.basename(finding.filename)}:"
                            f"{finding.line} "
                            f"[{finding.smell_name}] "
                            f"{rationale or '(reasoning not returned)'}\n"
                        )

                    if len(valid_findings) > 20:
                        self._append_output(
                            f"... {len(valid_findings) - 20} more findings in the CSV.\n"
                        )

                if invalid_response_count > 0:
                    self._append_output(
                        "Note: at least one response did not match the expected schema. "
                        "See the *_raw.jsonl file for the complete response.\n"
                    )

                if provider_error_count > 0:
                    self._append_output(
                        "Note: at least one provider call produced an error. "
                        "See the *_raw.jsonl file for details.\n"
                    )

                if self._cancel_event.is_set():
                    self._append_output(
                        "Test interrupted by user request.\n"
                    )

                self._append_output(
                    "--- End of test ---\n\n"
                )

                self._status_var.set("Idle")
                self._set_running_state(False)
                self._sync_test_button_state()

            self.master.after(
                0,
                _ui_done,
            )

        except Exception as exc:
            import traceback

            err_text = (
                f"{type(exc).__name__}: {exc}"
            )

            tb_text = traceback.format_exc()

            def _ui_err(
                err: str = err_text,
                tb: str = tb_text,
            ) -> None:
                self._stop_heartbeat()

                self._progress.stop()
                self._progress.configure(
                    mode="determinate",
                    maximum=1,
                    value=0,
                )

                self._append_output(
                    f"Error during test: {err}\n"
                )

                self._append_output(
                    tb + "\n"
                )

                self._status_var.set("Idle")
                self._set_running_state(False)
                self._sync_test_button_state()

            self.master.after(
                0,
                _ui_err,
            )

    # ============================================================
    # HEARTBEAT / RUNNING STATE
    # ============================================================

    def _schedule_heartbeat(
        self,
    ) -> None:
        self._stop_heartbeat()

        def _tick() -> None:
            if self._run_started_at is None:
                return

            elapsed_s = int(
                monotonic()
                - self._run_started_at
            )

            i = self._running_index
            n = self._running_total
            filename = self._running_filename

            cancel = (
                " (cancelling...)"
                if self._cancel_event.is_set()
                else ""
            )

            if filename:
                self._status_var.set(
                    f"Running: {i}/{n} "
                    f"(file: {filename}) "
                    f"elapsed: {elapsed_s}s{cancel}"
                )
            else:
                self._status_var.set(
                    f"Running: {i}/{n} "
                    f"elapsed: {elapsed_s}s{cancel}"
                )

            self._heartbeat_after_id = self.master.after(
                1000,
                _tick,
            )

        self._heartbeat_after_id = self.master.after(
            250,
            _tick,
        )

    def _stop_heartbeat(
        self,
    ) -> None:
        if self._heartbeat_after_id is not None:
            try:
                self.master.after_cancel(
                    self._heartbeat_after_id
                )
            except Exception:
                pass

        self._heartbeat_after_id = None
        self._run_started_at = None

    def _set_running_state(
        self,
        running: bool,
    ) -> None:
        """
        Disable controls whose values must remain stable
        while an LLM experiment is running.
        """

        action_state = (
            "disabled"
            if (running or self._ui_disabled_no_smells)
            else "normal"
        )

        add_smell_state = (
            "disabled"
            if running
            else "normal"
        )

        self._test_btn.configure(
            state=action_state
        )

        self._save_default_btn.configure(
            state=action_state
        )

        self._smell_combo.configure(
            state=(
                "disabled"
                if running
                else (
                    "disabled"
                    if self._ui_disabled_no_smells
                    else "readonly"
                )
            )
        )

        self._add_smell_btn.configure(
            state=add_smell_state
        )

        self._cancel_btn.configure(
            state="normal" if running else "disabled"
        )

        provider_type_state = (
            "disabled"
            if (running or self._ui_disabled_no_smells)
            else "normal"
        )

        self._local_provider_radio.configure(
            state=provider_type_state
        )

        self._api_provider_radio.configure(
            state=provider_type_state
        )

        provider_values = self._provider_combo["values"]

        self._provider_combo.configure(
            state=(
                "disabled"
                if running
                else (
                    "disabled"
                    if (
                        self._ui_disabled_no_smells
                        or not provider_values
                    )
                    else "readonly"
                )
            )
        )

        self._manage_provider_btn.configure(
            state=(
                "disabled"
                if (running or self._ui_disabled_no_smells)
                else "normal"
            )
        )

        self._choose_input_btn.configure(
            state="disabled" if running else "normal"
        )

        self._choose_output_btn.configure(
            state="disabled" if running else "normal"
        )

        if running:
            self._prompt_text.configure(
                state="disabled"
            )
        else:
            self._refresh_prompt_view()

    # ============================================================
    # OUTPUT HELPERS
    # ============================================================

    def _append_output(
        self,
        text: str,
    ) -> None:
        self._output_text.configure(
            state="normal"
        )

        self._output_text.insert(
            "end",
            text,
        )

        self._output_text.update_idletasks()
        self._output_text.see("end")

        self._output_text.configure(
            state="disabled"
        )


def main() -> None:
    root = tk.Tk()
    PromptEngineeringGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()