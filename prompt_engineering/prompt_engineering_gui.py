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

from llm_detection.catalog_service import CatalogValidationError, LLMCatalogService
from llm_detection.orchestrator import LLMOrchestrator
from llm_detection.providers import LocalLLMProvider
from llm_detection.types import DetectionTarget, PromptMode, ProviderKind
from utils.file_utils import FileUtils
from gui.manage_code_smells_gui import AddSmellDialog
from gui.ollama_model_manager_gui import OllamaModelManagerDialog
from gui.folder_utils import open_folder

class PromptEngineeringGUI:
    def __init__(self, master: tk.Tk, catalog_service: Optional[LLMCatalogService] = None):
        self.master = master
        self.catalog_service = catalog_service or LLMCatalogService()

        # --- state ---
        self._current_smell_id: Optional[str] = None
        self._draft_dirty: bool = False
        self._ui_disabled_no_smells: bool = False

        self._mode_var = tk.StringVar(value=PromptMode.DRAFT.value)
        self._smell_display_to_id: dict[str, str] = {}

        self._local_provider_display_to_id: dict[str, str] = {}
        self._selected_local_provider_id: Optional[str] = None

        self._input_path_value: str = ""
        self._output_path_value: str = ""

        self._cancel_event = threading.Event()
        self._run_started_at: Optional[float] = None
        self._heartbeat_after_id: Optional[str] = None
        self._running_total: int = 0
        self._running_index: int = 0
        self._running_filename: str = ""

        # --- ui ---
        self._build_ui()
        self._load_smells_into_dropdown()
        self._load_local_providers_into_dropdown()
        self._sync_test_button_state()

        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.master.title("Prompt Engineering")
        # Slightly taller default to keep log fully visible without manual resize.
        self.master.geometry("900x700")

        style = ttk.Style(self.master)
        # Keep default theme/colors; only reduce visual "heaviness".
        try:
            style.configure("Calm.Horizontal.TProgressbar", thickness=8)
        except Exception:
            pass

        top = ttk.Frame(self.master)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.master.grid_columnconfigure(0, weight=1)
        # Make rows responsive: prompt + log should share vertical space.
        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=0)
        self.master.grid_rowconfigure(3, weight=0)
        self.master.grid_rowconfigure(4, weight=3)

        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=0)
        top.grid_columnconfigure(2, weight=0)

        ttk.Label(top, text="Code smell:").grid(row=0, column=0, sticky="e", padx=(0, 6))

        self._smell_combo = ttk.Combobox(top, state="readonly", width=50)
        self._smell_combo.grid(row=0, column=1, sticky="w")
        self._smell_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_smell_selected())

        self._add_smell_btn = ttk.Button(top, text="+", width=3, command=self._on_add_smell)
        self._add_smell_btn.grid(row=0, column=2, sticky="w", padx=(6, 0))

        # Prompt box + mode
        mode_frame = ttk.LabelFrame(self.master, text="Prompt")
        mode_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        mode_frame.grid_columnconfigure(0, weight=1)
        mode_frame.grid_rowconfigure(2, weight=1)

        best = ttk.Label(
            mode_frame,
            text=(
                "Best practices: Include ONLY the definition and smell-specific rules in the prompt. "
                "Do not include JSON schemas or output contracts; the orchestrator already enforces them."
            ),
            foreground="#444",
            wraplength=820,
        )
        best.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))

        radio_row = ttk.Frame(mode_frame)
        radio_row.grid(row=1, column=0, sticky="w", padx=10, pady=(8, 4))

        self._draft_radio = ttk.Radiobutton(
            radio_row,
            text="Draft (editable)",
            value=PromptMode.DRAFT.value,
            variable=self._mode_var,
            command=self._on_prompt_mode_changed,
        )
        self._draft_radio.grid(row=0, column=0, sticky="w", padx=(0, 12))

        self._default_radio = ttk.Radiobutton(
            radio_row,
            text="Default (read-only)",
            value=PromptMode.DEFAULT.value,
            variable=self._mode_var,
            command=self._on_prompt_mode_changed,
        )
        self._default_radio.grid(row=0, column=1, sticky="w")

        self._prompt_text = ScrolledText(mode_frame, height=8, wrap="word")
        self._prompt_text.grid(row=2, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self._prompt_text.bind("<KeyRelease>", self._on_prompt_edited)

        # Paths / Provider
        paths = ttk.LabelFrame(self.master, text="Paths / Provider")
        paths.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        paths.grid_columnconfigure(1, weight=1)
        paths.grid_columnconfigure(2, weight=0)

        ttk.Label(paths, text="Input path:").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self._input_path_label = ttk.Label(paths, text="No path selected")
        self._input_path_label.grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(paths, text="Choose Input Folder", command=self._choose_input_path).grid(
            row=0, column=2, sticky="e", padx=10, pady=6
        )

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

        output_buttons_frame = ttk.Frame(paths)
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
        self._open_output_btn.pack(side="left")

        ttk.Label(paths, text="Local LLM:").grid(row=2, column=0, sticky="w", padx=10, pady=(6, 10))
        self._local_provider_combo = ttk.Combobox(paths, state="readonly", width=50)
        self._local_provider_combo.grid(row=2, column=1, sticky="w", pady=(6, 10))
        self._local_provider_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_local_provider_selected())
        self._manage_models_btn = ttk.Button(
            paths,
            text="Browse / Download Models",
            command=self._open_model_manager,
        )
        self._manage_models_btn.grid(row=2, column=2, sticky="e", padx=10, pady=(6, 10))

        # Actions
        actions = ttk.Frame(self.master)
        actions.grid(row=3, column=0, sticky="ew", padx=10)
        actions.grid_columnconfigure(0, weight=1)

        self._test_btn = ttk.Button(actions, text="Test with Local LLM", command=self._on_test_clicked)
        self._test_btn.grid(row=0, column=0, sticky="w")

        self._cancel_btn = ttk.Button(actions, text="Cancel", command=self._on_cancel_clicked)
        self._cancel_btn.grid(row=0, column=0, sticky="w", padx=(160, 0))
        self._cancel_btn.configure(state="disabled")

        self._save_default_btn = ttk.Button(
            actions,
            text="Save Draft as Default",
            command=self._on_save_default_clicked,
        )
        self._save_default_btn.grid(row=0, column=1, sticky="w", padx=(10, 0))

        self._exit_btn = ttk.Button(actions, text="Exit", command=self._on_close)
        self._exit_btn.grid(row=0, column=2, sticky="e")

        self._status_var = tk.StringVar(value="Idle")
        self._status_label = ttk.Label(actions, textvariable=self._status_var)
        self._status_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self._progress = ttk.Progressbar(actions, mode="determinate", style="Calm.Horizontal.TProgressbar")
        self._progress.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        # Output
        out = ttk.LabelFrame(self.master, text="Results / Log")
        out.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)
        out.grid_columnconfigure(0, weight=1)
        out.grid_rowconfigure(0, weight=1)

        self._output_text = ScrolledText(out, height=12, wrap="word", state="disabled")
        self._output_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    # ---------------- Data loading ----------------

    def _load_smells_into_dropdown(self) -> None:
        catalog = self.catalog_service.load()

        self._smell_display_to_id.clear()
        values: list[str] = []

        for smell in sorted(catalog.smells, key=lambda s: s.display_name.lower()):
            label = f"{smell.display_name}  ({smell.smell_id})"
            self._smell_display_to_id[label] = smell.smell_id
            values.append(label)

        self._smell_combo["values"] = values

        if not values:
            self._ui_disabled_no_smells = True
            self._disable_all_controls_no_smells()
            self._append_output(
                "The code smell catalog is empty. Add a code smell using '+' to continue.\n"
            )
            return

        self._ui_disabled_no_smells = False
        # If the catalog was empty at startup, controls may still be disabled.
        # Re-enable the minimal set now that we have at least one smell.
        self._smell_combo.configure(state="readonly")
        self._draft_radio.configure(state="normal")
        self._default_radio.configure(state="normal")
        self._add_smell_btn.configure(state="normal")
        self._smell_combo.current(0)
        self._on_smell_selected()

    def _load_local_providers_into_dropdown(self) -> None:
        catalog = self.catalog_service.load()

        self._local_provider_display_to_id.clear()
        values: list[str] = []

        for provider in catalog.providers:
            if provider.kind != ProviderKind.LOCAL:
                continue
            model = provider.config.get("model_name")
            model_part = f" | model: {model}" if model else ""
            label = f"{provider.display_name}  ({provider.provider_id}){model_part}"
            self._local_provider_display_to_id[label] = provider.provider_id
            values.append(label)

        self._local_provider_combo["values"] = values

        if self._ui_disabled_no_smells:
            self._local_provider_combo.set("")
            self._local_provider_combo.configure(state="disabled")
            self._selected_local_provider_id = None
            self._sync_test_button_state()
            return

        if not values:
            self._local_provider_combo.set("")
            self._local_provider_combo.configure(state="disabled")
            self._selected_local_provider_id = None
            self._append_output(
                "No local models are configured. "
                "Use 'Browse / Download Models' to select, download, or register an Ollama model.\n"
            )
            self._sync_test_button_state()
            return

        self._local_provider_combo.configure(state="readonly")
        self._local_provider_combo.current(0)
        self._on_local_provider_selected()

    def _open_model_manager(self) -> None:
        OllamaModelManagerDialog(
            self.master,
            catalog_service=self.catalog_service,
            on_catalog_changed=self._load_local_providers_into_dropdown,
            on_model_ready=self._select_local_provider_by_id,
        )

    def _select_local_provider_by_id(self, provider_id: str) -> None:
        """Select the model that was installed/registered from Model Hub."""
        self._load_local_providers_into_dropdown()
        for display, pid in self._local_provider_display_to_id.items():
            if pid == provider_id:
                self._local_provider_combo.set(display)
                self._on_local_provider_selected()
                break

    def _on_local_provider_selected(self) -> None:
        selected = self._local_provider_combo.get()
        self._selected_local_provider_id = self._local_provider_display_to_id.get(selected)
        self._sync_test_button_state()

    def _sync_test_button_state(self) -> None:
        """Enable/disable the Test button based on current prerequisites."""
        if self._ui_disabled_no_smells:
            self._test_btn.configure(state="disabled")
            return
        if not self._current_smell_id:
            self._test_btn.configure(state="disabled")
            return
        if not (self._selected_local_provider_id or "").strip():
            self._test_btn.configure(state="disabled")
            return
        # If we are running, _set_running_state() will override this anyway.
        self._test_btn.configure(state="normal")

    def _disable_all_controls_no_smells(self) -> None:
        self._smell_combo.set("")
        self._smell_combo.configure(state="disabled")
        # Allow bootstrap: user can add the first smell directly from Prompt Engineering.
        self._add_smell_btn.configure(state="normal")
        self._draft_radio.configure(state="disabled")
        self._default_radio.configure(state="disabled")
        self._prompt_text.configure(state="disabled")
        self._test_btn.configure(state="disabled")
        self._save_default_btn.configure(state="disabled")
        self._local_provider_combo.configure(state="disabled")

    # ---------------- UI events ----------------

    def _on_add_smell(self) -> None:
        def on_success(new_smell_id: str) -> None:
            self._load_smells_into_dropdown()
            # If we started with an empty catalog, providers combo may have been disabled.
            self._load_local_providers_into_dropdown()
            # Select the new smell automatically
            for display, sid in self._smell_display_to_id.items():
                if sid == new_smell_id:
                    self._smell_combo.set(display)
                    self._on_smell_selected()
                    break

        AddSmellDialog(self.master, self.catalog_service, on_success)

    def _on_smell_selected(self) -> None:
        if not self._confirm_discard_unsaved_draft_if_needed(context="change the selected code smell"):
            self._restore_combo_to_current_smell()
            return

        selected = self._smell_combo.get()
        self._current_smell_id = self._smell_display_to_id.get(selected)
        self._draft_dirty = False
        self._refresh_prompt_view()
        self._sync_test_button_state()

    def _restore_combo_to_current_smell(self) -> None:
        if not self._current_smell_id:
            return
        for i, display in enumerate(self._smell_combo["values"]):
            if self._smell_display_to_id.get(display) == self._current_smell_id:
                self._smell_combo.current(i)
                return

    def _on_prompt_mode_changed(self) -> None:
        if self._mode_var.get() == PromptMode.DEFAULT.value:
            if not self._confirm_discard_unsaved_draft_if_needed(context="switch to the default prompt"):
                self._mode_var.set(PromptMode.DRAFT.value)
                return
        self._draft_dirty = False
        self._refresh_prompt_view()

    def _refresh_prompt_view(self) -> None:
        if not self._current_smell_id:
            self._set_prompt_text("", editable=False)
            return

        mode = PromptMode(self._mode_var.get())
        catalog = self.catalog_service.load()
        smell = catalog.get_smell(self._current_smell_id)

        if mode == PromptMode.DRAFT:
            self._set_prompt_text(smell.draft_prompt or "", editable=True)
        else:
            self._set_prompt_text(smell.default_prompt or "", editable=False)

    def _set_prompt_text(self, text: str, editable: bool) -> None:
        self._prompt_text.configure(state="normal")
        self._prompt_text.delete("1.0", "end")
        self._prompt_text.insert("1.0", text)
        self._prompt_text.configure(state="normal" if editable else "disabled")

    def _on_prompt_edited(self, _event) -> None:
        if self._mode_var.get() != PromptMode.DRAFT.value:
            return
        self._draft_dirty = True

    def _choose_input_path(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self._input_path_value = path
            self._input_path_label.configure(text=path)

    def _choose_output_path(self) -> None:
        path = filedialog.askdirectory()

        if path:
            self._output_path_value = path
            self._output_path_label.configure(text=path)
            self._open_output_btn.configure(state="normal")

    def _open_output_folder(self) -> None:
        """
        Open the Prompt Engineering results directory.

        Prompt Engineering stores generated CSV/JSONL files inside the
        'output' subdirectory. If that directory does not exist yet,
        the selected base output directory is opened instead.
        """
        base_path = (self._output_path_value or "").strip()

        if not base_path:
            messagebox.showwarning(
                "Output folder",
                "Select an output folder first.",
            )
            return

        results_path = os.path.join(base_path, "output")

        if os.path.isdir(results_path):
            path_to_open = results_path
        else:
            path_to_open = base_path

        try:
            open_folder(path_to_open)
        except Exception as e:
            messagebox.showerror(
                "Output folder",
                f"Could not open the output folder:\n{e}",
            )

    def _on_test_clicked(self) -> None:
        smell_id = self._current_smell_id
        if not smell_id:
            messagebox.showerror("Error", "Select a code smell.")
            return

        provider_id = (self._selected_local_provider_id or "").strip()
        if not provider_id:
            messagebox.showerror(
                "Error",
                "No local LLM provider is selected. Configure a local provider in the catalog.",
            )
            return

        mode = PromptMode(self._mode_var.get())

        input_path = (self._input_path_value or "").strip()
        output_path = (self._output_path_value or "").strip()

        try:
            self.catalog_service.validate_prompt_engineering_input_path(input_path)
        except CatalogValidationError as e:
            messagebox.showerror("Invalid Input Path", str(e))
            return

        if not output_path:
            messagebox.showerror("Error", "Select an output folder.")
            return

        prompt_text = self._get_current_prompt_text().strip()
        if not prompt_text:
            messagebox.showerror("Error", "The prompt is empty.")
            return

        # UC02 step 10: persist draft when testing
        if mode == PromptMode.DRAFT:
            try:
                self.catalog_service.save_draft_prompt(smell_id, prompt_text)
                self._draft_dirty = False
            except CatalogValidationError as e:
                messagebox.showerror("Error", str(e))
                return

        python_files = FileUtils.get_python_files(input_path)
        if not python_files:
            messagebox.showerror("Error", "The input folder contains no Python (.py) files.")
            return

        if len(python_files) > 15:
            ok = messagebox.askyesno(
                "Confirmation",
                f"The input folder contains {len(python_files)} .py files.\n"
                "The test may take a long time.\n\nContinue?",
            )
            if not ok:
                return

        os.makedirs(output_path, exist_ok=True)

        self._cancel_event.clear()
        self._running_total = len(python_files)
        self._running_index = 0
        self._running_filename = ""

        self._set_running_state(True)
        # Indeterminate is best for long single-file LLM calls; make it calmer by slowing it down.
        self._progress.configure(mode="indeterminate")
        self._progress.start(120)
        self._run_started_at = monotonic()
        self._status_var.set(f"Running: 0/{len(python_files)}  (starting...)")
        self._schedule_heartbeat()

        self._append_output(f"Input Path: {input_path}\n")
        self._append_output(f"Output Path: {output_path}\n")
        self._append_output(f"Smell ID: {smell_id}\n")
        self._append_output(f"Prompt mode: {mode.value}\n")
        self._append_output(f"Local provider: {provider_id}\n")
        self._append_output("Analyzing file(s)...\n")

        # Capture all Tk values on the UI thread. Tk widgets must not be read from
        # the background worker.
        run_prompt_text = prompt_text
        t = threading.Thread(
            target=self._run_test_thread,
            args=(smell_id, mode, python_files, output_path, provider_id, run_prompt_text),
            daemon=True,
        )
        t.start()

    def _on_cancel_clicked(self) -> None:
        if self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self._append_output("Cancellation requested. The test will stop after the current file.\n")

    def _on_save_default_clicked(self) -> None:
        smell_id = self._current_smell_id
        if not smell_id:
            messagebox.showerror("Error", "Select a code smell.")
            return

        if self._mode_var.get() == PromptMode.DRAFT.value:
            prompt_text = self._get_current_prompt_text().strip()
            if not prompt_text:
                messagebox.showerror("Error", "The draft prompt is empty.")
                return
            try:
                self.catalog_service.save_draft_prompt(smell_id, prompt_text)
                self._draft_dirty = False
            except CatalogValidationError as e:
                messagebox.showerror("Error", str(e))
                return

        ok = messagebox.askyesno(
            "Confirmation",
            "Save the draft prompt as the default prompt?\n"
            "(This code smell will become available for LLM-based detection.)",
        )
        if not ok:
            return

        try:
            self.catalog_service.promote_draft_to_default(smell_id)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self._append_output("Prompt saved as default.\n")
        self._mode_var.set(PromptMode.DEFAULT.value)
        self._refresh_prompt_view()

    def _get_current_prompt_text(self) -> str:
        return self._prompt_text.get("1.0", "end")

    def _confirm_discard_unsaved_draft_if_needed(self, context: str) -> bool:
        if not self._draft_dirty:
            return True
        return messagebox.askyesno(
            "Unsaved Changes",
            "The draft prompt has unsaved changes (the test has not been run).\n"
            f"If you continue to {context}, these changes will be lost.\n\nContinue?",
        )

    def _on_close(self) -> None:
        if self._draft_dirty:
            ok = messagebox.askyesno(
                "Exit",
                "The draft prompt has unsaved changes.\n"
                "Exit anyway?",
            )
            if not ok:
                return
        self.master.destroy()

    # ---------------- Background test ----------------

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

            # The prompt text was captured on the UI thread before starting this
            # worker, avoiding unsafe Tk access from a background thread.
            if mode == PromptMode.DRAFT:
                smell = replace(smell, draft_prompt=prompt_text.strip())
                catalog.upsert_smell(smell)

            provider = self._build_local_provider_by_id(catalog, provider_id)
            orchestrator = LLMOrchestrator(provider=provider, catalog=catalog)

            all_findings = []
            prompts_sent = 0
            total = len(python_files)
            raw_records: list[dict[str, Any]] = []

            for idx, filename in enumerate(python_files, start=1):
                if self._cancel_event.is_set():
                    break

                with open(filename, "r", encoding="utf-8") as f:
                    code = f.read()

                target = DetectionTarget(filename=filename, code=code)

                def _ui_start_file(i: int = idx, n: int = total, fn: str = filename, chars: int = len(code)) -> None:
                    self._running_index = i
                    self._running_total = n
                    self._running_filename = os.path.basename(fn)
                    self._append_output(f"[{i}/{n}] Starting analysis: {os.path.basename(fn)} (chars: {chars})\n")

                self.master.after(0, _ui_start_file)

                findings, stats, raw_by_file = orchestrator.detect_for_prompt_engineering_with_raw(
                    targets=[target],
                    smell_id=smell_id,
                    prompt_mode=mode,
                )
                all_findings.extend(findings)
                prompts_sent += stats.prompts_sent

                trace = orchestrator.last_traces[-1] if orchestrator.last_traces else None
                raw_records.append(
                    {
                        "filename": filename,
                        "smell_id": smell_id,
                        "prompt_mode": mode.value,
                        "provider_id": provider_id,
                        "raw_response": raw_by_file.get(filename, ""),
                        "native_reasoning": trace.native_reasoning if trace else "",
                        "generation_metadata": trace.metadata if trace else {},
                    }
                )

            # more consistent than "!= -1"
            valid_findings = [f for f in all_findings if getattr(f, "line", -1) > 0]
            parse_error_count = len(all_findings) - len(valid_findings)

            df = orchestrator.findings_to_dataframe(valid_findings)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_path, "output")
            os.makedirs(output_dir, exist_ok=True)

            out_file = os.path.join(output_dir, f"prompt_engineering_{smell_id}_{timestamp}.csv")
            df.to_csv(out_file, index=False)
            csv_rows = int(len(df.index))

            try:
                csv_size = int(os.path.getsize(out_file))
            except Exception:
                csv_size = -1

            raw_file = os.path.join(output_dir, f"prompt_engineering_{smell_id}_{timestamp}_raw.jsonl")
            with open(raw_file, "w", encoding="utf-8") as f:
                for rec in raw_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            try:
                raw_size = int(os.path.getsize(raw_file))
            except Exception:
                raw_size = -1

            def _ui_done() -> None:
                self._stop_heartbeat()
                self._progress.stop()
                self._progress.configure(mode="determinate", maximum=max(1, total), value=min(total, total))

                self._append_output(
                    f"Test completed. Prompts sent: {prompts_sent} | "
                    f"Targets: {total} | Findings: {len(all_findings)} "
                    f"(valid: {len(valid_findings)} | parse_errors: {parse_error_count})\n"
                )
                self._append_output(f"Analysis completed. Total code smells found: {len(valid_findings)}\n")
                self._append_output(f"Output folder: {output_dir}\n")
                self._append_output(f"Results saved to: {out_file}\n")
                self._append_output(f"Raw responses saved to: {raw_file}\n")
                self._append_output(f"CSV rows: {csv_rows} | CSV bytes: {csv_size}\n")
                self._append_output(f"Raw bytes: {raw_size}\n")

                if df.empty:
                    if parse_error_count > 0:
                        self._append_output("No valid findings extracted (only parsing/validation errors).\n")
                    else:
                        self._append_output("No findings returned by the LLM.\n")
                else:
                    self._append_output("Valid findings generated and saved to CSV.\n")
                    self._append_output("\nReasoning preview:\n")
                    for finding in valid_findings[:20]:
                        rationale = (getattr(finding, "reasoning", "") or "").strip()
                        self._append_output(
                            f"- {os.path.basename(finding.filename)}:{finding.line} "
                            f"[{finding.smell_name}] {rationale or '(no reasoning provided)'}\n"
                        )
                    if len(valid_findings) > 20:
                        self._append_output(
                            f"... more {len(valid_findings) - 20} findings in the CSV.\n"
                        )

                if any(getattr(f, "line", None) <= 0 for f in all_findings):
                    self._append_output(
                        "Note: At least one response contained invalid or truncated JSON. "
                        "See *_raw.jsonl for the raw response.\n"
                    )

                self._append_output("--- End of test ---\n\n")
                self._status_var.set("Idle")
                self._set_running_state(False)
                self._sync_test_button_state()

            self.master.after(0, _ui_done)

        except Exception as e:
            import traceback

            err_text = f"{type(e).__name__}: {e}"
            tb_text = traceback.format_exc()

            def _ui_err(err: str = err_text, tb: str = tb_text) -> None:
                self._stop_heartbeat()
                self._progress.stop()
                self._progress.configure(mode="determinate", maximum=1, value=0)
                self._append_output(f"Error during test: {err}\n")
                self._append_output(tb + "\n")
                self._status_var.set("Idle")
                self._set_running_state(False)
                self._sync_test_button_state()

            self.master.after(0, _ui_err)

    # ---------------- Heartbeat / running state ----------------

    def _schedule_heartbeat(self) -> None:
        self._stop_heartbeat()

        def _tick() -> None:
            if self._run_started_at is None:
                return
            elapsed_s = int(monotonic() - self._run_started_at)
            i = self._running_index
            n = self._running_total
            fn = self._running_filename
            cancel = " (cancelling...)" if self._cancel_event.is_set() else ""
            if fn:
                self._status_var.set(f"Running: {i}/{n}  (file: {fn})  elapsed: {elapsed_s}s{cancel}")
            else:
                self._status_var.set(f"Running: {i}/{n}  elapsed: {elapsed_s}s{cancel}")
            self._heartbeat_after_id = self.master.after(1000, _tick)

        self._heartbeat_after_id = self.master.after(250, _tick)

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_after_id is not None:
            try:
                self.master.after_cancel(self._heartbeat_after_id)
            except Exception:
                pass
        self._heartbeat_after_id = None
        self._run_started_at = None

    def _set_running_state(self, running: bool) -> None:
        state = "disabled" if running else "normal"

        # Keep things consistent even if UI is disabled due to empty smell catalog.
        # NOTE: the add-smell (+) button stays enabled when not running.
        action_state = "disabled" if (running or self._ui_disabled_no_smells) else "normal"
        add_smell_state = "disabled" if running else "normal"

        self._test_btn.configure(state=action_state)
        self._save_default_btn.configure(state=action_state)
        self._smell_combo.configure(state="disabled" if running else ("disabled" if self._ui_disabled_no_smells else "readonly"))
        self._add_smell_btn.configure(state=add_smell_state)
        self._cancel_btn.configure(state="normal" if running else "disabled")
        self._local_provider_combo.configure(state="disabled" if running else ("disabled" if self._ui_disabled_no_smells else "readonly"))
        self._manage_models_btn.configure(state="disabled" if running else "normal")

        if running:
            self._prompt_text.configure(state="disabled")
        else:
            self._refresh_prompt_view()

    # ---------------- Provider builder ----------------

    @staticmethod
    def _build_local_provider_by_id(catalog, provider_id: str) -> LocalLLMProvider:
        local = None
        for p in catalog.providers:
            if p.provider_id == provider_id and p.kind == ProviderKind.LOCAL:
                local = p
                break
        if local is None:
            raise RuntimeError(f"Local provider not found or not local: provider_id='{provider_id}'")

        model_name = str(local.config.get("model_name") or "qwen2.5-coder:7b")
        host = local.config.get("host") or local.config.get("base_url")
        host = str(host) if host else None
        options_val = local.config.get("options")
        options = dict(options_val) if isinstance(options_val, dict) else None
        response_format = local.config.get("format") or local.config.get("response_format")
        think = local.config.get("think")

        return LocalLLMProvider(
            model_name=model_name,
            host=host,
            options=options,
            response_format=response_format,
            think=think,
        )

    # ---------------- Output helpers ----------------

    def _append_output(self, text: str) -> None:
        self._output_text.configure(state="normal")
        self._output_text.insert("end", text)
        # Ensure geometry/layout is up-to-date before scrolling.
        self._output_text.update_idletasks()
        self._output_text.see("end")
        self._output_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    PromptEngineeringGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
