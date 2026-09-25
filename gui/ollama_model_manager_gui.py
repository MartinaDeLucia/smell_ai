from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from typing import Callable, Optional

from llm_detection.catalog_service import LLMCatalogService
from llm_detection.model_catalog import OllamaCatalogModel, OllamaCuratedCatalog, OllamaModelVariant
from llm_detection.providers import OllamaModelManager


def _format_size(size) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.1f} {units[idx]}"


class OllamaModelManagerDialog(tk.Toplevel):
    """Browsable Ollama Model Hub for CodeSmile.

    The local Ollama API is used for installed-model discovery and pull/delete
    operations. The "Explore" tab is backed by CodeSmile's curated catalog so a
    user does not need to know Ollama model tags in advance. The "Advanced" tab
    still accepts any valid Ollama tag.
    """

    FILTERS = (
        "All",
        "Recommended",
        "Coding",
        "Reasoning",
        "Lightweight",
        "Agentic",
        "General",
        "Legacy",
    )

    def __init__(
        self,
        parent,
        catalog_service: Optional[LLMCatalogService] = None,
        on_catalog_changed: Optional[Callable[[], None]] = None,
        on_model_ready: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(parent)
        self.catalog_service = catalog_service or LLMCatalogService()
        self.on_catalog_changed = on_catalog_changed
        self.on_model_ready = on_model_ready
        self.curated_catalog = OllamaCuratedCatalog()

        self.title("CodeSmile · Model Hub")
        self.geometry("1080x720")
        self.minsize(940, 620)
        self.transient(parent)

        self._host_var = tk.StringVar(value="http://localhost:11434")
        self._search_var = tk.StringVar()
        self._filter_var = tk.StringVar(value="All")
        self._variant_var = tk.StringVar()
        self._advanced_model_var = tk.StringVar()
        self._use_native_thinking_var = tk.BooleanVar(value=True)
        self._status_var = tk.StringVar(value="Connecting to Ollama…")
        self._connection_var = tk.StringVar(value="● Checking Ollama")
        self._catalog_meta_var = tk.StringVar(
            value=f"Curated catalog · verified {self.curated_catalog.verified_on}"
        )

        self._busy = False
        self._event_queue: queue.Queue[tuple] = queue.Queue()
        self._installed_models: list[dict] = []
        self._installed_tags: set[str] = set()
        self._visible_models: list[OllamaCatalogModel] = []
        self._selected_model: Optional[OllamaCatalogModel] = None

        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._search_var.trace_add("write", lambda *_: self._refresh_explore_models())
        self._filter_var.trace_add("write", lambda *_: self._refresh_explore_models())

        # Polling is scheduled from the Tk main thread. Background workers only
        # enqueue data and never touch widgets directly.
        self.after(50, self._drain_worker_events)
        self.after(80, self._refresh_installed_models)
        self.after(100, self._refresh_explore_models)

    # ------------------------------------------------------------------
    # Visual setup
    # ------------------------------------------------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass

        style.configure("HubTitle.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("HubSubtitle.TLabel", font=("Segoe UI", 10), foreground="#5b6472")
        style.configure("HubSection.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("HubModel.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("HubMeta.TLabel", font=("Segoe UI", 9), foreground="#687386")
        style.configure("HubCard.TFrame", relief="solid", borderwidth=1)
        style.configure("HubPrimary.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.configure("Hub.TNotebook", tabmargins=(0, 4, 0, 0))
        style.configure("Hub.TNotebook.Tab", padding=(16, 8))
        style.configure("Hub.Treeview", rowheight=34, font=("Segoe UI", 10))
        style.configure("Hub.Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("Hub.Horizontal.TProgressbar", thickness=8)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(18, 14, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ttk.Label(header, text="Model Hub", style="HubTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Discover, compare and install local Ollama models without knowing model tags in advance.",
            style="HubSubtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        connection = ttk.Frame(header)
        connection.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(connection, textvariable=self._connection_var).grid(row=0, column=0, sticky="e")
        ttk.Label(connection, textvariable=self._catalog_meta_var, style="HubMeta.TLabel").grid(
            row=1, column=0, sticky="e", pady=(3, 0)
        )

        host_bar = ttk.Frame(self, padding=(18, 2, 18, 8))
        host_bar.grid(row=1, column=0, sticky="ew")
        host_bar.grid_columnconfigure(1, weight=1)
        ttk.Label(host_bar, text="Ollama host:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._host_entry = ttk.Entry(host_bar, textvariable=self._host_var)
        self._host_entry.grid(row=0, column=1, sticky="ew")
        self._refresh_btn = ttk.Button(
            host_bar, text="Refresh", command=self._refresh_installed_models
        )
        self._refresh_btn.grid(row=0, column=2, padx=(8, 0))

        self._notebook = ttk.Notebook(self, style="Hub.TNotebook")
        self._notebook.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 8))

        self._explore_tab = ttk.Frame(self._notebook, padding=10)
        self._installed_tab = ttk.Frame(self._notebook, padding=10)
        self._advanced_tab = ttk.Frame(self._notebook, padding=10)
        self._notebook.add(self._explore_tab, text="Explore models")
        self._notebook.add(self._installed_tab, text="Installed")
        self._notebook.add(self._advanced_tab, text="Advanced")

        self._build_explore_tab()
        self._build_installed_tab()
        self._build_advanced_tab()

        footer = ttk.Frame(self, padding=(18, 0, 18, 14))
        footer.grid(row=3, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self._progress = ttk.Progressbar(
            footer, mode="determinate", maximum=100, style="Hub.Horizontal.TProgressbar"
        )
        self._progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(footer, textvariable=self._status_var, style="HubMeta.TLabel").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        ttk.Button(footer, text="Close", command=self._on_close).grid(
            row=0, column=1, rowspan=2, sticky="e", padx=(12, 0)
        )

    def _build_explore_tab(self) -> None:
        tab = self._explore_tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        toolbar = ttk.Frame(tab)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        toolbar.grid_columnconfigure(1, weight=1)

        ttk.Label(toolbar, text="Search").grid(row=0, column=0, padx=(0, 6))
        search = ttk.Entry(toolbar, textvariable=self._search_var)
        search.grid(row=0, column=1, sticky="ew")
        ttk.Label(toolbar, text="Filter").grid(row=0, column=2, padx=(14, 6))
        ttk.Combobox(
            toolbar,
            textvariable=self._filter_var,
            values=self.FILTERS,
            state="readonly",
            width=15,
        ).grid(row=0, column=3)

        pane = ttk.Panedwindow(tab, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(pane, padding=(0, 0, 8, 0))
        right = ttk.Frame(pane, style="HubCard.TFrame", padding=18)
        pane.add(left, weight=3)
        pane.add(right, weight=4)

        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        ttk.Label(left, text="Available families", style="HubSection.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        columns = ("model", "focus", "installed", "smallest")
        self._explore_tree = ttk.Treeview(
            left, columns=columns, show="headings", selectmode="browse", style="Hub.Treeview"
        )
        self._explore_tree.heading("model", text="Model")
        self._explore_tree.heading("focus", text="Focus")
        self._explore_tree.heading("installed", text="Status")
        self._explore_tree.heading("smallest", text="Smallest")
        self._explore_tree.column("model", width=185, anchor="w")
        self._explore_tree.column("focus", width=135, anchor="w")
        self._explore_tree.column("installed", width=92, anchor="center")
        self._explore_tree.column("smallest", width=80, anchor="e")
        self._explore_tree.grid(row=1, column=0, sticky="nsew")
        self._explore_tree.bind("<<TreeviewSelect>>", self._on_explore_selected)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self._explore_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self._explore_tree.configure(yscrollcommand=scrollbar.set)

        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(3, weight=1)

        self._detail_name = ttk.Label(right, text="Select a model", style="HubModel.TLabel")
        self._detail_name.grid(row=0, column=0, sticky="w")
        self._detail_publisher = ttk.Label(right, text="", style="HubMeta.TLabel")
        self._detail_publisher.grid(row=1, column=0, sticky="w", pady=(1, 10))
        self._detail_summary = ttk.Label(right, text="", wraplength=480, justify="left")
        self._detail_summary.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        detail_body = ttk.Frame(right)
        detail_body.grid(row=3, column=0, sticky="nsew")
        detail_body.grid_columnconfigure(1, weight=1)

        ttk.Label(detail_body, text="Capabilities", style="HubSection.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 5)
        )
        self._capabilities_var = tk.StringVar(value="—")
        ttk.Label(
            detail_body, textvariable=self._capabilities_var, wraplength=470, justify="left"
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        ttk.Label(detail_body, text="Variant:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=4)
        self._variant_combo = ttk.Combobox(
            detail_body, textvariable=self._variant_var, state="readonly", width=38
        )
        self._variant_combo.grid(row=2, column=1, sticky="ew", pady=4)
        self._variant_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_variant_changed())

        ttk.Label(detail_body, text="Parameters:").grid(row=3, column=0, sticky="w", pady=4)
        self._parameters_var = tk.StringVar(value="—")
        ttk.Label(detail_body, textvariable=self._parameters_var).grid(row=3, column=1, sticky="w", pady=4)

        ttk.Label(detail_body, text="Download:").grid(row=4, column=0, sticky="w", pady=4)
        self._download_size_var = tk.StringVar(value="—")
        ttk.Label(detail_body, textvariable=self._download_size_var).grid(row=4, column=1, sticky="w", pady=4)

        ttk.Label(detail_body, text="Context:").grid(row=5, column=0, sticky="w", pady=4)
        self._context_var = tk.StringVar(value="—")
        ttk.Label(detail_body, textvariable=self._context_var).grid(row=5, column=1, sticky="w", pady=4)

        ttk.Label(detail_body, text="Local profile:").grid(row=6, column=0, sticky="w", pady=4)
        self._profile_var = tk.StringVar(value="—")
        ttk.Label(detail_body, textvariable=self._profile_var).grid(row=6, column=1, sticky="w", pady=4)

        self._variant_status_var = tk.StringVar(value="")
        ttk.Label(detail_body, textvariable=self._variant_status_var, style="HubMeta.TLabel").grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        actions = ttk.Frame(right)
        actions.grid(row=4, column=0, sticky="ew", pady=(16, 0))
        actions.grid_columnconfigure(0, weight=1)
        self._official_btn = ttk.Button(actions, text="Official page", command=self._open_selected_official_page)
        self._official_btn.grid(row=0, column=0, sticky="w")
        self._primary_btn = ttk.Button(
            actions,
            text="Download & Use",
            command=self._use_selected_variant,
            style="HubPrimary.TButton",
        )
        self._primary_btn.grid(row=0, column=1, sticky="e")

    def _build_installed_tab(self) -> None:
        tab = self._installed_tab
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ttk.Label(tab, text="Models available in your local Ollama", style="HubSection.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        columns = ("model", "size", "parameters", "quantization")
        self._installed_tree = ttk.Treeview(
            tab, columns=columns, show="headings", selectmode="browse", style="Hub.Treeview"
        )
        self._installed_tree.heading("model", text="Model")
        self._installed_tree.heading("size", text="Size")
        self._installed_tree.heading("parameters", text="Parameters")
        self._installed_tree.heading("quantization", text="Quantization")
        self._installed_tree.column("model", width=360, anchor="w")
        self._installed_tree.column("size", width=110, anchor="e")
        self._installed_tree.column("parameters", width=130, anchor="center")
        self._installed_tree.column("quantization", width=140, anchor="center")
        self._installed_tree.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self._installed_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self._installed_tree.configure(yscrollcommand=scrollbar.set)

        actions = ttk.Frame(tab)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="Refresh", command=self._refresh_installed_models).pack(side="left")
        self._installed_use_btn = ttk.Button(
            actions, text="Register & Use", command=self._use_installed_selection, style="HubPrimary.TButton"
        )
        self._installed_use_btn.pack(side="right")
        self._installed_remove_btn = ttk.Button(
            actions, text="Remove from Ollama", command=self._remove_installed_selection
        )
        self._installed_remove_btn.pack(side="right", padx=(0, 8))

    def _build_advanced_tab(self) -> None:
        tab = self._advanced_tab
        tab.grid_columnconfigure(0, weight=1)

        card = ttk.Frame(tab, style="HubCard.TFrame", padding=20)
        card.grid(row=0, column=0, sticky="ew", pady=(4, 10))
        card.grid_columnconfigure(0, weight=1)

        ttk.Label(card, text="Custom Ollama tag", style="HubSection.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            card,
            text=(
                "Use this when the model you want is not in CodeSmile's curated list. "
                "Any valid Ollama tag can still be downloaded and registered."
            ),
            style="HubSubtitle.TLabel",
            wraplength=780,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        entry_row = ttk.Frame(card)
        entry_row.grid(row=2, column=0, sticky="ew")
        entry_row.grid_columnconfigure(0, weight=1)
        ttk.Entry(entry_row, textvariable=self._advanced_model_var).grid(row=0, column=0, sticky="ew")
        self._advanced_pull_btn = ttk.Button(
            entry_row,
            text="Download & Use",
            command=self._download_advanced_tag,
            style="HubPrimary.TButton",
        )
        self._advanced_pull_btn.grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            entry_row, text="Register installed", command=self._register_advanced_installed
        ).grid(row=0, column=2, padx=(8, 0))

        ttk.Checkbutton(
            card,
            text="Enable provider-native thinking when Ollama reports support",
            variable=self._use_native_thinking_var,
        ).grid(row=3, column=0, sticky="w", pady=(10, 0))

        library = ttk.Frame(tab, padding=(4, 14))
        library.grid(row=1, column=0, sticky="ew")
        library.grid_columnconfigure(0, weight=1)
        ttk.Label(
            library,
            text="Can't find a model? Browse Ollama's official library, copy its tag, then paste it above.",
            style="HubSubtitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            library,
            text="Open Ollama Library",
            command=lambda: webbrowser.open("https://ollama.com/library"),
        ).grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------
    # Model discovery / selection
    # ------------------------------------------------------------------

    def _refresh_explore_models(self) -> None:
        if not hasattr(self, "_explore_tree"):
            return
        category = self._filter_var.get().lower()
        models = self.curated_catalog.search(self._search_var.get(), category=category)
        self._visible_models = models

        current_id = self._selected_model.model_id if self._selected_model else None
        for item in self._explore_tree.get_children():
            self._explore_tree.delete(item)

        for model in models:
            installed_variants = sum(
                1 for variant in model.variants if variant.tag in self._installed_tags
            )
            status = f"✓ {installed_variants}" if installed_variants else "Not installed"
            smallest = model.smallest_variant.download if model.smallest_variant else ""
            focus = self._focus_label(model)
            self._explore_tree.insert(
                "",
                "end",
                iid=model.model_id,
                values=(model.name, focus, status, smallest),
            )

        target_id = current_id if current_id and self._explore_tree.exists(current_id) else None
        if target_id is None and models:
            target_id = models[0].model_id
        if target_id:
            self._explore_tree.selection_set(target_id)
            self._explore_tree.focus(target_id)
            self._explore_tree.see(target_id)
            self._on_explore_selected()
        elif not models:
            self._clear_model_details()

    @staticmethod
    def _focus_label(model: OllamaCatalogModel) -> str:
        labels = []
        categories = set(model.categories)
        if "coding" in categories:
            labels.append("Coding")
        if "reasoning" in categories:
            labels.append("Reasoning")
        if "lightweight" in categories:
            labels.append("Lightweight")
        if "agentic" in categories:
            labels.append("Agentic")
        if "legacy" in categories:
            labels.append("Legacy")
        return " · ".join(labels[:2]) or "General"

    def _on_explore_selected(self, _event=None) -> None:
        selection = self._explore_tree.selection()
        if not selection:
            return
        try:
            model = self.curated_catalog.get(selection[0])
        except KeyError:
            return
        self._selected_model = model
        self._detail_name.configure(text=model.name)
        self._detail_publisher.configure(text=f"{model.publisher} · Ollama model family")
        self._detail_summary.configure(text=model.summary)
        self._capabilities_var.set("  •  ".join(model.capabilities) or "—")
        self._official_btn.configure(state="normal" if model.official_url else "disabled")

        labels = [self._variant_label(variant) for variant in model.variants]
        self._variant_combo["values"] = labels
        default_index = next(
            (i for i, variant in enumerate(model.variants) if variant.tag == model.default_tag),
            0,
        )
        if labels:
            self._variant_combo.current(default_index)
            self._variant_var.set(labels[default_index])
        else:
            self._variant_var.set("")
        self._on_variant_changed()

    @staticmethod
    def _variant_label(variant: OllamaModelVariant) -> str:
        return f"{variant.parameters}  ·  {variant.download}  ·  {variant.context}  —  {variant.tag}"

    def _selected_variant(self) -> Optional[OllamaModelVariant]:
        if self._selected_model is None:
            return None
        selected = self._variant_var.get()
        for variant in self._selected_model.variants:
            if self._variant_label(variant) == selected:
                return variant
        return self._selected_model.variant_by_tag(self._selected_model.default_tag)

    def _on_variant_changed(self) -> None:
        variant = self._selected_variant()
        if variant is None:
            self._parameters_var.set("—")
            self._download_size_var.set("—")
            self._context_var.set("—")
            self._profile_var.set("—")
            self._variant_status_var.set("")
            self._primary_btn.configure(state="disabled")
            return

        self._parameters_var.set(variant.parameters)
        self._download_size_var.set(variant.download)
        self._context_var.set(variant.context)
        self._profile_var.set(variant.profile)
        installed = variant.tag in self._installed_tags
        if installed:
            self._variant_status_var.set("✓ Already installed in Ollama — no download required")
            self._primary_btn.configure(text="Use in CodeSmile", state="normal")
        else:
            self._variant_status_var.set("Not installed locally")
            self._primary_btn.configure(text="Download & Use", state="normal")

    def _clear_model_details(self) -> None:
        self._selected_model = None
        self._detail_name.configure(text="No matching model")
        self._detail_publisher.configure(text="Try another search or filter.")
        self._detail_summary.configure(text="")
        self._capabilities_var.set("—")
        self._variant_combo["values"] = []
        self._variant_var.set("")
        self._on_variant_changed()

    def _open_selected_official_page(self) -> None:
        if self._selected_model and self._selected_model.official_url:
            webbrowser.open(self._selected_model.official_url)

    # ------------------------------------------------------------------
    # Ollama operations
    # ------------------------------------------------------------------

    def _manager(self, host: Optional[str] = None) -> OllamaModelManager:
        return OllamaModelManager(host=host)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for widget in (
            self._refresh_btn,
            self._primary_btn,
            self._installed_use_btn,
            self._installed_remove_btn,
            self._advanced_pull_btn,
        ):
            try:
                widget.configure(state=state)
            except Exception:
                pass
        try:
            self._host_entry.configure(state=state)
        except Exception:
            pass

    def _run_background(self, work, on_success) -> None:
        if self._busy:
            return
        self._set_busy(True)

        def runner():
            try:
                result = work()
            except Exception as exc:
                self._event_queue.put(("error", exc))
                return
            self._event_queue.put(("success", result, on_success))

        threading.Thread(target=runner, daemon=True).start()

    def _drain_worker_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "progress":
                _, status, completed, total = event
                self._apply_progress(status, completed, total)
            elif kind == "error":
                self._finish_error(event[1])
            elif kind == "success":
                _, result, callback = event
                self._finish_success(result, callback)

        if self.winfo_exists():
            self.after(50, self._drain_worker_events)

    def _finish_error(self, exc: Exception) -> None:
        self._set_busy(False)
        self._progress.stop()
        self._progress.configure(mode="determinate", value=0)
        self._status_var.set("Operation failed")
        self._connection_var.set("● Ollama unavailable")
        messagebox.showerror("Ollama", str(exc), parent=self)

    def _finish_success(self, result, callback) -> None:
        self._set_busy(False)
        self._progress.stop()
        self._progress.configure(mode="determinate", value=100)
        callback(result)

    def _refresh_installed_models(self) -> None:
        self._status_var.set("Reading installed Ollama models…")
        self._connection_var.set("● Checking Ollama")
        self._progress.configure(mode="indeterminate")
        self._progress.start(80)

        host = self._host_var.get().strip() or None
        manager = self._manager(host)
        self._run_background(manager.list_installed_models, self._show_installed_models)

    def _show_installed_models(self, models: list[dict]) -> None:
        self._installed_models = models
        self._installed_tags = self.curated_catalog.installed_tags(models)

        for item in self._installed_tree.get_children():
            self._installed_tree.delete(item)

        for index, model in enumerate(models):
            name = model.get("model", "")
            iid = f"installed-{index}"
            self._installed_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    name,
                    _format_size(model.get("size")),
                    model.get("parameter_size") or "",
                    model.get("quantization_level") or "",
                ),
            )

        self._connection_var.set("● Ollama connected")
        self._status_var.set(f"{len(models)} local model(s) installed")
        self._progress.stop()
        self._progress.configure(mode="determinate", value=0)
        self._refresh_explore_models()
        self._on_variant_changed()

    def _progress_callback(self, status: str, completed, total) -> None:
        self._event_queue.put(("progress", status, completed, total))

    def _apply_progress(self, status: str, completed, total) -> None:
        self._status_var.set(status or "Downloading…")
        try:
            c = float(completed)
            t = float(total)
        except (TypeError, ValueError):
            self._progress.configure(mode="indeterminate")
            self._progress.start(80)
            return

        if t > 0:
            self._progress.stop()
            self._progress.configure(
                mode="determinate", maximum=100, value=max(0, min(100, c / t * 100))
            )

    def _use_selected_variant(self) -> None:
        variant = self._selected_variant()
        if variant is None:
            return
        if variant.tag in self._installed_tags:
            self._inspect_and_register(variant.tag)
        else:
            self._download_and_register(variant.tag)

    def _download_and_register(self, model_name: str) -> None:
        model_name = (model_name or "").strip()
        if not model_name:
            messagebox.showwarning("Model", "Insert an Ollama model tag.", parent=self)
            return

        self._status_var.set(f"Downloading {model_name}…")
        self._progress.configure(mode="indeterminate")
        self._progress.start(80)

        host = self._host_var.get().strip() or None
        use_native_thinking = self._use_native_thinking_var.get()
        manager = self._manager(host)

        def work():
            manager.pull_model(model_name, self._progress_callback)
            supports_thinking = (
                manager.supports_native_thinking(model_name) if use_native_thinking else False
            )
            return model_name, supports_thinking

        self._run_background(work, self._register_downloaded_model)

    def _register_downloaded_model(self, result) -> None:
        model_name, supports_thinking = result
        self._register_model(model_name, supports_thinking=supports_thinking)
        self._refresh_installed_models()

    def _inspect_and_register(self, model_name: str) -> None:
        model_name = (model_name or "").strip()
        if not model_name:
            return
        self._status_var.set(f"Preparing {model_name}…")
        self._progress.configure(mode="indeterminate")
        self._progress.start(80)

        host = self._host_var.get().strip() or None
        use_native_thinking = self._use_native_thinking_var.get()
        manager = self._manager(host)

        def work():
            supports = manager.supports_native_thinking(model_name) if use_native_thinking else False
            return model_name, supports

        self._run_background(work, lambda result: self._register_model(result[0], result[1]))

    def _register_model(self, model_name: str, supports_thinking: bool) -> None:
        provider_id = self.catalog_service.add_or_update_local_ollama_provider(
            model_name,
            host=self._host_var.get().strip() or None,
            options={
                "temperature": 0,
                "num_predict": 768,
                "num_ctx": 8192,
            },
            response_format="json",
            think=True if supports_thinking else None,
        )

        if self.on_catalog_changed:
            self.on_catalog_changed()
        if self.on_model_ready:
            self.on_model_ready(provider_id)

        self._status_var.set(
            f"Ready: {model_name}"
            + (" · native thinking enabled" if supports_thinking else "")
        )
        messagebox.showinfo(
            "CodeSmile",
            f"'{model_name}' is installed, registered and ready to use.",
            parent=self,
        )

    def _selected_installed_tag(self) -> Optional[str]:
        selection = self._installed_tree.selection()
        if not selection:
            return None
        values = self._installed_tree.item(selection[0], "values")
        return str(values[0]).strip() if values else None

    def _use_installed_selection(self) -> None:
        model_name = self._selected_installed_tag()
        if not model_name:
            messagebox.showwarning("Model", "Select an installed model first.", parent=self)
            return
        self._inspect_and_register(model_name)

    def _remove_installed_selection(self) -> None:
        model_name = self._selected_installed_tag()
        if not model_name:
            messagebox.showwarning("Model", "Select an installed model first.", parent=self)
            return
        if not messagebox.askyesno(
            "Remove model",
            f"Remove '{model_name}' from local Ollama storage?\n\n"
            "This does not remove CodeSmile results or prompts.",
            parent=self,
        ):
            return

        host = self._host_var.get().strip() or None
        manager = self._manager(host)
        self._status_var.set(f"Removing {model_name}…")
        self._progress.configure(mode="indeterminate")
        self._progress.start(80)

        def work():
            manager.delete_model(model_name)
            return model_name

        self._run_background(work, lambda _name: self._refresh_installed_models())

    def _download_advanced_tag(self) -> None:
        self._download_and_register(self._advanced_model_var.get())

    def _register_advanced_installed(self) -> None:
        model_name = self._advanced_model_var.get().strip()
        if not model_name:
            messagebox.showwarning("Model", "Insert an Ollama model tag.", parent=self)
            return
        if model_name not in self._installed_tags:
            messagebox.showwarning(
                "Model",
                f"'{model_name}' is not currently listed by this Ollama instance.\n"
                "Use 'Download & Use' instead.",
                parent=self,
            )
            return
        self._inspect_and_register(model_name)

    def _on_close(self) -> None:
        if self._busy:
            messagebox.showinfo(
                "Ollama",
                "A model operation is still running. Close this window after it finishes.",
                parent=self,
            )
            return
        self.destroy()
