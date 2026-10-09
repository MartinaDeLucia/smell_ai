import json
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from components.project_analyzer import ProjectAnalyzer
from gui.textbox_redirect import TextBoxRedirect
from gui.folder_utils import open_folder
from llm_detection.catalog_service import LLMCatalogService
from llm_detection.types import ProviderKind, PromptMode
from llm_detection.providers import LocalLLMProvider, ApiLLMProvider
from llm_detection.orchestrator import LLMOrchestrator
from gui.ollama_model_manager_gui import OllamaModelManagerDialog

class CodeSmellDetectorGUI:
    """
    The main GUI for the AI-specific Code Smells Detector application.
    """

    def __init__(self, master):
        self.master = master
        self.catalog_service = LLMCatalogService()
        self.setup_gui()
        self.configure_stdout()
        self.project_analyzer = None

    def setup_gui(self):
        """
        Sets up the GUI layout and components.
        """
        self.master.title("AI-Specific Code Smells Detector")
        self.master.geometry("700x650")

        # Input Path Selection
        self.input_label = tk.Label(self.master, text="Input Path:")
        self.input_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.input_path = tk.Label(
            self.master, text="No path selected", anchor="w"
        )
        self.input_path.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        self.input_button = tk.Button(
            self.master,
            text="Choose Input Folder",
            bg="lightblue",
            command=self.choose_input_path,
        )
        self.input_button.grid(row=0, column=2, padx=5, pady=2)

        # Output Path Selection
        self.output_label = tk.Label(self.master, text="Output Path:")
        self.output_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        self.output_path = tk.Label(
            self.master, text="No path selected", anchor="w"
        )
        self.output_path.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        output_buttons_frame = tk.Frame(self.master)
        output_buttons_frame.grid(
            row=1,
            column=2,
            padx=5,
            pady=2,
            sticky="w",
        )

        self.output_button = tk.Button(
            output_buttons_frame,
            text="Choose Folder",
            bg="lightblue",
            command=self.choose_output_path,
        )
        self.output_button.pack(
            side=tk.LEFT,
            padx=(0, 4),
        )

        self.open_output_button = tk.Button(
            output_buttons_frame,
            text="Open Results Folder",
            command=self.open_output_folder,
            state="disabled",
        )
        self.open_output_button.pack(side=tk.LEFT)
        # Walker Selection
        self.walker_label = tk.Label(
            self.master, text="Select number of walkers:"
        )
        self.walker_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)

        self.walker_picker = tk.Spinbox(self.master, from_=1, to=10, width=5)
        self.walker_picker.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        # Parallel Checkbox
        self.parallel_var = tk.BooleanVar()
        self.parallel_check = tk.Checkbutton(
            self.master, text="Parallel", variable=self.parallel_var
        )
        self.parallel_check.grid(row=3, column=0, sticky="w", padx=5, pady=2)

        # Resume Checkbox
        self.resume_var = tk.BooleanVar()
        self.resume_check = tk.Checkbutton(
            self.master, text="Resume", variable=self.resume_var
        )
        self.resume_check.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        # Multiple Checkbox
        self.multiple_var = tk.BooleanVar()
        self.multiple_check = tk.Checkbutton(
            self.master, text="Multiple", variable=self.multiple_var
        )
        self.multiple_check.grid(row=3, column=2, sticky="w", padx=5, pady=2)

        # -------- LLM Section --------
        # LLM Checkbox
        self.llm_var = tk.BooleanVar()
        self.llm_check = tk.Checkbutton(
            self.master, 
            text="LLM Detection", 
            variable=self.llm_var,
            command=self.toggle_llm_controls
        )
        self.llm_check.grid(row=4, column=0, sticky="w", padx=5, pady=5)

        # LLM Frame (hidden by default)
        self.llm_frame = tk.LabelFrame(self.master, text="LLM Configuration", padx=10, pady=10)
        self.llm_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.llm_frame.grid_remove()  # Hide initially

        # Provider Type Selection (Local/API)
        self.provider_type_var = tk.StringVar(value="local")
        
        provider_type_frame = tk.Frame(self.llm_frame)
        provider_type_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        
        tk.Label(provider_type_frame, text="Provider Type:").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            provider_type_frame, 
            text="Local", 
            variable=self.provider_type_var, 
            value="local",
            command=self.update_provider_list
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            provider_type_frame, 
            text="API", 
            variable=self.provider_type_var, 
            value="api",
            command=self.update_provider_list
        ).pack(side=tk.LEFT, padx=5)

        # Provider Selection
        tk.Label(self.llm_frame, text="Select Provider:").grid(row=1, column=0, sticky="w", pady=5)
        self.provider_combo = ttk.Combobox(self.llm_frame, state="readonly", width=30)
        self.provider_combo.grid(row=1, column=1, sticky="w", pady=5)
        self.manage_models_button = tk.Button(
            self.llm_frame,
            text="Browse / Download Models",
            command=self.open_model_manager,
        )
        self.manage_models_button.grid(row=1, column=2, sticky="w", padx=8, pady=5)

        # Code Smell Selection
        tk.Label(self.llm_frame, text="Select Code Smells:").grid(row=2, column=0, sticky="nw", pady=5)
        
        smell_frame = tk.Frame(self.llm_frame)
        smell_frame.grid(row=2, column=1, sticky="w", pady=5)
        
        smell_scrollbar = tk.Scrollbar(smell_frame)
        smell_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.smell_listbox = tk.Listbox(
            smell_frame, 
            selectmode=tk.MULTIPLE, 
            height=6, 
            width=40,
            yscrollcommand=smell_scrollbar.set
        )
        self.smell_listbox.pack(side=tk.LEFT, fill=tk.BOTH)
        smell_scrollbar.config(command=self.smell_listbox.yview)

        self.show_reasoning_var = tk.BooleanVar(value=True)
        self.show_reasoning_check = tk.Checkbutton(
            self.llm_frame,
            text="Show finding reasoning in output",
            variable=self.show_reasoning_var,
        )
        self.show_reasoning_check.grid(row=3, column=1, sticky="w", pady=(0, 5))

        # Initialize LLM data
        self.load_llm_data()

        # Output Textbox
        self.output_textbox = tk.Text(
            self.master, height=10, width=60, state="disabled"
        )
        self.output_textbox.grid(
            row=6, column=0, columnspan=3, pady=10, padx=5, sticky="nsew"
        )
        self.output_textbox.bind("<Key>", self.disable_key_press)

        # Run and Exit Buttons
        self.run_button = tk.Button(
            self.master, text="Run", bg="lightgreen", command=self.run_program
        )
        self.run_button.grid(row=7, column=0, pady=5, padx=5)

        self.exit_button = tk.Button(
            self.master, text="Exit", bg="pink", command=self.master.quit
        )
        self.exit_button.grid(row=7, column=2, pady=5, padx=5)

        # Grid Configuration
        self.master.grid_rowconfigure(6, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

    def configure_stdout(self):
        """
        Redirects stdout to the GUI Text widget.
        """
        output_redirect = TextBoxRedirect(self.output_textbox)
        sys.stdout = output_redirect

    def disable_key_press(self, event):
        """
        Disables user input in the output Text widget.
        """
        return "break"

    def choose_input_path(self):
        """
        Opens a folder selection dialog for input path.
        """
        path = filedialog.askdirectory()
        if path:
            self.input_path.configure(text=path)

    def choose_output_path(self):
        """
        Opens a folder selection dialog for output path.
        """
        path = filedialog.askdirectory()

        if path:
            self.output_path.configure(text=path)
            self.open_output_button.configure(state="normal")

    def open_output_folder(self):
        """
        Open the selected output directory in the operating system's
        default file manager.
        """
        path = str(self.output_path.cget("text") or "").strip()

        if not path or path == "No path selected":
            messagebox.showwarning(
                "Output folder",
                "Please select an output folder first.",
            )
            return

        try:
            open_folder(path)
        except Exception as e:
            messagebox.showerror(
                "Output folder",
                f"Could not open the output folder:\n{e}",
            )

    def load_llm_data(self):
        """
        Load LLM providers and detectable smells from catalog.
        """
        try:
            self.catalog = self.catalog_service.load()
            self.update_provider_list()
            self.update_smell_list()
        except Exception as e:
            print(f"Warning: Could not load LLM catalog: {e}")

    def open_model_manager(self):
        """Open the Ollama model manager and refresh providers after changes."""
        OllamaModelManagerDialog(
            self.master,
            catalog_service=self.catalog_service,
            on_catalog_changed=self.load_llm_data,
            on_model_ready=self._select_provider_by_id,
        )

    def _select_provider_by_id(self, provider_id):
        """Refresh the local-provider list and select the model chosen in Model Hub."""
        self.provider_type_var.set("local")
        self.update_provider_list()
        catalog = self.catalog_service.load()
        provider = next(
            (p for p in catalog.providers if p.provider_id == provider_id and p.kind == ProviderKind.LOCAL),
            None,
        )
        if provider is not None:
            self.provider_combo.set(provider.display_name)

    def toggle_llm_controls(self):
        """
        Show/hide LLM configuration frame based on checkbox state.
        """
        if self.llm_var.get():
            self.llm_frame.grid()
        else:
            self.llm_frame.grid_remove()

    def update_provider_list(self):
        """
        Update provider combobox based on selected provider type (Local/API).
        """
        try:
            self.catalog = self.catalog_service.load()
            provider_type = self.provider_type_var.get()
            kind = ProviderKind.LOCAL if provider_type == "local" else ProviderKind.API
            
            providers = [
                p for p in self.catalog.providers 
                if p.kind == kind
            ]
            
            provider_names = [p.display_name for p in providers]
            self.provider_combo['values'] = provider_names
            
            if provider_names:
                self.provider_combo.current(0)
            else:
                self.provider_combo.set('')
                
        except Exception as e:
            print(f"Error updating provider list: {e}")

    def update_smell_list(self):
        """
        Update smell listbox with only detectable smells (those with default prompt).
        """
        try:
            detectable_smells = self.catalog_service.list_detectable_smells()
            
            self.smell_listbox.delete(0, tk.END)
            
            for smell in detectable_smells:
                self.smell_listbox.insert(tk.END, smell.display_name)
                
        except Exception as e:
            print(f"Error updating smell list: {e}")

    def run_program(self):
        """
        Executes the analysis program with
        selected parameters in a separate thread.
        """
        # Validate paths
        input_path = self.input_path.cget("text")
        output_path = self.output_path.cget("text")

        if (
            input_path == "No path selected"
            or output_path == "No path selected"
        ):
            print("Error: Please select valid input and output paths.")
            return

        # Gather parameters
        try:
            num_walkers = int(self.walker_picker.get())
        except ValueError:
            print("Error: max_workers must be a number.")
            return
        
        if num_walkers <= 0:
            print("Error: max_workers must be greater than 0.")
            return
        
        is_parallel = self.parallel_var.get()
        is_resume = self.resume_var.get()
        is_multiple = self.multiple_var.get()
        
        # LLM parameters
        use_llm = self.llm_var.get()
        show_reasoning = self.show_reasoning_var.get()
        llm_provider_id = None
        selected_smell_ids = []
        
        if use_llm:
            # Validate provider selection (UC01 Step 9)
            provider_name = self.provider_combo.get()
            if not provider_name:
                print("Error: Please select an LLM provider.")
                return
            
            # Find provider by display name
            provider_type = self.provider_type_var.get()
            kind = ProviderKind.LOCAL if provider_type == "local" else ProviderKind.API
            matching_providers = [
                p for p in self.catalog.providers 
                if p.display_name == provider_name and p.kind == kind
            ]
            
            if not matching_providers:
                print(f"Error: Provider '{provider_name}' not found.")
                return
                
            llm_provider_id = matching_providers[0].provider_id
            
            # Validate smell selection (UC01 Step 13)
            selected_indices = self.smell_listbox.curselection()
            if not selected_indices:
                # Check if there are any detectable smells (UC01 Scenario 11.a1)
                detectable_smells = self.catalog_service.list_detectable_smells()
                if not detectable_smells:
                    print("Warning: Non sono presenti Code Smell detectabili tramite LLM, l'analisi procederà in modo statico")
                    use_llm = False
                else:
                    print("Error: Please select at least one code smell.")
                    return
            else:
                # Get selected smell IDs
                detectable_smells = self.catalog_service.list_detectable_smells()
                for idx in selected_indices:
                    smell_name = self.smell_listbox.get(idx)
                    matching_smell = next(
                        (s for s in detectable_smells if s.display_name == smell_name), 
                        None
                    )
                    if matching_smell:
                        selected_smell_ids.append(matching_smell.smell_id)

        # Start analysis in a new thread
        analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(
                input_path,
                output_path,
                num_walkers,
                is_parallel,
                is_resume,
                is_multiple,
                use_llm,
                llm_provider_id,
                selected_smell_ids,
                show_reasoning,
            ),
            daemon=True,
        )
        analysis_thread.start()

    def run_analysis(
        self,
        input_path,
        output_path,
        num_walkers,
        is_parallel,
        is_resume,
        is_multiple,
        use_llm=False,
        llm_provider_id=None,
        selected_smell_ids=None,
        show_reasoning=True,
    ):
        """
        Performs the actual analysis. This runs on a separate thread.
        """
        try:
            print(f"Input Path: {input_path}")
            print(f"Output Path: {output_path}")
            print(f"Number of Walkers: {num_walkers}")
            print(f"Parallel Execution: {is_parallel}")
            print(f"Resume Execution: {is_resume}")
            print(f"Analyze multiple projects: {is_multiple}")
            print(f"LLM Detection: {use_llm}")
            
            if use_llm and llm_provider_id and selected_smell_ids:
                print(f"LLM Provider: {llm_provider_id}")
                print(f"Selected Smells: {', '.join(selected_smell_ids)}")

            self.project_analyzer = ProjectAnalyzer(output_path)

            if not is_resume:
                self.project_analyzer.clean_output_directory()

            if is_multiple:
                # Validate that there are at least 2 projects
                project_count = sum(
                    1 for item in os.listdir(input_path)
                    if os.path.isdir(os.path.join(input_path, item))
                    and item not in {"output", "execution_log.txt"}
                )
                
                if project_count < 2:
                    print("Multiple mode isn’t available with only one project.")
                    return
                
                print("Analyzing project(s)...")

                if is_parallel:
                    # In parallel mode, resume is not supported
                    if is_resume:
                        print("Warning: In parallel mode, resume mode is ignored.")
                    
                    self.project_analyzer.analyze_projects_parallel(
                        base_path=input_path,
                        max_workers=num_walkers,
                    )
                else:
                    self.project_analyzer.analyze_projects_sequential(
                        base_path=input_path, resume=is_resume
                    )

                self.project_analyzer.merge_all_results()
            else:
                # Warn if Parallel or Resume are set without Multiple mode
                if is_parallel and is_resume:
                    print("Warning: Parallel mode and Resume mode are ignored without Multiple mode.")
                elif is_parallel:
                    print("Warning: Parallel mode is ignored without Multiple mode.")
                elif is_resume:
                    print("Warning: Resume mode is ignored without Multiple mode.")
                
                total_smells = self.project_analyzer.analyze_project(
                    input_path
                )
                print(
                    f"Analysis completed. "
                    f"Total code smells found: {total_smells}"
                )
            
            # LLM Detection (UC01 Steps 10-16)
            if use_llm and llm_provider_id and selected_smell_ids:
                print("\n--- Starting LLM Detection ---")
                self._run_llm_detection(
                    input_path, 
                    output_path, 
                    llm_provider_id, 
                    selected_smell_ids,
                    show_reasoning=show_reasoning,
                )

        except Exception as e:
            print(f"An error occurred during analysis: {e}")

    def _check_python_files(self, path):
        """
        Check if the given path contains any Python files.
        """
        if os.path.isfile(path):
            return path.endswith('.py')
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    return True
        return False

    def _run_llm_detection(
        self,
        input_path,
        output_path,
        provider_id,
        smell_ids,
        show_reasoning=True,
    ):
        """Run LLM detection on the input path and save explainable results."""
        try:
            # Reload the catalog at run time so prompt/model changes made in other
            # windows are immediately visible.
            self.catalog = self.catalog_service.load()
            provider_def = self.catalog.get_provider(provider_id)
            
            # Create provider instance
            if provider_def.kind == ProviderKind.LOCAL:
                config = provider_def.config
                provider = LocalLLMProvider(
                    model_name=config.get("model_name", "qwen2.5-coder:7b"),
                    host=config.get("host"),
                    options=config.get("options"),
                    response_format=config.get("format"),
                    think=config.get("think"),
                )
                print(f"Using local LLM: {provider_def.display_name}")
            else:
                config = provider_def.config
                provider = ApiLLMProvider(
                    base_url=config.get("base_url", "http://localhost:8000"),
                    timeout_s=config.get("timeout_s", 60.0)
                )
                print(f"Using API provider: {provider_def.display_name}")
            
            # Create orchestrator
            orchestrator = LLMOrchestrator(provider, self.catalog)
            
            # Collect Python files as detection targets
            from llm_detection.types import DetectionTarget
            targets = []
            
            if os.path.isfile(input_path):
                if input_path.endswith('.py'):
                    with open(input_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    targets.append(DetectionTarget(
                        filename=os.path.basename(input_path),
                        code=code
                    ))
            else:
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    code = f.read()
                                relative_path = os.path.relpath(file_path, input_path)
                                targets.append(DetectionTarget(
                                    filename=relative_path,
                                    code=code
                                ))
                            except Exception as e:
                                print(f"Warning: Could not read {file_path}: {e}")
            
            if not targets:
                print("No Python files found for LLM detection.")
                return
            
            print(f"Running LLM detection on {len(targets)} file(s)...")
            
            # Run detection
            findings, stats = orchestrator.detect(
                targets=targets,
                smell_ids=smell_ids,
                prompt_mode=PromptMode.DEFAULT
            )
            
            print(f"\nLLM Detection completed:")
            print(f"  - Files processed: {stats.targets_processed}")
            print(f"  - Smells analyzed: {stats.smells_processed}")
            print(f"  - Prompts sent: {stats.prompts_sent}")
            print(f"  - Findings detected: {len(findings)}")
            
            # Breakdown per file
            if findings:
                print(f"\nFindings breakdown by file:")
                findings_by_file = {}
                for finding in findings:
                    filename = finding.filename
                    if filename not in findings_by_file:
                        findings_by_file[filename] = []
                    findings_by_file[filename].append(finding)
                
                for filename in sorted(findings_by_file.keys()):
                    file_findings = findings_by_file[filename]
                    print(f"  - {filename}: {len(file_findings)} code smell(s)")

                if show_reasoning:
                    print("\n--- Finding Reasoning ---")
                    for finding in findings:
                        rationale = (getattr(finding, "reasoning", "") or "").strip()
                        print(
                            f"{finding.filename}:{finding.line} [{finding.smell_name}] "
                            f"{rationale or '(reasoning not returned by model)'}"
                        )
            
            # Save results. The CSV contains the portable per-finding reasoning;
            # a JSONL sidecar preserves provider-native thinking when available.
            if findings:
                self._save_llm_findings(findings, output_path)
                self._save_llm_traces(orchestrator.last_traces, output_path)
                print(f"\nLLM findings saved to: {output_path}")
            else:
                print("\nNo LLM findings detected.")
                
        except Exception as e:
            print(f"Error during LLM detection: {e}")
            import traceback
            traceback.print_exc()

    def _save_llm_traces(self, traces, output_path):
        """Persist raw final responses and optional provider-native thinking."""
        if not traces:
            return

        os.makedirs(output_path, exist_ok=True)
        trace_file = os.path.join(output_path, "llm_generation_trace.jsonl")
        with open(trace_file, "w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(
                    json.dumps(
                        {
                            "filename": trace.filename,
                            "smell_id": trace.smell_id,
                            "response": trace.response,
                            "native_reasoning": trace.native_reasoning,
                            "metadata": trace.metadata or {},
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        print(f"LLM generation trace saved to: {trace_file}")

    def _save_llm_findings(self, findings, output_path):
        """
        Save LLM findings to CSV file in the output directory.
        """
        import pandas as pd
        
        # Convert findings to dataframe format
        rows = [finding.to_overview_row() for finding in findings]
        df = pd.DataFrame(rows)
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save to CSV
        output_file = os.path.join(output_path, "llm_detection_results.csv")
        df.to_csv(output_file, index=False)
        print(f"LLM results saved to: {output_file}")
