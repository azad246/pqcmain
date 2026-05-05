import tkinter as tk
from tkinter import ttk
import sys
import threading
import subprocess
import json
from pathlib import Path
import os
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import csv
import shutil
from datetime import datetime
from PIL import Image, ImageTk

# Add project root to path so dashboard can access config
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    import config
except ImportError:
    pass

class PQCDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- Main Window Configuration ---
        self.title("PQC-IoT Sentinel — Control Dashboard")
        self.geometry("1200x800")
        self.minsize(900, 600)
        self.configure(bg="#1e1e2e")
        
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        # Modern Dark Theme Palette
        self.bg_main = "#1e1e2e"
        self.bg_alt = "#181825"
        self.fg_main = "#cdd6f4"
        self.fg_sub = "#a6adc8"
        self.accent = "#89b4fa"
        
        style.configure(".", background=self.bg_main, foreground=self.fg_main, font=("Segoe UI", 10))
        style.configure("TFrame", background=self.bg_main)
        style.configure("TLabel", background=self.bg_main, foreground=self.fg_main)
        style.configure("Header.TLabel", font=("Segoe UI", 24, "bold"), foreground=self.accent)
        style.configure("Sub.TLabel", font=("Segoe UI", 12), foreground=self.fg_sub)
        style.configure("TLabelframe", background=self.bg_main, foreground=self.accent, bordercolor=self.fg_sub)
        style.configure("TLabelframe.Label", background=self.bg_main, foreground=self.accent, font=("Segoe UI", 11, "bold"))
        
        style.configure("TButton", background=self.bg_alt, foreground=self.fg_main, font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", self.accent)], foreground=[("active", self.bg_main)])
        
        style.configure("Sidebar.TFrame", background=self.bg_alt)
        style.configure("Sidebar.TLabel", background=self.bg_alt, foreground=self.fg_main)
        
        style.configure("Treeview", background=self.bg_main, foreground=self.fg_main, fieldbackground=self.bg_main, bordercolor=self.bg_alt)
        style.configure("Treeview.Heading", background=self.bg_alt, foreground=self.accent, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[('selected', self.accent)], foreground=[('selected', self.bg_main)])
            
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1) 

        # --- Component Initialization ---
        self.create_sidebar()
        self.create_main_content()
        self.create_status_bar()
        
        self.show_view("Dataset")

    def create_sidebar(self):
        self.sidebar_frame = ttk.Frame(self, width=250, padding=15, style="Sidebar.TFrame")
        self.sidebar_frame.grid(row=0, column=0, sticky="nswe")
        self.sidebar_frame.grid_propagate(False)

        title_lbl = ttk.Label(self.sidebar_frame, text="PQC-IoT Sentinel", font=("Segoe UI", 16, "bold"), style="Sidebar.TLabel")
        title_lbl.pack(pady=(10, 5))
        subtitle_lbl = ttk.Label(self.sidebar_frame, text="Control Interface", font=("Segoe UI", 10), foreground=self.fg_sub, style="Sidebar.TLabel")
        subtitle_lbl.pack(pady=(0, 30))

        nav_buttons = [
            "Dataset",
            "Preprocessing",
            "Train Models",
            "Federation",
            "Crypto",
            "Simulation",
            "Benchmark",
            "Results",
            "Paper Figures"
        ]

        for btn_text in nav_buttons:
            btn = ttk.Button(
                self.sidebar_frame, 
                text=btn_text, 
                command=lambda t=btn_text: self.show_view(t)
            )
            btn.pack(fill=tk.X, pady=4, ipady=6)
            
        ttk.Frame(self.sidebar_frame).pack(expand=True)
        
        exit_btn = ttk.Button(self.sidebar_frame, text="Exit Application", command=self.quit)
        exit_btn.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

    def create_main_content(self):
        self.content_frame = ttk.Frame(self, padding=30)
        self.content_frame.grid(row=0, column=1, sticky="nswe")

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("System Initialized and Ready.")
        
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, padding=(10, 5), anchor=tk.W)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="we")

    def set_status(self, message):
        self.status_var.set(message)
        self.update_idletasks()

    def log_message(self, message):
        """Append a message to the log area."""
        if hasattr(self, 'log_area'):
            self.log_area.config(state=tk.NORMAL)
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.see(tk.END)
            self.log_area.config(state=tk.DISABLED)
            self.update_idletasks()

    def show_view(self, view_name):
        """Routes UI generation to the relevant method dynamically."""
        self.set_status(f"Navigated to: {view_name} Module")
        
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        title = ttk.Label(self.content_frame, text=view_name, style="Header.TLabel")
        title.pack(anchor=tk.W, pady=(0, 10))
        
        descriptions = {
            "Dataset": "Data Loading module: Ingests raw N-BaIoT device CSVs, appends labels, and handles schema generation.",
            "Preprocessing": "Pipeline module: Applies MinMaxScaler bounds, executes VarianceThreshold filtering, and enforces 70/15/15 stratified edge-device splits.",
            "Train Models": "Local Execution module: Trains isolated Random Forest classifiers and deep Keras Autoencoders on specific node clusters.",
            "Federation": "Orchestration module: Bootstraps the secure Flower (flwr) gRPC Server and multiplexes connections for decentralized client execution.",
            "Crypto": "Security module: Generates Kyber512 PQC KEM keys, establishes AES-128 tunnels, and measures hybrid payload serialization overhead.",
            "Benchmark": "Analysis module: Simulates and evaluates Centralized RSA against Standard FL and our Proposed Secure PQC-FL architectures.",
            "Results": "Verification module: Exposes zero-shot generalization capabilities via cross-dataset anomaly mapping against the UNSW-NB15 protocol.",
            "Paper Figures": "Publication module: Converts raw CSV and JSON tracking metrics into high-resolution (300dpi) visualizations for IEEE/ACM formats."
        }
        
        desc_text = descriptions.get(view_name, "Loading module parameters...")
        desc = ttk.Label(self.content_frame, text=desc_text, style="Sub.TLabel", wraplength=800)
        desc.pack(anchor=tk.W, pady=(0, 20))
        
        if view_name == "Dataset":
            self.build_dataset_view()
        elif view_name == "Preprocessing":
            self.build_preprocessing_view()
        elif view_name == "Train Models":
            self.build_train_models_view()
        elif view_name == "Federation":
            self.build_federation_view()
        elif view_name == "Crypto":
            self.build_crypto_view()
        elif view_name == "Simulation":
            self.build_simulation_view()
        elif view_name == "Benchmark":
            self.build_benchmark_view()
        elif view_name == "Results":
            self.build_results_view()
        elif view_name == "Paper Figures":
            self.build_figures_view()
        else:
            self.build_placeholder_view(view_name)

    # ==============================================================================
    # 1. DATASET VIEW
    # ==============================================================================
    def build_dataset_view(self):
        status_frame = ttk.Frame(self.content_frame)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(status_frame, text="Node Initialization Status:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=(0, 15))
        
        self.node_indicators = {}
        for i in range(1, 4):
            lbl = tk.Label(status_frame, text=f"Node {i}", bg="#e74c3c", fg="white", width=12, font=("Segoe UI", 10, "bold"), relief=tk.RAISED)
            lbl.pack(side=tk.LEFT, padx=5)
            self.node_indicators[i] = lbl
            
        table_frame = ttk.LabelFrame(self.content_frame, text=" Dataset Files Registry ", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        columns = ("filename", "rows", "size", "status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        self.tree.heading("filename", text="File Path")
        self.tree.heading("rows", text="Rows")
        self.tree.heading("size", text="Size (MB)")
        self.tree.heading("status", text="Status")
        
        self.tree.column("filename", width=380)
        self.tree.column("rows", width=100, anchor=tk.CENTER)
        self.tree.column("size", width=100, anchor=tk.CENTER)
        self.tree.column("status", width=150, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        action_frame = ttk.LabelFrame(self.content_frame, text=" Orchestration & Logs ", padding=10)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        load_btn = ttk.Button(action_frame, text="▶ Run data_loader.py", command=self.run_data_loader)
        load_btn.pack(anchor=tk.W, pady=(0, 10), ipady=3, ipadx=5)
        
        self.dataset_log_area = tk.Text(action_frame, height=8, width=80, state=tk.DISABLED, bg="#f8f9fa", font=("Consolas", 10), relief=tk.FLAT)
        self.dataset_log_area.pack(fill=tk.BOTH, expand=True)
        
        self.refresh_dataset_table()

    def update_node_status(self, node_id, state):
        colors = {"missing": "#e74c3c", "loading": "#f1c40f", "loaded": "#2ecc71"} 
        if node_id in self.node_indicators:
            lbl = self.node_indicators[node_id]
            bg_col = colors.get(state, "gray")
            fg_col = "black" if state == "loading" else "white"
            lbl.config(bg=bg_col, fg=fg_col)

    def log_dataset_msg(self, message):
        self.dataset_log_area.config(state=tk.NORMAL)
        self.dataset_log_area.insert(tk.END, message + "\n")
        self.dataset_log_area.see(tk.END)
        self.dataset_log_area.config(state=tk.DISABLED)

    def refresh_dataset_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        datasets_dir = config.BASE_DIR / 'datasets'
        if not datasets_dir.exists(): return
            
        processed_dir = datasets_dir / 'processed'
        
        for i in range(1, 4):
            expected_file = processed_dir / f'device{i}.csv'
            if expected_file.exists():
                self.update_node_status(i, "loaded")
            else:
                self.update_node_status(i, "missing")
                
        files_found = list(datasets_dir.rglob('*.csv'))
        for f in sorted(files_found):
            size_mb = f.stat().st_size / (1024 * 1024)
            rows = "N/A"
            if size_mb < 25: 
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        rows = sum(1 for _ in file) - 1 
                        if rows < 0: rows = 0
                except Exception:
                    pass
            elif size_mb >= 25:
                rows = "Big Data"
            
            status = "Processed" if "processed" in f.parts else "Raw Source"
            self.tree.insert("", tk.END, values=(str(f.relative_to(config.BASE_DIR)), rows, f"{size_mb:.2f}", status))

    def run_data_loader(self):
        threading.Thread(target=self._run_data_loader_thread, daemon=True).start()
        
    def _run_data_loader_thread(self):
        self.after(0, self.dataset_log_area.config, {'state': tk.NORMAL})
        self.after(0, self.dataset_log_area.delete, '1.0', tk.END)
        self.after(0, self.dataset_log_area.config, {'state': tk.DISABLED})
        
        self.after(0, self.log_dataset_msg, ">>> Initiating Subprocess: python data_loader.py")
        
        for i in range(1, 4):
            self.after(0, self.update_node_status, i, "loading")
            
        script_path = config.BASE_DIR / 'data_loader.py'
        try:
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                self.after(0, self.log_dataset_msg, line.strip())
            process.wait()
            if process.returncode == 0:
                self.after(0, self.log_dataset_msg, "\n[SUCCESS] data_loader.py finished securely.")
            else:
                self.after(0, self.log_dataset_msg, f"\n[ERROR] Process crashed with exit code {process.returncode}")
        except Exception as e:
            self.after(0, self.log_dataset_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.after(0, self.refresh_dataset_table)

    # ==============================================================================
    # 2. TRAIN MODELS VIEW
    # ==============================================================================
    def build_train_models_view(self):
        """Renders the comprehensive interactive Training execution panel."""
        
        # --- 1. Edge Node Cards Frame ---
        cards_frame = ttk.Frame(self.content_frame)
        cards_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.node_cards = {}
        
        for i in range(1, 4):
            # Dynamic Card Shell
            card = ttk.LabelFrame(cards_frame, text=f" IoT Edge Node {i} ", padding=15)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)
            
            # Status Indicator
            status_lbl = ttk.Label(card, text="Status: Idle", font=("Segoe UI", 10, "italic"))
            status_lbl.pack(anchor=tk.W, pady=(0, 8))
            
            # Progress Bar tracking Script Lifecycle
            prog = ttk.Progressbar(card, orient=tk.HORIZONTAL, mode='determinate')
            prog.pack(fill=tk.X, pady=8)
            prog['value'] = 0
            
            # F1 Metric Display
            f1_lbl = ttk.Label(card, text="F1-Score: N/A", font=("Segoe UI", 12, "bold"))
            f1_lbl.pack(anchor=tk.W, pady=(8, 0))
            
            # Cache UI references
            self.node_cards[i] = {
                "status": status_lbl,
                "progress": prog,
                "f1": f1_lbl
            }
            
        # --- 2. Action & Terminal Log Frame ---
        action_frame = ttk.LabelFrame(self.content_frame, text=" Local Orchestration & Logs ", padding=10)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        self.train_btn = ttk.Button(
            action_frame, 
            text="▶ Train All Nodes (RF + Autoencoder)", 
            command=self.run_train_models
        )
        self.train_btn.pack(anchor=tk.W, pady=(0, 10), ipady=3, ipadx=5)
        
        self.train_log_area = tk.Text(
            action_frame, height=12, width=80, state=tk.DISABLED, 
            bg="#f8f9fa", font=("Consolas", 10), relief=tk.FLAT
        )
        self.train_log_area.pack(fill=tk.BOTH, expand=True)
        
        # Load any natively cached results immediately 
        self.refresh_train_metrics()

    def log_train_msg(self, message):
        """Thread-safe injection of text into the training log area."""
        self.train_log_area.config(state=tk.NORMAL)
        self.train_log_area.insert(tk.END, message + "\n")
        self.train_log_area.see(tk.END)
        self.train_log_area.config(state=tk.DISABLED)

    def _update_node_card(self, node_id, status_text=None, progress_val=None, f1_val=None):
        """Thread-safe UI updater for specific Node tracking cards."""
        if node_id in self.node_cards:
            card = self.node_cards[node_id]
            
            if status_text is not None:
                card["status"].config(text=status_text)
                if "Completed" in status_text:
                    card["status"].config(foreground="#2ecc71") # Green
                elif "Running" in status_text:
                    card["status"].config(foreground="#f39c12") # Orange
                else:
                    card["status"].config(foreground="black")
                    
            if progress_val is not None:
                card["progress"]['value'] = progress_val
                
            if f1_val is not None:
                formatted_f1 = f"F1-Score: {f1_val:.4f}" if isinstance(f1_val, float) else f"F1-Score: {f1_val}"
                card["f1"].config(text=formatted_f1)

    def refresh_train_metrics(self):
        """Reads the underlying JSON results securely to restore state across App restarts."""
        rf_metrics_path = config.RESULTS_PATH / 'local_rf_metrics.json'
        if rf_metrics_path.exists():
            try:
                with open(rf_metrics_path, 'r') as f:
                    rf_data = json.load(f)
                    
                for i in range(1, 4):
                    node_key = f"node_{i}"
                    if node_key in rf_data:
                        f1 = rf_data[node_key].get('f1_score', 'N/A')
                        self._update_node_card(i, "Status: Completed", 100, f1)
            except Exception:
                pass

    def run_train_models(self):
        """Triggers local DL models isolated into a protected backend Thread."""
        self.train_btn.state(['disabled'])
        threading.Thread(target=self._run_training_thread, daemon=True).start()
        
    def _run_training_thread(self):
        # Flush the UI Log cleanly
        self.after(0, self.train_log_area.config, {'state': tk.NORMAL})
        self.after(0, self.train_log_area.delete, '1.0', tk.END)
        self.after(0, self.train_log_area.config, {'state': tk.DISABLED})
        
        # Reset visual node states
        for i in range(1, 4):
            self.after(0, self._update_node_card, i, "Status: Pending", 0, "N/A")
            
        # We sequentially fire RF and then Deep Autoencoder scripts
        scripts = ['local_rf_model.py', 'local_autoencoder.py']
        
        for script in scripts:
            self.after(0, self.log_train_msg, f"\n{'='*60}\n>>> Executing: python {script}\n{'='*60}")
            
            script_path = config.BASE_DIR / script
            try:
                process = subprocess.Popen(
                    [sys.executable, '-u', str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Base progress jumps by 50% per major script completion
                base_progress = 0 if script == 'local_rf_model.py' else 50
                model_name = "Random Forest" if script == 'local_rf_model.py' else "Deep Autoencoder"
                
                for line in iter(process.stdout.readline, ''):
                    clean_line = line.strip()
                    self.after(0, self.log_train_msg, clean_line)
                    
                    # Intercept output heuristically to animate Node cards tracking live execution
                    lower_line = clean_line.lower()
                    for node_id in range(1, 4):
                        if f"node {node_id}" in lower_line:
                            self.after(0, self._update_node_card, node_id, f"Running: {model_name}", base_progress + 25)
                        
                process.wait()
                
                if process.returncode != 0:
                    self.after(0, self.log_train_msg, f"\n[ERROR] {script} crashed with exit code {process.returncode}")
                    
            except Exception as e:
                self.after(0, self.log_train_msg, f"[FATAL EXCEPTION] {str(e)}")
                
        self.after(0, self.log_train_msg, "\n>>> [SUCCESS] All local models and thresholds generated perfectly.")
        
        # Hydrate JSON back into cards natively
        self.after(0, self.refresh_train_metrics)
        self.after(0, lambda: self.train_btn.state(['!disabled']))

    # ==============================================================================
    # 3. FEDERATION VIEW
    # ==============================================================================
    def build_federation_view(self):
        # Top Frame: Controls
        control_frame = ttk.Frame(self.content_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.pqc_enabled_var = tk.BooleanVar(value=True)
        self.pqc_checkbox = ttk.Checkbutton(
            control_frame, 
            text="Enable Kyber-512 PQC Encryption", 
            variable=self.pqc_enabled_var
        )
        self.pqc_checkbox.pack(side=tk.LEFT, padx=(0, 15))
        
        self.start_fed_btn = ttk.Button(
            control_frame, 
            text="▶ Start Federation", 
            command=self.start_federation
        )
        self.start_fed_btn.pack(side=tk.LEFT)
        
        # ── Live Federation Metrics ─────────────────────────────────────
        metrics_frame = ttk.LabelFrame(
            self.content_frame, text=" Live Federation Metrics ", padding=12
        )
        metrics_frame.pack(fill=tk.X, pady=(0, 15))

        # Row 0 — Round counter (full width)
        self.round_label = ttk.Label(
            metrics_frame, text="Round — / —",
            font=("Segoe UI", 12, "bold"), foreground=self.accent
        )
        self.round_label.grid(row=0, column=0, columnspan=5, sticky=tk.W, pady=(0, 8))

        # Row 1 — five metric tiles
        tile_cfg = [
            ("Global Accuracy", "N/A",  "#89b4fa", "accuracy_label"),
            ("Global F1-Score", "N/A",  "#a6e3a1", "f1_label"),
            ("Global Loss",     "N/A",  "#f38ba8", "loss_label"),
            ("DP ε (budget)",   "N/A",  "#cba6f7", "epsilon_label"),
            ("Reputation ⚠",   "N/A",  "#fab387", "flagged_label"),
        ]
        for col, (title, init_val, colour, attr) in enumerate(tile_cfg):
            tile = ttk.Frame(metrics_frame, padding=6)
            tile.grid(row=1, column=col, padx=(0, 12), sticky=tk.W)

            ttk.Label(
                tile, text=title,
                font=("Segoe UI", 9), foreground=self.fg_sub
            ).pack(anchor=tk.W)

            val_lbl = tk.Label(
                tile, text=init_val,
                font=("Segoe UI", 14, "bold"),
                fg=colour, bg=self.bg_main, padx=4
            )
            val_lbl.pack(anchor=tk.W)
            setattr(self, attr, val_lbl)

        # Epsilon badge — colour-coded privacy risk indicator
        self.epsilon_badge = tk.Label(
            metrics_frame,
            text="● STRONG", font=("Segoe UI", 8, "bold"),
            fg="white", bg="#2ecc71", padx=6, pady=2, relief=tk.FLAT
        )
        self.epsilon_badge.grid(row=2, column=3, sticky=tk.W, pady=(2, 0))

        eps_hint = ttk.Label(
            metrics_frame,
            text="ε<1 strong  |  1≤ε<10 moderate  |  ε≥10 weak",
            font=("Segoe UI", 8), foreground=self.fg_sub
        )
        eps_hint.grid(row=2, column=4, sticky=tk.W, padx=(4, 0), pady=(2, 0))


        # Visual Frame
        viz_frame = ttk.LabelFrame(self.content_frame, text=" Secure Architecture Visualization ", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(viz_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Configure>", self._draw_canvas)
        
        self.lines = []
        self.is_federating = False
        
        self.fed_log_area = tk.Text(self.content_frame, height=10, width=80, state=tk.DISABLED, bg=self.bg_main, fg=self.fg_main, font=("Consolas", 10), relief=tk.FLAT)
        self.fed_log_area.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

    def log_fed_msg(self, message):
        if hasattr(self, 'fed_log_area') and self.fed_log_area.winfo_exists():
            self.fed_log_area.config(state=tk.NORMAL)
            self.fed_log_area.insert(tk.END, message + "\n")
            self.fed_log_area.see(tk.END)
            self.fed_log_area.config(state=tk.DISABLED)
        
    def _draw_canvas(self, event=None):
        if not hasattr(self, 'canvas') or not self.canvas.winfo_exists():
            return
        
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        if w < 10: w = 800
        if h < 10: h = 400
        
        sx, sy = w // 2, 80
        self.canvas.create_rectangle(sx-80, sy-30, sx+80, sy+30, fill="#3498db", outline="black", width=2)
        self.canvas.create_text(sx, sy, text="FL Global Server", fill="white", font=("Segoe UI", 11, "bold"))
        
        num_nodes = 3
        spacing = w // (num_nodes + 1)
        ny = h - 80
        
        self.lines = []
        for i in range(1, num_nodes + 1):
            nx = i * spacing
            
            line = self.canvas.create_line(sx, sy+30, nx, ny-30, fill="gray", width=4, dash=(5, 5))
            self.lines.append(line)
            
            self.canvas.create_rectangle(nx-60, ny-30, nx+60, ny+30, fill="#2ecc71", outline="black", width=2)
            self.canvas.create_text(nx, ny, text=f"Edge Node {i}", fill="white", font=("Segoe UI", 11, "bold"))

    def animate_lines(self):
        if not self.is_federating:
            if hasattr(self, 'lines'):
                for line in self.lines:
                    self.canvas.itemconfig(line, fill="gray")
            return
            
        color = "#e74c3c" if int(time.time() * 2) % 2 == 0 else "#f1c40f"
        if hasattr(self, 'lines'):
            for line in self.lines:
                self.canvas.itemconfig(line, fill=color)
            
        self.after(500, self.animate_lines)

    def start_federation(self):
        self.start_fed_btn.state(['disabled'])
        self.is_federating = True

        # Reset all metric tiles
        self.round_label.config(text="Round — / —")
        for attr, placeholder in [
            ('accuracy_label', 'N/A'),
            ('f1_label',       'N/A'),
            ('loss_label',     'N/A'),
            ('epsilon_label',  'N/A'),
            ('flagged_label',  'N/A'),
        ]:
            if hasattr(self, attr):
                getattr(self, attr).config(text=placeholder)
        if hasattr(self, 'epsilon_badge'):
            self.epsilon_badge.config(text="● STARTING", bg="#6c757d")
        
        metrics_file = config.RESULTS_PATH / "fl_round_metrics.csv"
        if metrics_file.exists():
            try:
                metrics_file.unlink()
            except:
                pass
                
        self.animate_lines()
        self.poll_federation_metrics()
        
        threading.Thread(target=self._federation_thread, daemon=True).start()
        
    def _federation_thread(self):
        script_path = config.BASE_DIR / 'federated' / 'launch_federation.py'
        env = os.environ.copy()
        env['PQC_ENABLED'] = '1' if self.pqc_enabled_var.get() else '0'
        
        try:
            self.after(0, self.fed_log_area.config, {'state': tk.NORMAL})
            self.after(0, self.fed_log_area.delete, '1.0', tk.END)
            self.after(0, self.fed_log_area.config, {'state': tk.DISABLED})
            self.after(0, self.log_fed_msg, ">>> Executing: python federated/launch_federation.py")
            
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            for line in iter(process.stdout.readline, ''):
                self.after(0, self.log_fed_msg, line.strip())
            process.wait()
            if process.returncode != 0:
                self.after(0, self.log_fed_msg, f"\n[ERROR] Process crashed with exit code {process.returncode}")
            else:
                self.after(0, self.log_fed_msg, "\n[SUCCESS] Federation complete.")
        except Exception as e:
            self.after(0, self.log_fed_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.is_federating = False
        self.after(0, lambda: self.start_fed_btn.state(['!disabled']))
        self.after(0, self._reset_lines)
        
    def _reset_lines(self):
        if hasattr(self, 'lines'):
            for line in self.lines:
                self.canvas.itemconfig(line, fill="gray")

    def poll_federation_metrics(self):
        if not self.is_federating:
            self._read_metrics_file()
            return
            
        self._read_metrics_file()
        self.after(2000, self.poll_federation_metrics)
        
    def _read_metrics_file(self):
        """Parse fl_round_metrics.csv and update all live metric tiles."""
        metrics_file = config.RESULTS_PATH / "fl_round_metrics.csv"
        if not metrics_file.exists():
            return
        try:
            with open(metrics_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows   = [r for r in reader if any(v.strip() for v in r.values())]
            if not rows:
                return

            last = rows[-1]
            # Normalise keys: strip whitespace, lower-case
            last = {k.strip().lower(): v.strip() for k, v in last.items()}

            # ── Round counter ──────────────────────────────────────────────────
            rnd_raw = next((v for k, v in last.items() if 'round' in k), None)
            total   = getattr(config, 'FL_ROUNDS', 10)
            if rnd_raw:
                self.round_label.config(text=f"Round {rnd_raw} / {total}")

            # ── Helper: safe float from dict ──────────────────────────────────
            def _f(keys, fmt='.4f'):
                for k in keys:
                    v = last.get(k, '')
                    if v:
                        try:
                            return format(float(v), fmt)
                        except ValueError:
                            return v
                return 'N/A'

            # ── Accuracy ──────────────────────────────────────────────────
            acc_val = _f(['global_accuracy', 'accuracy'])
            if hasattr(self, 'accuracy_label'):
                self.accuracy_label.config(text=acc_val)

            # ── F1-Score ─────────────────────────────────────────────────
            f1_val = _f(['global_f1_score', 'f1_score', 'f1'])
            if hasattr(self, 'f1_label'):
                self.f1_label.config(text=f1_val)

            # ── Loss ───────────────────────────────────────────────────────
            loss_val = _f(['global_loss', 'loss'], '.4f')
            if hasattr(self, 'loss_label'):
                self.loss_label.config(text=loss_val)

            # ── DP Epsilon ───────────────────────────────────────────────
            eps_raw = next(
                (last[k] for k in last if 'epsilon' in k and last[k]), None
            )
            if hasattr(self, 'epsilon_label') and eps_raw:
                try:
                    eps_f = float(eps_raw)
                    self.epsilon_label.config(text=f'{eps_f:.4f}')

                    # Colour-coded badge
                    if hasattr(self, 'epsilon_badge'):
                        if eps_f < 1.0:
                            badge_col, badge_txt = '#2ecc71', '● STRONG  (ε<1)'
                        elif eps_f < 10.0:
                            badge_col, badge_txt = '#f39c12', '● MODERATE (1≤ε<10)'
                        else:
                            badge_col, badge_txt = '#e74c3c', '● WEAK     (ε≥10)'
                        self.epsilon_badge.config(text=badge_txt, bg=badge_col)
                except ValueError:
                    self.epsilon_label.config(text=eps_raw)

            # ── Clients flagged by reputation system ─────────────────────
            flagged_raw = next(
                (last[k] for k in last if 'flag' in k and last[k]), None
            )
            if hasattr(self, 'flagged_label') and flagged_raw:
                try:
                    n = int(float(flagged_raw))
                    colour = '#e74c3c' if n > 0 else '#2ecc71'
                    self.flagged_label.config(text=str(n), fg=colour)
                except ValueError:
                    self.flagged_label.config(text=flagged_raw)

        except Exception:
            pass


    # ==============================================================================
    # 4. CRYPTO VIEW
    # ==============================================================================
    def build_crypto_view(self):
        cards_frame = ttk.Frame(self.content_frame)
        cards_frame.pack(fill=tk.X, pady=(0, 20))
        
        rsa_card = ttk.LabelFrame(cards_frame, text=" Classical: RSA-2048 ", padding=15)
        rsa_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        ttk.Label(rsa_card, text="Key Size: 256 Bytes", font=("Segoe UI", 10)).pack(anchor=tk.W)
        ttk.Label(rsa_card, text="Security Level: 112-bit Equivalent", font=("Segoe UI", 10)).pack(anchor=tk.W)
        ttk.Label(rsa_card, text="Quantum Safe: No", font=("Segoe UI", 10)).pack(anchor=tk.W)
        lbl_rsa_safe = tk.Label(rsa_card, text="VULNERABLE", bg="#e74c3c", fg="white", font=("Segoe UI", 12, "bold"), padx=5, pady=2)
        lbl_rsa_safe.pack(anchor=tk.E, pady=(5,0))

        pqc_card = ttk.LabelFrame(cards_frame, text=" PQC: Kyber512 (NTRU-like) ", padding=15)
        pqc_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(pqc_card, text="Key Size: ~800 Bytes", font=("Segoe UI", 10)).pack(anchor=tk.W)
        ttk.Label(pqc_card, text="Security Level: AES-128 Equivalent", font=("Segoe UI", 10)).pack(anchor=tk.W)
        ttk.Label(pqc_card, text="Quantum Safe: Yes", font=("Segoe UI", 10)).pack(anchor=tk.W)
        lbl_pqc_safe = tk.Label(pqc_card, text="QUANTUM SAFE", bg="#2ecc71", fg="white", font=("Segoe UI", 12, "bold"), padx=5, pady=2)
        lbl_pqc_safe.pack(anchor=tk.E, pady=(5,0))

        action_frame = ttk.Frame(self.content_frame)
        action_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.run_crypto_btn = ttk.Button(
            action_frame, 
            text="▶ Run Crypto Benchmark", 
            command=self.run_crypto_benchmark
        )
        self.run_crypto_btn.pack(side=tk.LEFT)
        
        self.crypto_status_lbl = ttk.Label(action_frame, text="", font=("Segoe UI", 10, "italic"))
        self.crypto_status_lbl.pack(side=tk.LEFT, padx=(15, 0))

        self.chart_frame = ttk.LabelFrame(self.content_frame, text=" Benchmark Results (Timings) ", padding=10)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.display_crypto_chart()

    def run_crypto_benchmark(self):
        self.run_crypto_btn.state(['disabled'])
        self.crypto_status_lbl.config(text="Running benchmarks... (This may take a moment)")
        
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
            
        threading.Thread(target=self._crypto_benchmark_thread, daemon=True).start()
        
    def _crypto_benchmark_thread(self):
        script_path = config.BASE_DIR / 'crypto' / 'crypto_benchmark.py'
        try:
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in iter(process.stdout.readline, ''):
                pass
            process.wait()
        except Exception as e:
            pass
            
        self.after(0, lambda: self.crypto_status_lbl.config(text="Benchmark complete!"))
        self.after(0, lambda: self.run_crypto_btn.state(['!disabled']))
        self.after(0, self.display_crypto_chart)

    def display_crypto_chart(self):
        json_path = config.RESULTS_PATH / 'crypto_benchmark.json'
        
        if not json_path.exists():
            return
            
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
                
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
                
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            pqc_data = results.get("PQC (Kyber512)", {})
            rsa_data = results.get("Classical (RSA-2048)", {})
            
            pqc_times = [
                pqc_data.get('keygen_time_ms_mean', 0),
                pqc_data.get('encrypt_time_ms_mean', 0)
            ]
            rsa_times = [
                rsa_data.get('keygen_time_ms_mean', 0),
                rsa_data.get('encrypt_time_ms_mean', 0)
            ]
            
            labels = ['Key Generation', 'Encryption']
            x = np.arange(len(labels))
            width = 0.35
            
            ax.bar(x - width/2, pqc_times, width, label='PQC (Kyber512)', color='#2ecc71')
            ax.bar(x + width/2, rsa_times, width, label='Classical (RSA-2048)', color='#e74c3c')
            
            ax.set_ylabel('Time (ms)')
            ax.set_title('PQC vs Classical Operations Time')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            pass

    # ==============================================================================
    # 5. RESULTS VIEW
    # ==============================================================================
    def build_results_view(self):
        fig_frame = ttk.LabelFrame(self.content_frame, text=" Publication Figures Viewer ", padding=10)
        fig_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        control_frame = ttk.Frame(fig_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Select Figure:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.figure_var = tk.StringVar()
        self.figure_map = {
            "Confusion Matrix": "fig1_confusion_matrix.png",
            "ROC Curve": "fig2_roc_curves.png",
            "F1 Comparison": "fig3_f1_comparison.png",
            "Crypto Benchmark": "fig4_crypto_overhead.png",
            "Convergence Curve": "fig5_convergence_curve.png"
        }
        
        self.fig_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.figure_var, 
            values=list(self.figure_map.keys()),
            state="readonly",
            width=25
        )
        self.fig_combo.pack(side=tk.LEFT)
        self.fig_combo.bind("<<ComboboxSelected>>", self.display_selected_figure)
        
        self.img_label = tk.Label(fig_frame, text="No figure selected or figure missing.", bg="#f8f9fa", relief=tk.SUNKEN)
        self.img_label.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.LabelFrame(self.content_frame, text=" Benchmark Comparison Metrics ", padding=10)
        table_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.metrics_tree = ttk.Treeview(table_frame, show="headings", height=4)
        self.metrics_tree.pack(fill=tk.X)
        self.load_metrics_table()
        
        export_frame = ttk.Frame(self.content_frame)
        export_frame.pack(fill=tk.X)
        
        self.export_btn = ttk.Button(
            export_frame, 
            text="📥 Export All Results", 
            command=self.export_results
        )
        self.export_btn.pack(side=tk.LEFT)
        self.export_status = ttk.Label(export_frame, text="", font=("Segoe UI", 10, "italic"))
        self.export_status.pack(side=tk.LEFT, padx=(15, 0))

    def display_selected_figure(self, event=None):
        selection = self.figure_var.get()
        if not selection:
            return
            
        filename = self.figure_map.get(selection)
        fig_path = config.RESULTS_PATH / 'paper_figures' / filename
        
        if not fig_path.exists():
            fig_path = config.RESULTS_PATH / filename
            
        if not fig_path.exists():
            self.img_label.config(image='', text=f"Figure not found: {filename}")
            return
            
        try:
            img = Image.open(fig_path)
            resample_method = getattr(Image, 'Resampling', Image).LANCZOS
            img.thumbnail((700, 500), resample_method)
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text="")
            self.img_label.image = photo
        except Exception as e:
            self.img_label.config(image='', text=f"Error loading image: {str(e)}")

    def load_metrics_table(self):
        csv_path = config.RESULTS_PATH / 'benchmark_comparison.csv'
        if not csv_path.exists():
            self.metrics_tree["columns"] = ("status",)
            self.metrics_tree.heading("status", text="Status")
            self.metrics_tree.insert("", "end", values=("benchmark_comparison.csv not found",))
            return
            
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return
                    
                self.metrics_tree["columns"] = header
                for col in header:
                    self.metrics_tree.heading(col, text=col)
                    self.metrics_tree.column(col, width=120, anchor=tk.CENTER)
                    
                for row in reader:
                    self.metrics_tree.insert("", "end", values=row)
        except Exception as e:
            pass

    def export_results(self):
        source_dir = config.RESULTS_PATH
        if not source_dir.exists():
            self.export_status.config(text="No results directory found.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = config.BASE_DIR / f'results_export_{timestamp}'
        
        try:
            shutil.copytree(source_dir, export_dir)
            self.export_status.config(text=f"Success! Copied to {export_dir.name}", foreground="#2ecc71")
        except Exception as e:
            self.export_status.config(text=f"Error exporting: {str(e)}", foreground="#e74c3c")

    # ==============================================================================
    # 6. SIMULATION VIEW
    # ==============================================================================
    def build_simulation_view(self):
        control_frame = ttk.Frame(self.content_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.sim_attack_btn = ttk.Button(
            control_frame, 
            text="🚨 Simulate Attack (Mirai)", 
            command=self.simulate_attack
        )
        self.sim_attack_btn.pack(side=tk.LEFT)
        
        self.toggle_fed_btn = ttk.Button(
            control_frame,
            text="🔄 Toggle Federation Animation",
            command=self.toggle_sim_federation
        )
        self.toggle_fed_btn.pack(side=tk.LEFT, padx=10)
        
        viz_frame = ttk.LabelFrame(self.content_frame, text=" IoT Network Architecture & Attack Simulation ", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        self.sim_canvas = tk.Canvas(viz_frame, bg="white")
        self.sim_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.sim_canvas.bind("<Configure>", self._draw_sim_canvas)
        
        self.sim_devices = []
        self.sim_nodes = []
        self.sim_device_arrows = []
        self.sim_server_arrows = []
        self.sim_server = None
        self.sim_federating = False
        
        self.animate_sim_arrows()

    def _draw_sim_canvas(self, event=None):
        if not hasattr(self, 'sim_canvas') or not self.sim_canvas.winfo_exists():
            return
            
        self.sim_canvas.delete("all")
        w = self.sim_canvas.winfo_width()
        h = self.sim_canvas.winfo_height()
        
        if w < 10: w = 800
        if h < 10: h = 400
        
        x_dev = w * 0.15
        x_node = w * 0.5
        x_server = w * 0.85
        
        y_coords = [h * 0.2, h * 0.5, h * 0.8]
        labels = ["Camera", "Thermostat", "Doorbell"]
        
        self.sim_devices = []
        self.sim_nodes = []
        self.sim_device_arrows = []
        self.sim_server_arrows = []
        
        sy = h * 0.5
        self.sim_server = self.sim_canvas.create_rectangle(x_server-60, sy-40, x_server+60, sy+40, fill="#3498db", outline="black", width=2)
        self.sim_canvas.create_text(x_server, sy, text="FL Server", fill="white", font=("Segoe UI", 11, "bold"))
        
        for i, y in enumerate(y_coords):
            arr1 = self.sim_canvas.create_line(x_dev+50, y, x_node-60, y, arrow=tk.LAST, fill="gray", width=3, dash=(4,4))
            self.sim_device_arrows.append(arr1)
            
            arr2 = self.sim_canvas.create_line(x_node+60, y, x_server-60, sy, arrow=tk.LAST, fill="gray", width=3, dash=(4,4))
            self.sim_server_arrows.append(arr2)
            
            dev = self.sim_canvas.create_rectangle(x_dev-50, y-30, x_dev+50, y+30, fill="#bdc3c7", outline="black", width=2)
            self.sim_devices.append(dev)
            self.sim_canvas.create_text(x_dev, y, text=labels[i], fill="black", font=("Segoe UI", 10, "bold"))
            
            node = self.sim_canvas.create_rectangle(x_node-60, y-30, x_node+60, y+30, fill="#2ecc71", outline="black", width=2)
            self.sim_nodes.append(node)
            self.sim_canvas.create_text(x_node, y, text=f"Edge Node {i+1}", fill="white", font=("Segoe UI", 10, "bold"))

    def animate_sim_arrows(self):
        active = getattr(self, 'is_federating', False) or self.sim_federating
        
        if not hasattr(self, '_sim_blink_state'):
            self._sim_blink_state = False
        self._sim_blink_state = not self._sim_blink_state
        
        if not active:
            if hasattr(self, 'sim_server_arrows'):
                for arr in self.sim_server_arrows:
                    self.sim_canvas.itemconfig(arr, fill="gray")
        else:
            color = "#2ecc71" if self._sim_blink_state else "gray"
            if hasattr(self, 'sim_server_arrows'):
                for arr in self.sim_server_arrows:
                    self.sim_canvas.itemconfig(arr, fill=color)
            
        if hasattr(self, 'sim_canvas') and self.sim_canvas.winfo_exists():
            self.after(500, self.animate_sim_arrows)

    def toggle_sim_federation(self):
        self.sim_federating = not self.sim_federating

    def simulate_attack(self):
        self.sim_attack_btn.state(['disabled'])
        
        import random
        idx = random.randint(0, 2)
        dev = self.sim_devices[idx]
        node = self.sim_nodes[idx]
        arrow = self.sim_device_arrows[idx]
        
        self.sim_canvas.itemconfig(dev, fill="#e74c3c")
        self.sim_canvas.itemconfig(arrow, fill="#e74c3c")
        
        self.after(1000, lambda: self.sim_canvas.itemconfig(node, fill="#f39c12"))
        self.after(2500, lambda: self._resolve_attack(dev, node, arrow))
        
    def _resolve_attack(self, dev, node, arrow):
        if hasattr(self, 'sim_canvas') and self.sim_canvas.winfo_exists():
            self.sim_canvas.itemconfig(dev, fill="#bdc3c7")
            self.sim_canvas.itemconfig(arrow, fill="gray")
            self.sim_canvas.itemconfig(node, fill="#2ecc71")
            self.sim_attack_btn.state(['!disabled'])


    # ==============================================================================
    # 7. PREPROCESSING VIEW
    # ==============================================================================
    def build_preprocessing_view(self):
        action_frame = ttk.LabelFrame(self.content_frame, text=" Preprocessing Execution ", padding=20)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        self.prep_btn = ttk.Button(
            action_frame, 
            text="▶ Run Preprocessing Pipeline", 
            command=self.run_preprocessing
        )
        self.prep_btn.pack(anchor=tk.W, pady=(0, 15), ipady=5, ipadx=10)
        
        self.prep_log_area = tk.Text(action_frame, height=20, width=80, state=tk.DISABLED, bg=self.bg_main, fg=self.fg_main, font=("Consolas", 10), relief=tk.FLAT)
        self.prep_log_area.pack(fill=tk.BOTH, expand=True)

    def log_prep_msg(self, message):
        self.prep_log_area.config(state=tk.NORMAL)
        self.prep_log_area.insert(tk.END, message + "\n")
        self.prep_log_area.see(tk.END)
        self.prep_log_area.config(state=tk.DISABLED)

    def run_preprocessing(self):
        self.prep_btn.state(['disabled'])
        threading.Thread(target=self._run_preprocessing_thread, daemon=True).start()

    def _run_preprocessing_thread(self):
        self.after(0, self.prep_log_area.config, {'state': tk.NORMAL})
        self.after(0, self.prep_log_area.delete, '1.0', tk.END)
        self.after(0, self.prep_log_area.config, {'state': tk.DISABLED})
        
        self.after(0, self.log_prep_msg, ">>> Executing: python preprocess.py")
        script_path = config.BASE_DIR / 'preprocess.py'
        try:
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                self.after(0, self.log_prep_msg, line.strip())
            process.wait()
            if process.returncode == 0:
                self.after(0, self.log_prep_msg, "\n[SUCCESS] Preprocessing completed.")
            else:
                self.after(0, self.log_prep_msg, f"\n[ERROR] Process crashed with exit code {process.returncode}")
        except Exception as e:
            self.after(0, self.log_prep_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.after(0, lambda: self.prep_btn.state(['!disabled']))

    # ==============================================================================
    # 8. BENCHMARK VIEW
    # ==============================================================================
    def build_benchmark_view(self):
        action_frame = ttk.LabelFrame(self.content_frame, text=" Benchmark Execution ", padding=20)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.bench_btn1 = ttk.Button(btn_frame, text="▶ Run Core Benchmark", command=lambda: self.run_benchmark('benchmark/run_benchmark.py'))
        self.bench_btn1.pack(side=tk.LEFT, padx=(0, 10), ipady=5, ipadx=10)
        
        self.bench_btn2 = ttk.Button(btn_frame, text="▶ Run Cross-Dataset Test", command=lambda: self.run_benchmark('benchmark/cross_dataset_test.py'))
        self.bench_btn2.pack(side=tk.LEFT, ipady=5, ipadx=10)
        
        self.bench_log_area = tk.Text(action_frame, height=20, width=80, state=tk.DISABLED, bg=self.bg_main, fg=self.fg_main, font=("Consolas", 10), relief=tk.FLAT)
        self.bench_log_area.pack(fill=tk.BOTH, expand=True)

    def log_bench_msg(self, message):
        self.bench_log_area.config(state=tk.NORMAL)
        self.bench_log_area.insert(tk.END, message + "\n")
        self.bench_log_area.see(tk.END)
        self.bench_log_area.config(state=tk.DISABLED)

    def run_benchmark(self, script_rel_path):
        self.bench_btn1.state(['disabled'])
        self.bench_btn2.state(['disabled'])
        threading.Thread(target=self._run_benchmark_thread, args=(script_rel_path,), daemon=True).start()

    def _run_benchmark_thread(self, script_rel_path):
        self.after(0, self.bench_log_area.config, {'state': tk.NORMAL})
        self.after(0, self.bench_log_area.delete, '1.0', tk.END)
        self.after(0, self.bench_log_area.config, {'state': tk.DISABLED})
        
        self.after(0, self.log_bench_msg, f">>> Executing: python {script_rel_path}")
        script_path = config.BASE_DIR / script_rel_path
        try:
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                self.after(0, self.log_bench_msg, line.strip())
            process.wait()
            if process.returncode == 0:
                self.after(0, self.log_bench_msg, "\n[SUCCESS] Benchmark completed.")
            else:
                self.after(0, self.log_bench_msg, f"\n[ERROR] Process crashed with exit code {process.returncode}")
        except Exception as e:
            self.after(0, self.log_bench_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.after(0, lambda: self.bench_btn1.state(['!disabled']))
        self.after(0, lambda: self.bench_btn2.state(['!disabled']))

    # ==============================================================================
    # 9. PAPER FIGURES VIEW
    # ==============================================================================
    def build_figures_view(self):
        action_frame = ttk.LabelFrame(self.content_frame, text=" Paper Figures Generation ", padding=20)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig_btn = ttk.Button(
            action_frame, 
            text="▶ Generate Paper Figures", 
            command=self.run_figures
        )
        self.fig_btn.pack(anchor=tk.W, pady=(0, 15), ipady=5, ipadx=10)
        
        self.fig_log_area = tk.Text(action_frame, height=20, width=80, state=tk.DISABLED, bg=self.bg_main, fg=self.fg_main, font=("Consolas", 10), relief=tk.FLAT)
        self.fig_log_area.pack(fill=tk.BOTH, expand=True)

    def log_fig_msg(self, message):
        self.fig_log_area.config(state=tk.NORMAL)
        self.fig_log_area.insert(tk.END, message + "\n")
        self.fig_log_area.see(tk.END)
        self.fig_log_area.config(state=tk.DISABLED)

    def run_figures(self):
        self.fig_btn.state(['disabled'])
        threading.Thread(target=self._run_figures_thread, daemon=True).start()

    def _run_figures_thread(self):
        self.after(0, self.fig_log_area.config, {'state': tk.NORMAL})
        self.after(0, self.fig_log_area.delete, '1.0', tk.END)
        self.after(0, self.fig_log_area.config, {'state': tk.DISABLED})
        
        self.after(0, self.log_fig_msg, ">>> Executing: python results/generate_paper_figures.py")
        script_path = config.BASE_DIR / 'results' / 'generate_paper_figures.py'
        try:
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                self.after(0, self.log_fig_msg, line.strip())
            process.wait()
            if process.returncode == 0:
                self.after(0, self.log_fig_msg, "\n[SUCCESS] Figures generated perfectly.")
            else:
                self.after(0, self.log_fig_msg, f"\n[ERROR] Process crashed with exit code {process.returncode}")
        except Exception as e:
            self.after(0, self.log_fig_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.after(0, lambda: self.fig_btn.state(['!disabled']))

    # ==============================================================================
    # 10. PLACEHOLDER VIEW (Default Route)
    # ==============================================================================
    def build_placeholder_view(self, view_name):

        action_frame = ttk.LabelFrame(self.content_frame, text=" Module Execution Context ", padding=20)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        execute_btn = ttk.Button(
            action_frame, 
            text=f"▶ Run {view_name} Process", 
            command=lambda: self.execute_placeholder_script(view_name)
        )
        execute_btn.pack(anchor=tk.W, pady=(0, 15), ipady=5, ipadx=10)
        
        self.log_area = tk.Text(action_frame, height=20, width=80, state=tk.DISABLED, bg="#f8f9fa", font=("Consolas", 10), relief=tk.FLAT)
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def execute_placeholder_script(self, view_name):
        self.set_status(f"Executing {view_name} script in background...")
        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete('1.0', tk.END)
        self.log_area.config(state=tk.DISABLED)
        
        self.log_message(f"--- Booting {view_name} Pipeline ---")
        self.log_message(f"[INFO] This is a placeholder UI.")
        self.log_message(f"[INFO] We will connect this to the {view_name} python script shortly.")
        self.log_message(f"[SUCCESS] Test simulation finished.")
        self.set_status("Ready.")

if __name__ == "__main__":
    app = PQCDashboard()
    app.mainloop()
