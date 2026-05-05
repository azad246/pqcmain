"""
dashboard/patch_utils.py
========================
One-shot patch scripts previously kept in scratch/.
Run these manually if you need to retroactively apply patches
to main_dashboard.py (e.g. after a clean checkout).

Usage:
    python dashboard/patch_utils.py --patch views
    python dashboard/patch_utils.py --patch fed_logs
    python dashboard/patch_utils.py --patch all
"""

import argparse
from pathlib import Path

DASHBOARD_FILE = Path(__file__).parent / 'main_dashboard.py'


# ==============================================================================
# PATCH 1 — Add Preprocessing / Benchmark / Paper-Figures views
# (Originally: scratch/update_dashboard.py)
# ==============================================================================

NEW_VIEWS = '''    # ==============================================================================
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
        self.prep_log_area.insert(tk.END, message + "\\n")
        self.prep_log_area.see(tk.END)
        self.prep_log_area.config(state=tk.DISABLED)

    def run_preprocessing(self):
        self.prep_btn.state([\'disabled\'])
        threading.Thread(target=self._run_preprocessing_thread, daemon=True).start()

    def _run_preprocessing_thread(self):
        self.after(0, self.prep_log_area.config, {\'state\': tk.NORMAL})
        self.after(0, self.prep_log_area.delete, \'1.0\', tk.END)
        self.after(0, self.prep_log_area.config, {\'state\': tk.DISABLED})
        
        self.after(0, self.log_prep_msg, ">>> Executing: python preprocess.py")
        script_path = config.BASE_DIR / \'preprocess.py\'
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            for line in iter(process.stdout.readline, \'\'):
                self.after(0, self.log_prep_msg, line.strip())
            process.wait()
            if process.returncode == 0:
                self.after(0, self.log_prep_msg, "\\n[SUCCESS] Preprocessing completed.")
            else:
                self.after(0, self.log_prep_msg, f"\\n[ERROR] Process crashed with exit code {process.returncode}")
        except Exception as e:
            self.after(0, self.log_prep_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.after(0, lambda: self.prep_btn.state([\'!disabled\']))

    # ==============================================================================
    # 8. BENCHMARK VIEW
    # ==============================================================================
    def build_benchmark_view(self):
        action_frame = ttk.LabelFrame(self.content_frame, text=" Benchmark Execution ", padding=20)
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.bench_btn1 = ttk.Button(btn_frame, text="▶ Run Core Benchmark", command=lambda: self.run_benchmark(\'benchmark/run_benchmark.py\'))
        self.bench_btn1.pack(side=tk.LEFT, padx=(0, 10), ipady=5, ipadx=10)
        
        self.bench_btn2 = ttk.Button(btn_frame, text="▶ Run Cross-Dataset Test", command=lambda: self.run_benchmark(\'benchmark/cross_dataset_test.py\'))
        self.bench_btn2.pack(side=tk.LEFT, ipady=5, ipadx=10)
        
        self.bench_log_area = tk.Text(action_frame, height=20, width=80, state=tk.DISABLED, bg=self.bg_main, fg=self.fg_main, font=("Consolas", 10), relief=tk.FLAT)
        self.bench_log_area.pack(fill=tk.BOTH, expand=True)

    def log_bench_msg(self, message):
        self.bench_log_area.config(state=tk.NORMAL)
        self.bench_log_area.insert(tk.END, message + "\\n")
        self.bench_log_area.see(tk.END)
        self.bench_log_area.config(state=tk.DISABLED)

    def run_benchmark(self, script_rel_path):
        self.bench_btn1.state([\'disabled\'])
        self.bench_btn2.state([\'disabled\'])
        threading.Thread(target=self._run_benchmark_thread, args=(script_rel_path,), daemon=True).start()

    def _run_benchmark_thread(self, script_rel_path):
        self.after(0, self.bench_log_area.config, {\'state\': tk.NORMAL})
        self.after(0, self.bench_log_area.delete, \'1.0\', tk.END)
        self.after(0, self.bench_log_area.config, {\'state\': tk.DISABLED})
        
        self.after(0, self.log_bench_msg, f">>> Executing: python {script_rel_path}")
        script_path = config.BASE_DIR / script_rel_path
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            for line in iter(process.stdout.readline, \'\'):
                self.after(0, self.log_bench_msg, line.strip())
            process.wait()
            if process.returncode == 0:
                self.after(0, self.log_bench_msg, "\\n[SUCCESS] Benchmark completed.")
            else:
                self.after(0, self.log_bench_msg, f"\\n[ERROR] Process crashed with exit code {process.returncode}")
        except Exception as e:
            self.after(0, self.log_bench_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.after(0, lambda: self.bench_btn1.state([\'!disabled\']))
        self.after(0, lambda: self.bench_btn2.state([\'!disabled\']))

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
        self.fig_log_area.insert(tk.END, message + "\\n")
        self.fig_log_area.see(tk.END)
        self.fig_log_area.config(state=tk.DISABLED)

    def run_figures(self):
        self.fig_btn.state([\'disabled\'])
        threading.Thread(target=self._run_figures_thread, daemon=True).start()

    def _run_figures_thread(self):
        self.after(0, self.fig_log_area.config, {\'state\': tk.NORMAL})
        self.after(0, self.fig_log_area.delete, \'1.0\', tk.END)
        self.after(0, self.fig_log_area.config, {\'state\': tk.DISABLED})
        
        self.after(0, self.log_fig_msg, ">>> Executing: python results/generate_paper_figures.py")
        script_path = config.BASE_DIR / \'results\' / \'generate_paper_figures.py\'
        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            for line in iter(process.stdout.readline, \'\'):
                self.after(0, self.log_fig_msg, line.strip())
            process.wait()
            if process.returncode == 0:
                self.after(0, self.log_fig_msg, "\\n[SUCCESS] Figures generated perfectly.")
            else:
                self.after(0, self.log_fig_msg, f"\\n[ERROR] Process crashed with exit code {process.returncode}")
        except Exception as e:
            self.after(0, self.log_fig_msg, f"[FATAL EXCEPTION] {str(e)}")
            
        self.after(0, lambda: self.fig_btn.state([\'!disabled\']))

    # ==============================================================================
    # 10. PLACEHOLDER VIEW (Default Route)
    # ==============================================================================
    def build_placeholder_view(self, view_name):
'''

_VIEWS_TARGET = '''    # ==============================================================================
    # 7. PLACEHOLDER VIEW (Default Route)
    # ==============================================================================
    def build_placeholder_view(self, view_name):'''


def patch_views(content: str) -> tuple[str, bool]:
    """Insert Preprocessing / Benchmark / Paper-Figures views."""
    if _VIEWS_TARGET in content:
        return content.replace(_VIEWS_TARGET, NEW_VIEWS), True
    return content, False


# ==============================================================================
# PATCH 2 — Add federation log area + live output streaming
# (Originally: scratch/update_dashboard_fed_logs.py)
# ==============================================================================

_FED_LOG_TARGET = '''        self.lines = []
        self.is_federating = False'''

_FED_LOG_REPLACEMENT = '''        self.lines = []
        self.is_federating = False
        
        self.fed_log_area = tk.Text(self.content_frame, height=10, width=80, state=tk.DISABLED, bg=self.bg_main, fg=self.fg_main, font=("Consolas", 10), relief=tk.FLAT)
        self.fed_log_area.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

    def log_fed_msg(self, message):
        if hasattr(self, 'fed_log_area') and self.fed_log_area.winfo_exists():
            self.fed_log_area.config(state=tk.NORMAL)
            self.fed_log_area.insert(tk.END, message + "\\n")
            self.fed_log_area.see(tk.END)
            self.fed_log_area.config(state=tk.DISABLED)'''

_FED_THREAD_TARGET = '''        try:
            process = subprocess.Popen(
                [sys.executable, '-u', str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            for line in iter(process.stdout.readline, ''):
                pass 
            process.wait()
        except Exception as e:
            pass'''

_FED_THREAD_REPLACEMENT = '''        try:
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
                self.after(0, self.log_fed_msg, f"\\n[ERROR] Process crashed with exit code {process.returncode}")
            else:
                self.after(0, self.log_fed_msg, "\\n[SUCCESS] Federation complete.")
        except Exception as e:
            self.after(0, self.log_fed_msg, f"[FATAL EXCEPTION] {str(e)}")'''


def patch_fed_logs(content: str) -> tuple[str, bool]:
    """Add federation log widget and live output streaming."""
    changed = False
    if _FED_LOG_TARGET in content:
        content = content.replace(_FED_LOG_TARGET, _FED_LOG_REPLACEMENT)
        changed = True
    if _FED_THREAD_TARGET in content:
        content = content.replace(_FED_THREAD_TARGET, _FED_THREAD_REPLACEMENT)
        changed = True
    return content, changed


# ==============================================================================
# RUNNER
# ==============================================================================

def apply_patches(patches: list[str]):
    content = DASHBOARD_FILE.read_text(encoding='utf-8')
    any_changed = False

    if 'views' in patches or 'all' in patches:
        content, ok = patch_views(content)
        print(f"[views patch]   {'Applied ✓' if ok else 'Target not found — already applied or layout changed'}")
        any_changed = any_changed or ok

    if 'fed_logs' in patches or 'all' in patches:
        content, ok = patch_fed_logs(content)
        print(f"[fed_logs patch] {'Applied ✓' if ok else 'Target not found — already applied or layout changed'}")
        any_changed = any_changed or ok

    if any_changed:
        DASHBOARD_FILE.write_text(content, encoding='utf-8')
        print(f"\n[OK] Saved → {DASHBOARD_FILE}")
    else:
        print("\n[INFO] No changes written.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patch main_dashboard.py")
    parser.add_argument(
        '--patch',
        choices=['views', 'fed_logs', 'all'],
        default='all',
        help="Which patch to apply (default: all)"
    )
    args = parser.parse_args()
    apply_patches([args.patch])
