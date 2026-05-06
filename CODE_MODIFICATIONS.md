# Code Modifications - Quick Reference Guide

## 1. INITIALIZATION (In `__init__` method)

**Added these lines to initialize tracking variables:**

```python
# Initialize privacy budget tracking
self.current_epsilon = 0.0
self.epsilon_budget = 1.0
self.sig_refresh_id = None
self.sim_rejection_text = None
```

---

## 2. FEDERATION PANEL - Privacy Budget Display

**Location:** `build_federation_view()` method

**Key Variables Created:**
```python
# Canvas-based progress bar with custom coloring
self.eps_progress_canvas = tk.Canvas(pb_frame, width=400, height=20, bg="white", relief=tk.SUNKEN, bd=1)
self.eps_progress_rect = None  # Holds the progress rectangle

# Label showing current usage
self.eps_usage_lbl = tk.Label(pb_frame, text="Privacy budget used: ε=0.00 / 1.0", ...)
```

**New Method:** `_update_privacy_budget_display()`
```python
def _update_privacy_budget_display(self):
    """Update the privacy budget progress bar with color-coding."""
    if not hasattr(self, 'eps_progress_canvas') or not self.eps_progress_canvas.winfo_exists():
        return
    
    # Calculate progress as percentage (0-1, capped at 1.0)
    progress = min(self.current_epsilon / self.epsilon_budget, 1.0)
    
    # Determine color based on epsilon thresholds
    if self.current_epsilon < 0.5:
        color = "#2ecc71"  # Green: strong privacy
    elif self.current_epsilon < 1.0:
        color = "#f1c40f"  # Amber: moderate privacy
    else:
        color = "#e74c3c"  # Red: weak privacy
    
    # Update canvas progress rectangle
    self.eps_progress_canvas.delete("progress_rect")
    canvas_width = self.eps_progress_canvas.winfo_width()
    if canvas_width < 10:
        canvas_width = 400
    
    progress_width = canvas_width * progress
    self.eps_progress_rect = self.eps_progress_canvas.create_rectangle(
        0, 0, progress_width, 20, fill=color, outline=color, tags="progress_rect"
    )
    self.eps_progress_canvas.tag_lower("progress_rect")
    
    # Update label
    self.eps_usage_lbl.config(
        text=f"Privacy budget used: ε={self.current_epsilon:.2f} / {self.epsilon_budget:.1f}",
        fg=color
    )
```

**Updated `_read_metrics_file()` method (Added at end):**
```python
# ── Privacy Budget Update (extract epsilon for progress bar) ────────
if eps_raw:
    try:
        self.current_epsilon = float(eps_raw)
        self._update_privacy_budget_display()
    except ValueError:
        pass
```

---

## 3. CRYPTO PANEL - Signature Verification Status

**Location:** `build_crypto_view()` method

**Key Variables Created:**
```python
# Dictionary to store label widgets for each node
self.sig_status_widgets = {
    1: tk.Label(...),
    2: tk.Label(...),
    3: tk.Label(...)
}

# Track scheduled refresh to prevent duplicate callbacks
self.sig_refresh_id = None
```

**New Method:** `poll_signature_status()`
```python
def poll_signature_status(self):
    """Poll signature verification status from fl_round_metrics.csv every 5 seconds."""
    if not self.is_federating:
        return
    
    self._read_signature_verification()
    if hasattr(self, 'sig_refresh_id'):
        if self.sig_refresh_id:
            self.after_cancel(self.sig_refresh_id)
    self.sig_refresh_id = self.after(5000, self.poll_signature_status)
```

**New Method:** `_read_signature_verification()`
```python
def _read_signature_verification(self):
    """Extract signature verification status from fl_round_metrics.csv."""
    metrics_file = config.RESULTS_PATH / "fl_round_metrics.csv"
    if not metrics_file.exists():
        return
    
    try:
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if any(v.strip() for v in r.values())]
        
        if not rows:
            return
        
        last = rows[-1]
        # Normalize keys
        last = {k.strip().lower(): v.strip() for k, v in last.items()}
        
        # Check verification status for each node
        for i in range(1, 4):
            if i not in self.sig_status_widgets:
                continue
            
            # Look for node{i}_verified or node{i}_signature_verified keys
            verified_key = next(
                (k for k in last if f'node{i}' in k and 'verif' in k), None
            )
            
            if verified_key:
                value = last[verified_key].lower()
                is_verified = value in ('true', '1', 'yes')
                
                if is_verified:
                    self.sig_status_widgets[i].config(
                        text="✓ VERIFIED", 
                        fg="#2ecc71"  # Green
                    )
                else:
                    self.sig_status_widgets[i].config(
                        text="✗ REJECTED", 
                        fg="#e74c3c"  # Red
                    )
                    # Trigger simulation flash when signature fails
                    self._trigger_node_rejection(i)
            else:
                self.sig_status_widgets[i].config(
                    text="○ Monitoring", 
                    fg=self.fg_sub
                )
    except Exception:
        pass
```

**Updated `start_federation()` method (Added these lines):**
```python
# Reset signature verification status
for i in range(1, 4):
    if i in self.sig_status_widgets:
        self.sig_status_widgets[i].config(text="○ Monitoring...", fg=self.fg_sub)

# ... later in method:
self.poll_signature_status()  # Start polling signature updates
```

---

## 4. SIMULATION PANEL - Signature Rejection Flash

**Location:** `build_simulation_view()` method

**Key Variables Created:**
```python
# Track rejection states to prevent duplicate triggers
self.node_rejection_state = {1: False, 2: False, 3: False}

# Store reference to rejection text on canvas
self.sim_rejection_text = None
```

**Updated `build_simulation_view()` (Added):**
```python
# Test Signature Rejection button (for demo purposes)
self.test_sig_reject_btn = ttk.Button(
    control_frame,
    text="🔴 Test Signature Rejection",
    command=self.test_signature_rejection
)
self.test_sig_reject_btn.pack(side=tk.LEFT, padx=10)

# ... later:
self.poll_signature_failures()  # Start monitoring for signature failures
```

**New Method:** `_trigger_node_rejection(node_id)`
```python
def _trigger_node_rejection(self, node_id):
    """Flash node in red on the simulation canvas when signature fails."""
    if not hasattr(self, 'sim_nodes') or node_id - 1 >= len(self.sim_nodes):
        return
    
    # Prevent duplicate triggers
    if self.node_rejection_state.get(node_id, False):
        return
    
    self.node_rejection_state[node_id] = True
    node_rect = self.sim_nodes[node_id - 1]
    
    if hasattr(self, 'sim_canvas') and self.sim_canvas.winfo_exists():
        # Flash node red
        self.sim_canvas.itemconfig(node_rect, fill="#e74c3c")
        
        # Show "SIGNATURE REJECTED" text
        if not self.sim_rejection_text:
            y = 50 + (node_id - 1) * 100
            self.sim_rejection_text = self.sim_canvas.create_text(
                self.sim_canvas.winfo_width() * 0.5,
                y,
                text="🔴 SIGNATURE REJECTED",
                fill="#e74c3c",
                font=("Segoe UI", 12, "bold"),
                tags="rejection_text"
            )
        
        # Restore normal state after 2 seconds
        self.after(2000, lambda: self._restore_node_state(node_id))
```

**New Method:** `_restore_node_state(node_id)`
```python
def _restore_node_state(self, node_id):
    """Restore node to normal state after rejection flash."""
    if not hasattr(self, 'sim_nodes') or node_id - 1 >= len(self.sim_nodes):
        return
    
    self.node_rejection_state[node_id] = False
    node_rect = self.sim_nodes[node_id - 1]
    
    if hasattr(self, 'sim_canvas') and self.sim_canvas.winfo_exists():
        self.sim_canvas.itemconfig(node_rect, fill="#2ecc71")
        
        # Remove rejection text
        if self.sim_rejection_text:
            self.sim_canvas.delete("rejection_text")
            self.sim_rejection_text = None
```

**New Method:** `test_signature_rejection()`
```python
def test_signature_rejection(self):
    """Demo function: Simulate signature rejection on random node."""
    import random
    node_id = random.randint(1, 3)
    self._trigger_node_rejection(node_id)
    # Update the corresponding label
    if node_id in self.sig_status_widgets:
        self.sig_status_widgets[node_id].config(
            text="✗ REJECTED (TEST)", 
            fg="#e74c3c"
        )
```

---

## 5. Summary of Changes

| Section | Type | Purpose |
|---------|------|---------|
| Privacy Budget Canvas | UI Element | Custom colored progress bar |
| Privacy Budget Display | New Method | Update progress bar every poll |
| Signature Status Widgets | UI Container | Store node verification labels |
| Signature Polling | New Method | Auto-refresh every 5 seconds |
| Signature Reading | New Method | Parse CSV and update widgets |
| Rejection Flash | New Method | Trigger red flash on node |
| Rejection Restore | New Method | Reset node after 2 seconds |
| Rejection Demo Button | UI Element | Test rejection flash |

---

## Files Modified

- **Main File:** `dashboard/main_dashboard.py`
- **Created Documentation:** `DASHBOARD_UPDATES.md`
- **Sample CSV:** `results/sample_fl_round_metrics.csv`

---

## Testing Commands

```bash
# 1. Copy sample metrics to active location
cp results/sample_fl_round_metrics.csv results/fl_round_metrics.csv

# 2. Run dashboard
python dashboard/main_dashboard.py

# 3. Click "Federation" tab
# 4. Click "🔴 Test Signature Rejection" button (Crypto tab)
# 5. Watch simulation canvas flash red with "SIGNATURE REJECTED" text

# 6. Click "▶ Start Federation" to see live updates from your federation script
```

