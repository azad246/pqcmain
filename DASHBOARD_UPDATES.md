# PQC-IoT Sentinel Dashboard Updates

## Overview
Updated `dashboard/main_dashboard.py` with three new features:
1. **Privacy Budget Display** (Federation Panel)
2. **Signature Verification Status** (Crypto Panel)  
3. **Signature Rejection Flash Animation** (Simulation Panel)

All features use only **Tkinter + ttk** and auto-refresh every 5 seconds.

---

## Feature 1: Privacy Budget Display (Federation Panel)

### Location
`build_federation_view()` - Lines ~560-575

### What It Does
- Displays live cumulative epsilon value read from `fl_round_metrics.csv`
- Shows colored progress bar:
  - **Green** (✓) if ε < 0.5 (strong privacy)
  - **Amber** (▲) if 0.5 ≤ ε < 1.0 (moderate privacy)
  - **Red** (✗) if ε ≥ 1.0 (weak privacy)
- Label shows: "Privacy budget used: ε=X.XX / 1.0"

### CSV Column Required
In `fl_round_metrics.csv`, add column:
```
epsilon  (or: cumulative_epsilon, dp_epsilon, etc.)
```

### Code Changes
**Updated method:** `_update_privacy_budget_display()`
- Reads `self.current_epsilon` (set during `_read_metrics_file()`)
- Draws colored rectangle on Canvas-based progress bar
- Updates label with current/max epsilon

**Auto-refresh:** Called every 2 seconds by `poll_federation_metrics()` → `_read_metrics_file()`

---

## Feature 2: Signature Verification Status (Crypto Panel)

### Location
`build_crypto_view()` - Lines ~1000-1015

### What It Does
- Displays verification status for 3 nodes (Node 1, 2, 3)
- Shows icons + text:
  - **✓ VERIFIED** (Green) if signature passed
  - **✗ REJECTED** (Red) if signature failed
  - **○ Monitoring** (Gray) if no data yet
- Auto-updates every 5 seconds

### CSV Columns Required
In `fl_round_metrics.csv`, add columns:
```
node1_verified   (True/False or 1/0)
node2_verified   (True/False or 1/0)
node3_verified   (True/False or 1/0)
```

### Code Changes
**New method:** `poll_signature_status()`
- Scheduled refresh every 5 seconds via `self.after(5000, ...)`
- Calls `_read_signature_verification()`

**New method:** `_read_signature_verification()`
- Parses `fl_round_metrics.csv` for `node{i}_verified` columns
- Updates widget text/color for each node
- Triggers `_trigger_node_rejection()` if signature failed

---

## Feature 3: Signature Rejection Flash Animation (Simulation Panel)

### Location
`build_simulation_view()` - Lines ~1040-1065

### What It Does
When a signature verification fails:
1. **Flash node box RED** on the simulation canvas for 2 seconds
2. Display **"🔴 SIGNATURE REJECTED"** text overlay
3. Restore to normal green state after timeout
4. Can test with "🔴 Test Signature Rejection" button

### Code Changes
**New method:** `_trigger_node_rejection(node_id)`
- Checks `node_rejection_state` to prevent duplicate triggers
- Changes node rectangle fill to red (#e74c3c)
- Creates overlay text with rejection message
- Schedules restoration after 2 seconds

**New method:** `_restore_node_state(node_id)`
- Resets node color to green (#2ecc71)
- Deletes rejection text overlay
- Updates rejection state tracker

**New method:** `test_signature_rejection()`
- Demo function: randomly selects a node and triggers rejection flash
- Updates Crypto panel label to show "REJECTED (TEST)"

---

## Integration with Federation Flow

### Start Federation → Auto-Updates
```
start_federation()
├─ Reset privacy budget to 0.0
├─ Reset signature widgets to "○ Monitoring"
├─ poll_federation_metrics() → every 2 sec
│  ├─ _read_metrics_file() → reads epsilon
│  ├─ Updates epsilon_label, epsilon_badge
│  └─ _update_privacy_budget_display() → refreshes progress bar
├─ poll_signature_status() → every 5 sec
│  └─ _read_signature_verification() → reads node{i}_verified
│     ├─ Updates sig_status_widgets
│     └─ Calls _trigger_node_rejection() if failed
└─ _federation_thread() → runs launch_federation.py
   └─ Writes fl_round_metrics.csv after each round
```

---

## Sample fl_round_metrics.csv Format

```csv
round,global_accuracy,global_f1_score,global_loss,epsilon,node1_verified,node2_verified,node3_verified,flagged_clients
1,0.8500,0.8450,0.4200,0.1500,True,True,True,0
2,0.8620,0.8580,0.3900,0.3200,True,True,False,1
3,0.8750,0.8710,0.3600,0.5100,True,False,True,1
```

---

## Key Implementation Details

### Privacy Budget Canvas Progress Bar
```python
# Uses tk.Canvas instead of ttk.Progressbar for custom colors
self.eps_progress_canvas = tk.Canvas(pb_frame, width=400, height=20, ...)
# Draw colored rectangle based on epsilon value
progress_width = canvas_width * (epsilon / 1.0)
self.eps_progress_canvas.create_rectangle(
    0, 0, progress_width, 20, fill=color, outline=color
)
```

### Signature Status Widgets Storage
```python
# Dictionary keyed by node_id (1, 2, 3)
self.sig_status_widgets = {
    1: <tk.Label>,
    2: <tk.Label>,
    3: <tk.Label>
}
# Updated via: self.sig_status_widgets[i].config(text="✓ VERIFIED", fg="#2ecc71")
```

### Simulation Flash Mechanism
```python
# Store rejection state per node
self.node_rejection_state = {1: False, 2: False, 3: False}
# Prevent duplicate simultaneous rejections
if self.node_rejection_state.get(node_id, False):
    return
# Schedule restoration after 2 seconds
self.after(2000, lambda: self._restore_node_state(node_id))
```

---

## Testing

### Manual Test Steps
1. **Privacy Budget**: Manually edit `results/fl_round_metrics.csv` with epsilon values → watch progress bar update
2. **Signature Status**: Edit `node{i}_verified` columns to True/False → verify Crypto panel updates
3. **Rejection Flash**: Click "🔴 Test Signature Rejection" button → see node flash red briefly

### Expected Behavior
- Privacy bar turns **green** → **amber** → **red** as epsilon increases
- Signature icons show **✓ (green)** or **✗ (red)** based on CSV
- Simulation node flashes **red** with **"SIGNATURE REJECTED"** overlay for 2 seconds

---

## Color Scheme

| Feature | Component | Green | Amber | Red |
|---------|-----------|-------|-------|-----|
| Privacy Budget | ε < 0.5 | ✓ | | |
| Privacy Budget | 0.5 ≤ ε < 1.0 | | ✓ | |
| Privacy Budget | ε ≥ 1.0 | | | ✓ |
| Signature Status | Verified | ✓ VERIFIED (Green) | | |
| Signature Status | Failed | | | ✗ REJECTED (Red) |
| Simulation | Normal | Node #2ecc71 | | |
| Simulation | Rejection | | | Node #e74c3c |

---

## Notes
- All refresh intervals are configurable (currently 2sec for metrics, 5sec for signatures)
- Privacy budget max is hardcoded to 1.0 (configurable via `self.epsilon_budget`)
- Rejection flash duration is 2 seconds (configurable in `_trigger_node_rejection()`)
- Features are auto-disabled when federation is not running

