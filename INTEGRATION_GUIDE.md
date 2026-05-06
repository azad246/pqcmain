# Dashboard Features - Visual Integration Guide

## Complete Feature Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PQC-IoT Sentinel Dashboard                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ FEDERATION PANEL (When Running Federation)                                 │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ Round 1 / 10                                                           │ │
│ │ ┌──────────────┬───────────────┬──────────────┬──────────┬──────────┐ │ │
│ │ │ Accuracy:    │ F1-Score:     │ Loss:        │ DP ε:    │ Rep. ⚠:  │ │ │
│ │ │ 0.8500       │ 0.8450        │ 0.4200       │ 0.1500   │ 0        │ │ │
│ │ └──────────────┴───────────────┴──────────────┴──────────┴──────────┘ │ │
│ │                                                                          │ │
│ │ ● STRONG (ε<1)    ε<1 strong | 1≤ε<10 moderate | ε≥10 weak           │ │
│ │                                                                          │ │
│ │ FEATURE 1: Privacy Budget Usage:                                       │ │
│ │ ┌───────────────────────┐ Privacy budget used: ε=0.15 / 1.0           │ │
│ │ │███░░░░░░░░░░░░░░░░░││  (GREEN PROGRESS BAR)                         │ │
│ │ └───────────────────────┘                                              │ │
│ │                                                                          │ │
│ │ [FL Global Server] ↔ [Edge Node 1] ↔ [Edge Node 2] ↔ [Edge Node 3]  │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ CRYPTO PANEL (Always Available)                                            │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ ┌─ Classical: RSA-2048 ─┐  ┌─ PQC: Kyber512 ─────┐                   │ │
│ │ │ Key Size: 256 Bytes   │  │ Key Size: ~800 Bytes│                   │ │
│ │ │ Security: 112-bit     │  │ Security: AES-128   │                   │ │
│ │ │ Quantum Safe: No      │  │ Quantum Safe: Yes   │                   │ │
│ │ │      VULNERABLE       │  │   QUANTUM SAFE      │                   │ │
│ │ └───────────────────────┘  └─────────────────────┘                   │ │
│ │                                                                          │ │
│ │ FEATURE 2: Signature Verification Status (Dilithium2)                 │ │
│ │ ┌────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ Node 1: ✓ VERIFIED                (GREEN)                         │ │ │
│ │ │ Node 2: ✗ REJECTED                (RED)                           │ │ │
│ │ │ Node 3: ✓ VERIFIED                (GREEN)                         │ │ │
│ │ └────────────────────────────────────────────────────────────────────┘ │ │
│ │                                                                          │ │
│ │ [▶ Run Crypto Benchmark]                                               │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ SIMULATION PANEL (Real-Time Network Visualization)                         │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ [🚨 Simulate Attack] [🔄 Toggle Federation] [🔴 Test Sig Rejection]  │ │
│ │                                                                          │ │
│ │ ┌────────────────────────────────────────────────────────────────────┐ │ │
│ │ │     [Camera]    [Thermostat]  [Doorbell]                          │ │ │
│ │ │        ↓             ↓              ↓                             │ │ │
│ │ │     [Node 1]     [Node 2]      [Node 3]    ← Normally GREEN      │ │ │
│ │ │        ↓             ↓              ↓                             │ │ │
│ │ │              [FL Global Server]                                  │ │ │
│ │ │                                                                    │ │ │
│ │ │ FEATURE 3: Signature Rejection Flash Animation                   │ │ │
│ │ │ When signature fails (from Crypto panel):                        │ │ │
│ │ │ • Node box flashes RED for 2 seconds                             │ │ │
│ │ │ • "🔴 SIGNATURE REJECTED" text appears                           │ │ │
│ │ │ • Node returns to GREEN after timeout                            │ │ │
│ │ │ • Correlates with Crypto panel showing ✗ REJECTED               │ │ │
│ │ └────────────────────────────────────────────────────────────────────┘ │ │
│ │                                                                          │ │
│ │ [View output logs from simulation/attack detection...]                 │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Privacy Budget - Detailed Behavior

### Progress Bar Color Transitions

```
Round 1: ε = 0.15
┌─────────────────────────────────────────┐
│██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ GREEN (15% filled)
└─────────────────────────────────────────┘
Privacy budget used: ε=0.15 / 1.0

Round 3: ε = 0.50
┌─────────────────────────────────────────┐
│██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ AMBER (50% filled)
└─────────────────────────────────────────┘
Privacy budget used: ε=0.50 / 1.0

Round 8: ε = 1.21
┌─────────────────────────────────────────┐
│█████████████████████████████████████████│ RED (100% filled - exceeded)
└─────────────────────────────────────────┘
Privacy budget used: ε=1.21 / 1.0
```

### Color Thresholds
- **GREEN** (`#2ecc71`): ε < 0.5 → Strong privacy maintained
- **AMBER** (`#f1c40f`): 0.5 ≤ ε < 1.0 → Privacy moderately consumed
- **RED** (`#e74c3c`): ε ≥ 1.0 → Budget exceeded, weak privacy

---

## Feature 2: Signature Verification - Real-Time Updates

### Status Display per Node

```
Timeline of updates (5-second polling):

T=0s:  Node 1: ○ Monitoring    (Gray - No data yet)
       Node 2: ○ Monitoring    
       Node 3: ○ Monitoring    

T=5s:  Node 1: ✓ VERIFIED      (Green - Signature valid)
       Node 2: ✓ VERIFIED      
       Node 3: ✓ VERIFIED      

T=10s: Node 1: ✓ VERIFIED      
       Node 2: ✗ REJECTED      (Red - Signature failed!)
       Node 3: ✓ VERIFIED      
       
       → Triggers: _trigger_node_rejection(2)
         → Simulation canvas shows flash on Node 2
         → "🔴 SIGNATURE REJECTED" appears on canvas
         → After 2 sec: Node 2 restores to green
         → After 2 sec: Restoration complete

T=15s: Node 1: ✓ VERIFIED      
       Node 2: ✓ VERIFIED      (Recovered)
       Node 3: ✓ VERIFIED      
```

---

## Feature 3: Rejection Flash Animation - Sequence Diagram

```
User clicks "🔴 Test Signature Rejection" button
         ↓
test_signature_rejection()
         ↓
    [Pick random node, say Node 2]
         ↓
_trigger_node_rejection(2)
         ├─ Check: Is Node 2 already in rejection state?
         │  └─ No → Continue
         ├─ Set: node_rejection_state[2] = True
         ├─ Update Crypto panel: Node 2 label → "✗ REJECTED (TEST)"
         ├─ Simulation canvas:
         │  ├─ Change node rect fill: #2ecc71 → #e74c3c (RED)
         │  └─ Create text overlay: "🔴 SIGNATURE REJECTED"
         └─ Schedule: self.after(2000, _restore_node_state(2))
                          ↓
                    [Wait 2 seconds...]
                          ↓
               _restore_node_state(2)
                   ├─ Set: node_rejection_state[2] = False
                   ├─ Simulation canvas:
                   │  ├─ Change node rect fill: #e74c3c → #2ecc71 (GREEN)
                   │  └─ Delete text overlay
                   └─ Done! Ready for next rejection
```

---

## Integration with FL Federation Process

```
START FEDERATION
        ↓
start_federation()
  ├─ Reset metrics display (all fields → "N/A")
  ├─ Reset privacy budget (epsilon = 0.0)
  ├─ Reset signature widgets (→ "○ Monitoring...")
  ├─ Poll federation metrics: every 2 seconds
  │   ├─ _read_metrics_file()
  │   │   ├─ Read fl_round_metrics.csv
  │   │   ├─ Extract: round, accuracy, f1, loss
  │   │   ├─ Extract: epsilon → _update_privacy_budget_display()
  │   │   └─ Call: node{i}_verified read methods
  │   │
  │   └─ Update UI tiles (accuracy_label, f1_label, etc.)
  │
  ├─ Poll signature status: every 5 seconds
  │   ├─ _read_signature_verification()
  │   │   ├─ Extract: node1_verified, node2_verified, node3_verified
  │   │   ├─ For each node:
  │   │   │   ├─ If TRUE → "✓ VERIFIED" (Green)
  │   │   │   ├─ If FALSE → "✗ REJECTED" (Red)
  │   │   │   │           └─ Call _trigger_node_rejection()
  │   │   │   │
  │   │   └─ Update sig_status_widgets[i]
  │   │
  │   └─ Trigger flash on Simulation canvas if rejected
  │
  └─ Run federation script: launch_federation.py
      ├─ Trains models
      ├─ Writes rounds to fl_round_metrics.csv
      └─ Dashboard polls continuously
```

---

## Required CSV Structure

### fl_round_metrics.csv Columns

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `round` | int | 1 | Federation round number |
| `global_accuracy` | float | 0.85 | Aggregate model accuracy |
| `global_f1_score` | float | 0.845 | F1 score across nodes |
| `global_loss` | float | 0.42 | Loss value |
| `epsilon` | float | 0.15 | **DP cumulative epsilon** |
| `node1_verified` | bool | True | **Node 1 signature status** |
| `node2_verified` | bool | True | **Node 2 signature status** |
| `node3_verified` | bool | False | **Node 3 signature status** |
| `flagged_clients` | int | 0 | Clients flagged by reputation |

### Minimum Valid Row

```csv
round,epsilon,node1_verified,node2_verified,node3_verified
1,0.15,True,True,True
```

---

## User Interaction Flows

### Flow 1: Normal Federation Run
```
1. Click "Federation" tab
2. Click "▶ Start Federation"
3. Watch:
   - Round counter updates (Round 1/10, 2/10, ...)
   - Accuracy/F1/Loss refresh every 2 seconds
   - Privacy budget bar grows: GREEN → AMBER → RED
   - Signature status shows ✓ or ✗ every 5 seconds
4. Federation completes
5. Results persist in UI
```

### Flow 2: Testing Rejection Flash
```
1. Click "Crypto" tab
2. Click "🔴 Test Signature Rejection"
3. Random node flashes red on Simulation canvas
4. "🔴 SIGNATURE REJECTED" text appears
5. After 2 seconds: Node returns to green
6. Text disappears
7. Can repeat multiple times for different nodes
```

### Flow 3: Real Signature Failure (During Federation)
```
1. During federation, fl_round_metrics.csv gets updated
2. If node{i}_verified = False:
   - Crypto panel shows: "✗ REJECTED" (Red)
   - Simulation canvas flashes that node RED
   - "🔴 SIGNATURE REJECTED" text appears
   - After 2 sec: Restores to green
3. If signature passes next round:
   - Crypto panel updates to: "✓ VERIFIED" (Green)
```

---

## Debugging Checklist

- [ ] `fl_round_metrics.csv` exists in `results/` directory
- [ ] CSV has columns: `epsilon`, `node1_verified`, `node2_verified`, `node3_verified`
- [ ] CSV values updated correctly by your federation script
- [ ] Dashboard shows file in logs when federation starts
- [ ] Privacy bar changes color at epsilon thresholds (0.5, 1.0)
- [ ] Signature widgets update every 5 seconds when running
- [ ] Test button triggers rejection flash on Simulation canvas
- [ ] Rejection flash lasts exactly 2 seconds
- [ ] No errors in console when running

---

## Performance Considerations

| Metric | Value | Impact |
|--------|-------|--------|
| Metrics polling interval | 2 seconds | Reduces CSV I/O strain |
| Signature polling interval | 5 seconds | Balances responsiveness & load |
| Flash duration | 2 seconds | Clear but not annoying |
| Canvas refresh rate | 500 ms (animations) | Smooth federated line animation |
| CSV row limit | Auto (all rows read) | Last row used for display |

---

## Customization Guide

### Adjust Privacy Budget Thresholds
In `_update_privacy_budget_display()`:
```python
if self.current_epsilon < 0.5:        # ← Change 0.5
    color = "#2ecc71"  # Green
elif self.current_epsilon < 1.0:      # ← Change 1.0
    color = "#f1c40f"  # Amber
else:
    color = "#e74c3c"  # Red
```

### Adjust Polling Intervals
```python
# Federation metrics (currently 2 seconds)
self.after(2000, self.poll_federation_metrics)

# Signature status (currently 5 seconds)
self.sig_refresh_id = self.after(5000, self.poll_signature_status)
```

### Adjust Flash Duration
In `_trigger_node_rejection()`:
```python
# Restore normal state after X milliseconds
self.after(2000, lambda: self._restore_node_state(node_id))  # ← Change 2000
```

### Change Colors
```python
# Privacy budget
"#2ecc71" → Green
"#f1c40f" → Amber
"#e74c3c" → Red

# Signature verification
"#2ecc71" → Green (verified)
"#e74c3c" → Red (rejected)
"#a6adc8" → Gray (monitoring)
```

