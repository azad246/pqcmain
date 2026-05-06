#!/usr/bin/env python3
"""
Quick Test Script for PQC-IoT Dashboard Features
Tests all three new features without running full federation
"""

import csv
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

try:
    import config
except ImportError:
    print("ERROR: Could not import config.py")
    sys.exit(1)

def create_test_metrics():
    """Generate sample fl_round_metrics.csv for testing."""
    metrics_file = config.RESULTS_PATH / "fl_round_metrics.csv"
    
    print(f"Creating test metrics at: {metrics_file}")
    
    # Ensure directory exists
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    test_data = [
        {
            "round": 1,
            "global_accuracy": 0.8500,
            "global_f1_score": 0.8450,
            "global_loss": 0.4200,
            "epsilon": 0.1500,
            "node1_verified": "True",
            "node2_verified": "True",
            "node3_verified": "True",
            "flagged_clients": 0
        },
        {
            "round": 2,
            "global_accuracy": 0.8620,
            "global_f1_score": 0.8580,
            "global_loss": 0.3900,
            "epsilon": 0.3200,
            "node1_verified": "True",
            "node2_verified": "True",
            "node3_verified": "False",  # ← Signature failure!
            "flagged_clients": 1
        },
        {
            "round": 3,
            "global_accuracy": 0.8750,
            "global_f1_score": 0.8710,
            "global_loss": 0.3600,
            "epsilon": 0.5100,
            "node1_verified": "True",
            "node2_verified": "False",  # ← Another failure!
            "node3_verified": "True",
            "flagged_clients": 1
        },
        {
            "round": 4,
            "global_accuracy": 0.8820,
            "global_f1_score": 0.8790,
            "global_loss": 0.3350,
            "epsilon": 0.7200,
            "node1_verified": "True",
            "node2_verified": "True",
            "node3_verified": "True",
            "flagged_clients": 0
        },
        {
            "round": 5,
            "global_accuracy": 0.8900,
            "global_f1_score": 0.8870,
            "global_loss": 0.3150,
            "epsilon": 0.9500,  # ← Approaching limit
            "node1_verified": "False",
            "node2_verified": "True",
            "node3_verified": "True",
            "flagged_clients": 1
        },
        {
            "round": 6,
            "global_accuracy": 0.8950,
            "global_f1_score": 0.8920,
            "global_loss": 0.2950,
            "epsilon": 1.0500,  # ← Exceeded limit! (RED)
            "node1_verified": "True",
            "node2_verified": "True",
            "node3_verified": "False",
            "flagged_clients": 2
        },
    ]
    
    try:
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_data[0].keys())
            writer.writeheader()
            writer.writerows(test_data)
        
        print(f"✓ Created {len(test_data)} test rounds")
        print(f"✓ File: {metrics_file}")
        return True
    except Exception as e:
        print(f"✗ Error creating metrics: {e}")
        return False

def display_test_data():
    """Display the test data that was created."""
    metrics_file = config.RESULTS_PATH / "fl_round_metrics.csv"
    
    if not metrics_file.exists():
        print("✗ Metrics file not found")
        return False
    
    print("\n" + "="*100)
    print("TEST DATA CREATED")
    print("="*100)
    
    try:
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Display header
        print(f"\n📊 Total Rounds: {len(rows)}\n")
        print(f"{'Round':<6} {'Epsilon':<12} {'Node1_Verified':<16} {'Node2_Verified':<16} {'Node3_Verified':<16} {'Notes':<20}")
        print("-" * 100)
        
        # Display rows with analysis
        for row in rows:
            rnd = row['round']
            eps = float(row['epsilon'])
            n1 = row['node1_verified'] == 'True'
            n2 = row['node2_verified'] == 'True'
            n3 = row['node3_verified'] == 'True'
            
            # Determine privacy status
            if eps < 0.5:
                eps_status = "🟢 GREEN"
            elif eps < 1.0:
                eps_status = "🟡 AMBER"
            else:
                eps_status = "🔴 RED"
            
            # Determine verification status
            rejections = []
            if not n1: rejections.append("Node1")
            if not n2: rejections.append("Node2")
            if not n3: rejections.append("Node3")
            
            notes = f"{eps_status} | " + (f"REJECTED: {', '.join(rejections)}" if rejections else "All verified")
            
            print(f"{rnd:<6} {eps:<12.4f} {str(n1):<16} {str(n2):<16} {str(n3):<16} {notes:<20}")
        
        print("\n" + "="*100)
        return True
    except Exception as e:
        print(f"✗ Error reading metrics: {e}")
        return False

def test_privacy_budget():
    """Test Feature 1: Privacy Budget Display."""
    print("\n" + "="*100)
    print("FEATURE 1: PRIVACY BUDGET DISPLAY TEST")
    print("="*100)
    
    print("""
Expected Behavior:
  1. Privacy budget bar in Federation panel should:
     - Show BLUE canvas with colored progress rectangle
     - Color progression: GREEN (ε<0.5) → AMBER (0.5≤ε<1.0) → RED (ε≥1.0)
  2. Label shows: "Privacy budget used: ε=X.XX / 1.0"
  3. Updates every 2 seconds during federation polling

Test Steps:
  1. Click "Federation" tab
  2. Click "▶ Start Federation" (if not already running)
  3. Watch the progress bar:
     - Round 1: ε=0.15 → GREEN (15% filled)
     - Round 2: ε=0.32 → GREEN (32% filled)
     - Round 3: ε=0.51 → AMBER (51% filled)
     - Round 4: ε=0.72 → AMBER (72% filled)
     - Round 5: ε=0.95 → AMBER (95% filled)
     - Round 6: ε=1.05 → RED (100% filled - capped)

Validation Checklist:
  □ Progress bar appears with colored fill
  □ Color changes from green to amber at ε=0.5
  □ Color changes from amber to red at ε=1.0
  □ Label updates correctly with epsilon value
  □ Bar grows smoothly as epsilon increases
  □ Updates coincide with federation metrics updates (2 sec interval)
""")
    
    print("Status: Feature 1 ready for manual testing ✓")

def test_signature_verification():
    """Test Feature 2: Signature Verification Status."""
    print("\n" + "="*100)
    print("FEATURE 2: SIGNATURE VERIFICATION STATUS TEST")
    print("="*100)
    
    print("""
Expected Behavior:
  1. Three rows in Crypto panel showing Node 1, 2, 3
  2. Each node shows:
     - ✓ VERIFIED (Green text, #2ecc71) - Signature passed
     - ✗ REJECTED (Red text, #e74c3c) - Signature failed
     - ○ Monitoring (Gray text) - No data yet
  3. Updates every 5 seconds during federation polling

Test Steps with Sample Data:
  Round 1: All verified (✓✓✓)
  Round 2: Node 3 rejected (✓✓✗) → Simulation flashes Node 3
  Round 3: Node 2 rejected (✓✗✓) → Simulation flashes Node 2
  Round 4: All verified (✓✓✓)
  Round 5: Node 1 rejected (✗✓✓) → Simulation flashes Node 1
  Round 6: Node 3 rejected (✓✓✗) → Simulation flashes Node 3

Validation Checklist:
  □ All 3 nodes show initial "○ Monitoring" status
  □ Icons update every 5 seconds (not 2 like metrics!)
  □ Green (✓) and Red (✗) colors display correctly
  □ Verified nodes show consistent checkmark
  □ Rejected nodes trigger Simulation flash (see Feature 3)
  □ Status updates correlate with CSV node{i}_verified values
""")
    
    print("Status: Feature 2 ready for manual testing ✓")

def test_rejection_flash():
    """Test Feature 3: Signature Rejection Flash."""
    print("\n" + "="*100)
    print("FEATURE 3: SIGNATURE REJECTION FLASH ANIMATION TEST")
    print("="*100)
    
    print("""
Expected Behavior:
  1. When signature fails (node{i}_verified = False):
     - Simulation canvas node box turns RED (#e74c3c)
     - "🔴 SIGNATURE REJECTED" text appears
     - After 2 seconds: Node returns to GREEN (#2ecc71)
     - Text disappears
  2. Only one node flashes at a time (prevents overlapping)
  3. Can be triggered by:
     - Real signature failure during federation
     - "🔴 Test Signature Rejection" button for demo

Test Steps (Manual):
  1. Click "Simulation" tab
  2. Click "🔴 Test Signature Rejection" button
  3. Observe:
     - Random node flashes RED for 2 seconds
     - "🔴 SIGNATURE REJECTED" text appears
     - Node returns to GREEN
     - Text disappears
  4. Repeat to test different nodes

Test Steps (Auto with Federation):
  1. Click "Federation" tab
  2. Click "▶ Start Federation"
  3. Watch Simulation canvas:
     - Round 2: Node 3 flashes (from test data)
     - Round 3: Node 2 flashes
     - Round 5: Node 1 flashes
     - Round 6: Node 3 flashes again

Validation Checklist:
  □ Test button works and triggers flash on random node
  □ Node color changes: GREEN → RED → GREEN
  □ "🔴 SIGNATURE REJECTED" text appears
  □ Text disappears after flash
  □ Flash duration is exactly 2 seconds
  □ Multiple rejection flashes work correctly
  □ Simulation line animation continues during flash
  □ No errors in console output
""")
    
    print("Status: Feature 3 ready for manual testing ✓")

def run_all_tests():
    """Run all feature tests."""
    print("\n" + "="*100)
    print("PQC-IoT DASHBOARD FEATURES - QUICK TEST SUITE")
    print("="*100)
    
    print("\n🔧 Preparing test environment...")
    
    # Create test metrics
    if not create_test_metrics():
        print("✗ Failed to create test metrics")
        return False
    
    # Display test data
    if not display_test_data():
        print("✗ Failed to display test data")
        return False
    
    # Run feature tests
    test_privacy_budget()
    test_signature_verification()
    test_rejection_flash()
    
    # Final instructions
    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("""
1. LAUNCH DASHBOARD:
   python dashboard/main_dashboard.py

2. NAVIGATE TO FEDERATION TAB:
   - Click "Federation" in sidebar
   - Click "▶ Start Federation"

3. TEST FEATURE 1 (Privacy Budget):
   - Watch the progress bar grow from GREEN → AMBER → RED
   - Should align with epsilon values in test data

4. TEST FEATURE 2 (Signature Verification):
   - Switch to "Crypto" tab
   - Watch node status update: ✓ → ✗ → ✓
   - Mismatches trigger Feature 3 (flash animation)

5. TEST FEATURE 3 (Rejection Flash):
   - Switch to "Simulation" tab
   - Click "🔴 Test Signature Rejection"
   - Watch random node flash RED with overlay text

6. VALIDATE WITH SAMPLE DATA:
   - Test data has predictable rejection sequence:
     Round 2: Node 3 fails
     Round 3: Node 2 fails
     Round 5: Node 1 fails
     Round 6: Node 3 fails

7. VIEW RESULTS:
   - All metrics, signatures, and flashes should sync
   - No console errors
   - UI remains responsive

📊 Test Data File: {results_path}
📖 Integration Guide: INTEGRATION_GUIDE.md
📝 Code Changes: CODE_MODIFICATIONS.md
""".format(results_path=config.RESULTS_PATH / "fl_round_metrics.csv"))
    
    print("="*100)
    print("✓ Test suite prepared! Launch dashboard to begin testing.")
    print("="*100)
    
    return True

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
