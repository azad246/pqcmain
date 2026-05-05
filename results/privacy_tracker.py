import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_CSV = BASE_DIR / "results" / "fl_round_metrics.csv"
FIGURES_DIR = BASE_DIR / "results" / "paper_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUT_PATH = FIGURES_DIR / "privacy_budget.png"

# DP Constants
DP_LIMIT = 1.0
DP_DELTA = 1e-5

def main():
    if not METRICS_CSV.exists():
        print(f"[ERROR] Could not find metrics file: {METRICS_CSV}")
        sys.exit(1)

    try:
        df = pd.read_csv(METRICS_CSV)
    except Exception as e:
        print(f"[ERROR] Failed to read {METRICS_CSV}: {e}")
        sys.exit(1)

    if "Epsilon" not in df.columns:
        print("[ERROR] 'Epsilon' column not found in the CSV. Make sure FL server logs epsilon.")
        sys.exit(1)

    # ── 1 & 2. Compute cumulative epsilon ────────────────────────────────────
    rounds = df["Round"].values
    epsilons = df["Epsilon"].values
    
    # Calculate difference to get actual per-round spent budget
    # The 'Epsilon' logged is already cumulative from Opacus,
    # so per-round budget spent is the diff between consecutive rounds.
    # If Opacus returns cumulative directly, then `epsilons` is cumulative.
    # Let's verify: Opacus get_epsilon() returns cumulative epsilon. 
    # Therefore, the Epsilon column is ALREADY cumulative.
    # To get per-round epsilon, we do diff().
    cumulative_epsilons = epsilons
    per_round_epsilons = np.insert(np.diff(cumulative_epsilons), 0, cumulative_epsilons[0])

    # ── 4 & 5. Prints ───────────────────────────────────────────────────────
    total_spent = cumulative_epsilons[-1] if len(cumulative_epsilons) > 0 else 0.0
    total_rounds = len(rounds)
    
    limit_reached_round = "Never"
    for r, c_eps in zip(rounds, cumulative_epsilons):
        if c_eps > DP_LIMIT:
            limit_reached_round = str(int(r))
            break

    print("=" * 60)
    print("  PQC-IoT Sentinel — Differential Privacy Tracker")
    print("=" * 60)
    print(f"Total privacy budget spent after {total_rounds} rounds: ε={total_spent:.4f}")
    print(f"Privacy limit (ε={DP_LIMIT}) reached at round: {limit_reached_round}")
    print(f"Delta (δ) used: {DP_DELTA}")
    print("\nNote for Paper:")
    print(f"(ε, δ)-Differential Privacy guarantees that the presence or absence")
    print(f"of any single device's local data will not change the probability")
    print(f"of any FL model output by more than a multiplicative factor of exp(ε)")
    print(f"plus an additive probability of δ. Lower ε means tighter privacy.")
    print("=" * 60)

    # ── 3 & 6. Plotting ─────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 1: Per-round budget spent (Bar Chart)
    ax1.bar(rounds, per_round_epsilons, color='steelblue', alpha=0.8)
    ax1.set_title("Privacy Budget Spent per Round", fontsize=14, pad=10)
    ax1.set_xlabel("Federated Learning Round", fontsize=12)
    ax1.set_ylabel("ε spent", fontsize=12)
    ax1.set_xticks(rounds)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Chart 2: Cumulative budget vs limit (Line Chart)
    ax2.plot(rounds, cumulative_epsilons, marker='o', color='crimson', linewidth=2, label="Cumulative ε")
    ax2.axhline(y=DP_LIMIT, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f"Privacy Limit (ε={DP_LIMIT})")
    ax2.set_title("Cumulative Privacy Budget (ε)", fontsize=14, pad=10)
    ax2.set_xlabel("Federated Learning Round", fontsize=12)
    ax2.set_ylabel("Total ε", fontsize=12)
    ax2.set_xticks(rounds)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(PLOT_OUT_PATH, dpi=300)
    print(f"Saved privacy plot to: {PLOT_OUT_PATH}")

if __name__ == "__main__":
    main()
