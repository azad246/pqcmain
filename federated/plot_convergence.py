import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path so we can import config
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# Set high-quality aesthetic style for research paper inclusion
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def plot_convergence():
    csv_path = config.RESULTS_PATH / 'fl_round_metrics.csv'
    
    if not csv_path.exists():
        print(f"[ERROR] Metrics file not found at {csv_path}.")
        print("Please ensure federated/launch_federation.py has been executed successfully.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. Standardize column names safely
    col_mapping = {}
    for col in df.columns:
        if 'round' in col.lower(): col_mapping[col] = 'round'
        elif 'loss' in col.lower(): col_mapping[col] = 'loss'
        elif 'accuracy' in col.lower(): col_mapping[col] = 'accuracy'
        elif 'f1' in col.lower(): col_mapping[col] = 'f1_score'
    df = df.rename(columns=col_mapping)
    
    if 'round' not in df.columns or 'accuracy' not in df.columns or 'loss' not in df.columns:
        print(f"[ERROR] Required columns not found. Present columns: {list(df.columns)}")
        return

    # Ensure chronological order by round
    df = df.sort_values(by='round')
    
    # --- 2. Plot Accuracy Convergence ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='round', y='accuracy', marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.title('Federated Learning Convergence (Accuracy)', fontsize=16, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=12, fontweight='bold')
    plt.ylabel('Global Accuracy', fontsize=12, fontweight='bold')
    plt.xticks(df['round'].astype(int))
    plt.ylim(0, 1.05)
    
    acc_save_path = config.RESULTS_PATH / 'fl_convergence_accuracy.png'
    plt.savefig(acc_save_path)
    plt.close()
    
    # --- 3. Plot Loss Convergence ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='round', y='loss', marker='s', color='#d62728', linewidth=2.5, markersize=8)
    plt.title('Federated Learning Convergence (Loss)', fontsize=16, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=12, fontweight='bold')
    plt.ylabel('Global Cross-Entropy Loss', fontsize=12, fontweight='bold')
    plt.xticks(df['round'].astype(int))
    
    loss_save_path = config.RESULTS_PATH / 'fl_convergence_loss.png'
    plt.savefig(loss_save_path)
    plt.close()
    
    print(f"\n[OK] High-resolution plots successfully generated for research paper:")
    print(f"  -> {acc_save_path}")
    print(f"  -> {loss_save_path}")
    
    # --- 4. Print Summary Statistics & Insights ---
    print("\n" + "="*50)
    print("Federation Training Insights")
    print("="*50)
    
    final_acc = df['accuracy'].iloc[-1]
    print(f"- Final Round Accuracy: {final_acc:.4f} ({final_acc * 100:.2f}%)")
    
    if len(df) > 1:
        first_acc = df['accuracy'].iloc[0]
        improvement = final_acc - first_acc
        total_rounds_ran = int(df['round'].iloc[-1])
        print(f"- Accuracy Improvement (Round 1 to {total_rounds_ran}): +{improvement:.4f} (+{improvement * 100:.2f}%)")
    else:
        print("- Accuracy Improvement: N/A (Only 1 round recorded)")
        
    # Find rounds required to breach 90% accuracy
    rounds_above_90 = df[df['accuracy'] >= 0.90]
    if not rounds_above_90.empty:
        target_round = rounds_above_90['round'].iloc[0]
        print(f"- Rounds required to reach 90% Accuracy: {int(target_round)}")
    else:
        print("- Rounds required to reach 90% Accuracy: Model did not exceed 90% threshold.")
        
    print("="*50)

if __name__ == "__main__":
    plot_convergence()
