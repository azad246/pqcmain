import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import config
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Canonical attack-type label names (spec §1)
# Only labels actually present in the data are used at evaluation time.
# ---------------------------------------------------------------------------
ATTACK_LABELS = [
    'benign',
    'mirai_scan',
    'mirai_udpflooding',
    'mirai_ackflooding',
    'mirai_synflooding',
    'mirai_junk',
    'bashlite_scan',
    'bashlite_junk',
    'bashlite_tcp',
    'bashlite_udp',
    'bashlite_combo',
]

# Limit TensorFlow GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

def ensemble_predict(node_id, X_test):
    """
    Combines Random Forest and Autoencoder predictions using the requested logic.
    Returns binary predictions for fair comparison across all three approaches.
    """
    # 1. Load Models
    rf_path = config.MODEL_SAVE_PATH / f'rf_node{node_id}.pkl'
    ae_path = config.MODEL_SAVE_PATH / f'ae_node{node_id}.keras'
    
    if not rf_path.exists() or not ae_path.exists():
        raise FileNotFoundError(f"Missing models for Node {node_id}. Run local_rf_model.py and local_autoencoder.py first.")
        
    rf_model = joblib.load(rf_path)
    ae_model = load_model(ae_path)
    
    # 2. Load AE Threshold — prefer thresholds.json (written by local_autoencoder.py),
    #    fall back to local_ae_metrics.json for backwards compatibility.
    thresh_path = config.RESULTS_PATH / 'thresholds.json'
    ae_path_fallback = config.RESULTS_PATH / 'local_ae_metrics.json'
    if thresh_path.exists():
        with open(thresh_path, 'r') as f:
            ae_threshold = json.load(f)[f'node{node_id}']
    elif ae_path_fallback.exists():
        with open(ae_path_fallback, 'r') as f:
            ae_metrics = json.load(f)
        ae_threshold = ae_metrics[f'node_{node_id}']['threshold']
    else:
        raise FileNotFoundError(
            f"No AE threshold file found. "
            f"Run local_autoencoder.py to generate thresholds.json."
        )
    
    # 3. Get RF Predictions
    rf_multi_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)
    # Convert multi-class RF prediction to binary (0=benign, 1=attack)
    rf_binary = (rf_multi_pred > 0).astype(int)
    # Probability of being an attack is 1.0 minus the probability of being class 0 (benign)
    rf_attack_prob = 1.0 - rf_prob[:, 0]
    
    # 4. Get AE Predictions
    ae_reconstructed = ae_model.predict(X_test, verbose=0)
    mse = np.mean(np.power(X_test - ae_reconstructed, 2), axis=1)
    ae_binary = (mse > ae_threshold).astype(int)
    
    # 5. Combine using Voting Logic
    ensemble_binary = np.zeros_like(rf_binary)
    for i in range(len(rf_binary)):
        if rf_binary[i] == 1 and ae_binary[i] == 1:
            # Both agree it is attack -> flag as attack
            ensemble_binary[i] = 1
        elif rf_binary[i] != ae_binary[i]:
            # Only one flags -> use RF decision
            ensemble_binary[i] = rf_binary[i]
        else:
            # Both agree it is benign -> flag as benign
            ensemble_binary[i] = 0
            
    # Note: Mathematically, this specific logic defaults to always trusting the RF model.
    
    return rf_binary, ae_binary, ensemble_binary, rf_attack_prob

def get_binary_metrics(y_true, y_pred, y_prob=None):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    else:
        # If no probability is provided (like raw AE predictions), calculate AUC from strict binary classes
        auc = roc_auc_score(y_true, y_pred)
        
    return {
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4)
    }

def run_ensemble_evaluation():
    print("="*80)
    print("Starting Ensemble Model Evaluation on Test Sets (Binary Anomaly Detection)")
    print("="*80)

    all_results         = {}   # overall binary metrics per node
    per_node_attack_metrics = {}   # per-class breakdown (spec §2)

    for node_id in range(1, config.NUM_NODES + 1):
        print(f"\n--- Evaluating Node {node_id} ---")
        test_path = (
            config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
            / f'node{node_id}_test.csv'
        )

        if not test_path.exists():
            print(f"[ERROR] Test data for Node {node_id} not found.")
            continue

        df = pd.read_csv(test_path, low_memory=False)

        # Normalise string labels to lowercase so they match ATTACK_LABELS
        if df['label'].dtype == object:
            df['label'] = df['label'].str.strip().str.lower()

        X_test        = df.drop(columns=['label']).values
        y_test_raw    = df['label'].values          # original labels (may be str or int)
        y_test_binary = (df['label'] != 'benign').astype(int).values \
            if df['label'].dtype == object \
            else (df['label'] > 0).astype(int).values

        try:
            rf_binary, ae_binary, ensemble_binary, rf_attack_prob = ensemble_predict(
                node_id, X_test
            )

            metrics_rf  = get_binary_metrics(y_test_binary, rf_binary, rf_attack_prob)
            metrics_ae  = get_binary_metrics(y_test_binary, ae_binary)
            metrics_ens = get_binary_metrics(y_test_binary, ensemble_binary, rf_attack_prob)

            all_results[f'node_{node_id}'] = {
                'RF':       metrics_rf,
                'AE':       metrics_ae,
                'Ensemble': metrics_ens,
            }

            # ── Binary summary table ─────────────────────────────────────
            print(f"{'Model':<12} | {'Precision':<10} | {'Recall':<10} | "
                  f"{'F1-Score':<10} | {'AUC-ROC':<10}")
            print("-" * 65)
            for label, m in [('RF Alone', metrics_rf),
                              ('AE Alone', metrics_ae),
                              ('Ensemble', metrics_ens)]:
                print(f"{label:<12} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | "
                      f"{m['f1_score']:<10.4f} | {m['auc_roc']:<10.4f}")

            # ── Per-attack-type classification report (spec §1) ──────────
            # Use the RF model's multi-class predictions against raw labels.
            rf_model   = joblib.load(config.MODEL_SAVE_PATH / f'rf_node{node_id}.pkl')
            y_pred_raw = rf_model.predict(X_test)   # multi-class predictions

            # Determine which ATTACK_LABELS are actually present
            present_in_true = set(str(v) for v in y_test_raw)
            present_in_pred = set(str(v) for v in y_pred_raw)
            present_labels  = [
                lbl for lbl in ATTACK_LABELS
                if lbl in present_in_true or lbl in present_in_pred
            ]
            # Fall back: if labels are numeric, use them directly
            if not present_labels:
                present_labels = sorted(
                    set(y_test_raw.tolist()) | set(y_pred_raw.tolist()),
                    key=lambda x: int(x) if str(x).isdigit() else 0,
                )

            report_str = classification_report(
                y_test_raw, y_pred_raw,
                labels=present_labels,
                target_names=present_labels,
                zero_division=0,
                digits=4,
            )
            report_dict = classification_report(
                y_test_raw, y_pred_raw,
                labels=present_labels,
                target_names=present_labels,
                zero_division=0,
                output_dict=True,
            )

            print(f"\n  [Node {node_id}] Per-Attack-Type Breakdown (RF predictions):")
            print(report_str)

            # Build the {attack_type: {precision, recall, f1}} structure (spec §2)
            skip = {'accuracy', 'macro avg', 'weighted avg'}
            node_attack_metrics = {}
            for attack, m in report_dict.items():
                if attack in skip or not isinstance(m, dict):
                    continue
                node_attack_metrics[attack] = {
                    'precision': round(m.get('precision', 0.0), 4),
                    'recall':    round(m.get('recall',    0.0), 4),
                    'f1':        round(m.get('f1-score',  0.0), 4),
                    'support':   int(m.get('support',     0)),
                }
            per_node_attack_metrics[f'node{node_id}'] = node_attack_metrics

        except Exception as e:
            print(f"[ERROR] Failed to evaluate Node {node_id}: {e}")

    # ── Save ensemble_metrics.json ───────────────────────────────────────
    if all_results:
        results_dir = config.RESULTS_PATH
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / 'ensemble_metrics.json'
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n[OK] Ensemble binary metrics saved → {out_path}")

    # ── Save per_attack_metrics.json (spec §2) ────────────────────────────
    # Structure: {"node1": {"mirai_scan": {"precision":x, "recall":x, "f1":x}, ...}, ...}
    if per_node_attack_metrics:
        results_dir = config.RESULTS_PATH
        results_dir.mkdir(parents=True, exist_ok=True)
        atk_path = results_dir / 'per_attack_metrics.json'
        with open(atk_path, 'w') as f:
            json.dump(per_node_attack_metrics, f, indent=4)
        print(f"[OK] Per-attack-type metrics saved → {atk_path}")


if __name__ == "__main__":
    run_ensemble_evaluation()
