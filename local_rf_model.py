"""
local_rf_model.py
=================
Merged from local_rf_model.py (standard) + local_rf_model_fast.py (fast).

Standard mode — reads pre-split node CSVs (from preprocess.py), trains a
                regularized RF (50 trees, depth 5), prints a full metrics table.
Fast mode     — loads the raw device CSV directly, splits on-the-fly, trains
                a regularized RF (50 trees, depth 5); quicker iteration.

Usage:
    python local_rf_model.py               # standard (default)
    python local_rf_model.py --mode fast   # fast
"""

import argparse
import json
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
import joblib
import config

warnings.filterwarnings('ignore')


# ==============================================================================
# STANDARD MODE  (original local_rf_model.py logic)
# ==============================================================================

def train_rf_node(node_id: int):
    """Train RF on pre-split node CSVs (standard mode)."""
    print(f"\n--- Training Local RF Model for Node {node_id} ---")

    nodes_dir  = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
    train_path = nodes_dir / f'node{node_id}_train.csv'
    val_path   = nodes_dir / f'node{node_id}_val.csv'

    if not train_path.exists() or not val_path.exists():
        print(f"[ERROR] Missing training or validation data for Node {node_id} at {nodes_dir}.")
        print("Please ensure preprocess.py has been run successfully.")
        return None

    print("Loading datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df   = pd.read_csv(val_path,   low_memory=False)

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_val   = val_df.drop(columns=['label'])
    y_val   = val_df['label']

    # Drop potential identifier / leakage columns before fitting
    _LEAKAGE_COLS = ['ip', 'mac', 'timestamp', 'src_ip', 'dst_ip', 'port', 'Unnamed: 0']
    X_train = X_train.drop(columns=[c for c in _LEAKAGE_COLS if c in X_train.columns])
    X_val   = X_val.drop(columns=[c for c in _LEAKAGE_COLS if c in X_val.columns])

    print("Training RandomForestClassifier(n_estimators=40, max_depth=3, extreme regularization)...")
    clf = RandomForestClassifier(
        n_estimators=40, max_depth=3,
        min_samples_split=100, min_samples_leaf=50,
        max_features='log2', max_samples=0.5,
        class_weight='balanced',
        random_state=config.RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)

    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"  [WARNING] Could not calculate AUC-ROC: {e}")
        auc = 0.0

    metrics = {
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1_score':  round(f1,        4),
        'auc_roc':   round(auc,       4),
    }
    print(f"-> Precision: {metrics['precision']:.4f}")
    print(f"-> Recall:    {metrics['recall']:.4f}")
    print(f"-> F1-Score:  {metrics['f1_score']:.4f}")
    print(f"-> AUC-ROC:   {metrics['auc_roc']:.4f}")

    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_SAVE_PATH / f'rf_node{node_id}.pkl'
    joblib.dump(clf, model_path)
    print(f"Model saved to: {model_path}")

    return metrics


def run_local_training():
    """Run standard RF training for all nodes."""
    print("=" * 60)
    print("Starting Local Model Training for PQC-IoT Sentinel Nodes")
    print("=" * 60)

    all_metrics = {}
    for node_id in range(1, config.NUM_NODES + 1):
        metrics = train_rf_node(node_id)
        if metrics:
            all_metrics[f'node_{node_id}'] = metrics

    if not all_metrics:
        print("\n[ERROR] No models were trained successfully.")
        return

    # Summary table
    print("\n" + "=" * 60)
    print("Local Random Forest Models Comparison")
    print("=" * 60)
    print(f"{'Node':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
    print("-" * 60)
    for node, m in all_metrics.items():
        name = node.replace('_', ' ').title()
        print(f"{name:<10} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['auc_roc']:<10.4f}")
    print("=" * 60)

    _save_metrics(all_metrics)


# ==============================================================================
# FAST MODE  (original local_rf_model_fast.py logic)
# ==============================================================================

def train_rf_node_direct(node_id: int):
    """Train RF directly on raw device CSV without pre-split files (fast mode)."""
    print(f"\n--- Training Local RF Model for Node {node_id} [fast mode] ---")

    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    device_file   = processed_dir / f'device{node_id}.csv'

    if not device_file.exists():
        print(f"[ERROR] Device file not found: {device_file}")
        return None

    print(f"Loading data from {device_file.name}...")
    df = pd.read_csv(device_file, low_memory=False)
    print(f"Loaded {len(df)} rows")

    # Encode labels (benign → 0)
    labels    = sorted([l for l in df['label'].unique() if l != 'benign'])
    labels    = ['benign'] + labels
    label_map = {l: i for i, l in enumerate(labels)}
    df['label_encoded'] = df['label'].map(label_map)

    # Drop potential identifier / leakage columns before selecting numeric features
    _LEAKAGE_COLS = ['ip', 'mac', 'timestamp', 'src_ip', 'dst_ip', 'port', 'Unnamed: 0']
    df = df.drop(columns=[c for c in _LEAKAGE_COLS if c in df.columns])

    X = df.drop(columns=['label', 'label_encoded', 'device_id'], errors='ignore').select_dtypes(include='number')
    y = df['label_encoded']
    X = X.fillna(0)

    print("Splitting into train/val (70/30)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=config.RANDOM_SEED
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    print("Training RF (n_estimators=40, max_depth=3, extreme regularization)...")
    clf = RandomForestClassifier(
        n_estimators=40, max_depth=3,
        min_samples_split=100, min_samples_leaf=50,
        max_features='log2', max_samples=0.5,
        class_weight='balanced',
        random_state=config.RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)

    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0

    metrics = {
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1_score':  round(f1,        4),
        'auc_roc':   round(auc,       4),
    }
    print(f"→ F1: {metrics['f1_score']:.4f}  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")

    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_SAVE_PATH / f'rf_node{node_id}.pkl'
    joblib.dump(clf, model_path)
    print(f"Model saved: {model_path}")

    return metrics


def run_fast_training():
    """Run fast RF training for all nodes."""
    print("=" * 60)
    print("Starting Local Model Training (Fast Mode)")
    print("=" * 60)

    all_metrics = {}
    for node_id in range(1, config.NUM_NODES + 1):
        metrics = train_rf_node_direct(node_id)
        if metrics:
            all_metrics[f'node_{node_id}'] = metrics

    if not all_metrics:
        print("\n[ERROR] No models trained.")
        return

    _save_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("Local RF Training Complete")
    print("=" * 60)


# ==============================================================================
# SHARED HELPER
# ==============================================================================

def _save_metrics(all_metrics: dict):
    results_path = config.RESULTS_PATH / 'local_rf_metrics.json'
    config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[OK] Aggregated metrics saved to: {results_path}")


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RF model training")
    parser.add_argument(
        '--mode',
        choices=['standard', 'fast'],
        default='standard',
        help="standard = pre-split CSVs, 100 trees (default) | fast = raw device CSV, 50 trees"
    )
    args = parser.parse_args()

    if args.mode == 'fast':
        run_fast_training()
    else:
        run_local_training()
