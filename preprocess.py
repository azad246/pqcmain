"""
preprocess.py
=============
Merged from preprocess.py (standard) + preprocess_fast.py (fast).

Standard mode  — applies MinMaxScaler, uses a fixed GLOBAL_MAPPING so all
                 nodes have identical output class indices (needed for FL).
Fast mode      — skips scaling, encodes labels dynamically per-node, faster
                 but less rigorous for federated alignment.

Usage:
    python preprocess.py               # standard (default)
    python preprocess.py --mode fast   # fast
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import config

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Shared global label mapping (standard mode — keeps all nodes aligned for FL)
# ---------------------------------------------------------------------------
GLOBAL_MAPPING = {
    "benign":        0,
    "gafgyt.combo":  1,
    "gafgyt.junk":   2,
    "gafgyt.scan":   3,
    "gafgyt.tcp":    4,
    "gafgyt.udp":    5,
    "mirai.ack":     6,
    "mirai.scan":    7,
    "mirai.syn":     8,
    "mirai.udp":     9,
    "mirai.udpplain": 10,
}


# ==============================================================================
# STANDARD MODE  (original preprocess.py logic)
# ==============================================================================

def encode_labels_standard(df: pd.DataFrame):
    """Strip device prefixes and apply the unified GLOBAL_MAPPING."""
    df['label'] = df['label'].apply(lambda x: x.split('.', 1)[1] if '.' in x else x)
    df['label_encoded'] = df['label'].map(GLOBAL_MAPPING)
    df = df.dropna(subset=['label_encoded'])
    df['label_encoded'] = df['label_encoded'].astype(int)
    return df, GLOBAL_MAPPING


def preprocess_node(device_id: int):
    """Standard preprocessing: MinMaxScaled, globally-aligned labels."""
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    file_path = processed_dir / f'device{device_id}.csv'

    if not file_path.exists():
        print(f"[WARNING] Processed file not found: {file_path}. Skipping Node {device_id}.")
        return

    print(f"\n{'='*50}")
    print(f"Preprocessing Node {device_id} Data  [standard mode]")
    print(f"{'='*50}")

    df = pd.read_csv(file_path, low_memory=False)

    # 1. Encode Labels
    df, label_mapping = encode_labels_standard(df)
    print(f"Label Mapping: {label_mapping}")

    nodes_dir = processed_dir / 'nodes'
    nodes_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = nodes_dir / f'node{device_id}_label_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=4)

    # 2. Separate Features and Target
    y = df['label_encoded']
    drop_cols = ['label', 'label_encoded', 'device_id']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=[np.number])
    print(f"Initial numeric features: {X.shape[1]}")
    X = X.fillna(0)

    # 3. VarianceThreshold bypassed for FL feature-dimension alignment
    print(f"Features remaining (VarianceThreshold bypassed for FL alignment): {X.shape[1]}")

    # 4. MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    final_df = X_scaled.copy()
    final_df['label'] = y.values

    # 5. Stratified Split (70-15-15)
    _split_and_save(final_df, device_id, nodes_dir, verbose=True)
    print(f"[OK] Node {device_id} preprocessing complete.")


def run_preprocessing():
    """Run standard preprocessing for all nodes."""
    print("Starting Preprocessing Pipeline...")
    for device_id in range(1, config.NUM_NODES + 1):
        preprocess_node(device_id)


# ==============================================================================
# FAST MODE  (original preprocess_fast.py logic)
# ==============================================================================

def encode_labels_fast(df: pd.DataFrame):
    """Encode labels dynamically per-node with benign always == 0."""
    unique_labels = sorted(df['label'].unique().tolist())
    if 'benign' in unique_labels:
        unique_labels.remove('benign')
        unique_labels = ['benign'] + unique_labels
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_encoded'] = df['label'].map(mapping)
    return df, mapping


def preprocess_node_fast(device_id: int):
    """Fast preprocessing: no scaler, per-node label encoding."""
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    file_path = processed_dir / f'device{device_id}.csv'

    if not file_path.exists():
        print(f"[WARNING] Processed file not found: {file_path}")
        return

    print(f"\n{'='*50}")
    print(f"Preprocessing Node {device_id}  [fast mode]")
    print(f"{'='*50}")

    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} rows")

    df, label_mapping = encode_labels_fast(df)
    print(f"Label Mapping: {label_mapping}")

    nodes_dir = processed_dir / 'nodes'
    nodes_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = nodes_dir / f'node{device_id}_label_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=4)

    y = df['label_encoded']
    drop_cols = ['label', 'label_encoded', 'device_id']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=[np.number])
    X = X.fillna(0)
    print(f"Features: {X.shape[1]}")

    final_df = X.copy()
    final_df['label'] = y.values

    _split_and_save(final_df, device_id, nodes_dir, verbose=False)
    print(f"✓ Node {device_id} preprocessing complete!")


def run_fast_preprocessing():
    """Run fast preprocessing for all nodes."""
    print("=" * 60)
    print("Starting Fast Preprocessing Pipeline")
    print("=" * 60)
    for dev_id in range(1, config.NUM_NODES + 1):
        preprocess_node_fast(dev_id)
    print("\n[OK] All nodes preprocessed.")


# ==============================================================================
# SHARED HELPER
# ==============================================================================

def _split_and_save(final_df: pd.DataFrame, device_id: int, nodes_dir: Path, verbose: bool = False):
    """Stratified 70/15/15 split with graceful fallback, then save CSVs."""
    val_test_ratio = config.VAL_SIZE + config.TEST_SIZE
    test_ratio_of_temp = config.TEST_SIZE / val_test_ratio

    try:
        train_df, temp_df = train_test_split(
            final_df, test_size=val_test_ratio,
            stratify=final_df['label'], random_state=config.RANDOM_SEED
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio_of_temp,
            stratify=temp_df['label'], random_state=config.RANDOM_SEED
        )
    except Exception as e:
        print(f"[WARNING] Stratified split failed ({e}), using random split.")
        train_df, temp_df = train_test_split(final_df, test_size=val_test_ratio, random_state=config.RANDOM_SEED)
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio_of_temp, random_state=config.RANDOM_SEED)

    if verbose:
        for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            print(f"\n--- {name} Set ---\nShape: {df.shape}")
            print("Class Distribution:")
            print(df['label'].value_counts().sort_index())
    else:
        print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

    train_df.to_csv(nodes_dir / f'node{device_id}_train.csv', index=False)
    val_df.to_csv(nodes_dir   / f'node{device_id}_val.csv',   index=False)
    test_df.to_csv(nodes_dir  / f'node{device_id}_test.csv',  index=False)


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument(
        '--mode',
        choices=['standard', 'fast'],
        default='standard',
        help="standard = MinMaxScaler + GLOBAL_MAPPING (default) | fast = no scaler, per-node encoding"
    )
    args = parser.parse_args()

    if args.mode == 'fast':
        run_fast_preprocessing()
    else:
        run_preprocessing()
