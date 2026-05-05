"""
local_autoencoder.py
====================
Merged from local_autoencoder.py (standard) + local_autoencoder_fast.py (fast).

Both modes train ONLY on benign traffic (label == 0) and determine an
adaptive per-node threshold by sweeping the 85th–99th percentiles of
benign validation errors and selecting the one that maximises F1.

Standard mode — reads pre-split node CSVs, regularized AE
                (64→Dropout→32→Dropout→16→32→Dropout→64), 50 epochs
                with EarlyStopping, full precision/recall/F1 evaluation.
Fast mode     — loads raw device CSV directly, same regularized AE,
                50 epochs with EarlyStopping, saves threshold as JSON.

Usage:
    python local_autoencoder.py               # standard (default)
    python local_autoencoder.py --mode fast   # fast
"""

import argparse
import json
import warnings
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Force CPU-only mode for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
    from tensorflow.keras import regularizers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError as e:
    print(f"[ERROR] TensorFlow import failed: {e}")
    print("Please reinstall: pip install --upgrade tensorflow")
    raise

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

import config


# ==============================================================================
# STANDARD MODE  (original local_autoencoder.py logic)
# ==============================================================================

def train_autoencoder_node(node_id: int):
    """
    Standard AE training:
      - Reads pre-split node CSVs (from preprocess.py standard mode).
      - Drops identifier/leakage columns before scaling.
      - Trains on benign-only rows (label == 0).
      - Regularized AE: 64→Dropout(0.2)→32→Dropout(0.2)→16→32→Dropout(0.2)→64.
      - EarlyStopping(patience=5), up to 50 epochs.
      - Adaptive threshold: sweeps 85th–99th pct, picks best F1.
      - Full precision/recall/F1 evaluation on the complete validation set.
    """
    print(f"\n--- Training Local Autoencoder for Node {node_id} ---")

    nodes_dir  = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
    train_path = nodes_dir / f'node{node_id}_train.csv'
    val_path   = nodes_dir / f'node{node_id}_val.csv'

    if not train_path.exists() or not val_path.exists():
        print(f"[ERROR] Missing data for Node {node_id}.")
        return None

    print("Loading datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df   = pd.read_csv(val_path,   low_memory=False)

    # 1. Drop leakage / identifier columns before feature extraction
    _LEAKAGE_COLS = ['ip', 'mac', 'timestamp', 'src_ip', 'dst_ip', 'port', 'Unnamed: 0']
    train_df = train_df.drop(columns=[c for c in _LEAKAGE_COLS if c in train_df.columns])
    val_df   = val_df.drop(columns=[c for c in _LEAKAGE_COLS if c in val_df.columns])

    # 2. Train on BENIGN ONLY (label == 0)
    benign_train_df = train_df[train_df['label'] == 0]
    X_train_benign  = benign_train_df.drop(columns=['label']).values.astype('float32')
    print(f"Total training samples: {len(train_df)}. Benign samples used for AE: {len(benign_train_df)}")

    if len(benign_train_df) == 0:
        print(f"[ERROR] No benign training samples for Node {node_id}.")
        return None

    # 3. Denoising autoencoder with heavy regularization:
    #    GaussianNoise→64(L1)→Dropout(0.3)→32(L1)→Dropout(0.3)→8(bottleneck)→32→Dropout(0.3)→64
    input_dim   = X_train_benign.shape[1]
    inp         = Input(shape=(input_dim,))
    x           = GaussianNoise(0.05)(inp)   # denoising: forces learning of broad patterns
    x           = Dense(64, activation='relu',
                        activity_regularizer=regularizers.l1(1e-4))(x)
    x           = Dropout(0.3)(x)
    x           = Dense(32, activation='relu',
                        activity_regularizer=regularizers.l1(1e-4))(x)
    x           = Dropout(0.3)(x)
    x           = Dense(8, activation='relu')(x)    # bottleneck — 8 dims
    x           = Dense(32, activation='relu')(x)
    x           = Dropout(0.3)(x)
    x           = Dense(64, activation='relu')(x)
    out         = Dense(input_dim, activation='sigmoid')(x)   # MinMax-scaled → [0,1]
    autoencoder = Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # 4. Train with EarlyStopping (up to 50 epochs)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print(f"Training Autoencoder (input_dim={input_dim}, bottleneck=8, denoising+L1) — up to 50 epochs...")
    autoencoder.fit(
        X_train_benign, X_train_benign,
        epochs=50, batch_size=256,
        shuffle=True, validation_split=0.1,
        callbacks=[es], verbose=0
    )

    # 5. Adaptive threshold — sweep 85th–99th pct on benign val errors, pick best F1
    X_val        = val_df.drop(columns=['label']).values.astype('float32')
    y_val_binary = (val_df['label'] > 0).astype(int).values

    val_benign_df = val_df[val_df['label'] == 0]
    X_val_benign  = val_benign_df.drop(columns=['label']).values.astype('float32')

    benign_preds = autoencoder.predict(X_val_benign, verbose=0)
    mse_benign   = np.mean(np.power(X_val_benign - benign_preds, 2), axis=1)

    print("Searching for best anomaly threshold (85th–99th percentile)...")
    val_preds = autoencoder.predict(X_val, verbose=0)
    mse_val   = np.mean(np.power(X_val - val_preds, 2), axis=1)
    threshold = _find_best_threshold(mse_benign, mse_val, y_val_binary)
    print(f"Best Anomaly Threshold: {threshold:.6f}")

    # 6. Evaluate on full validation set (binary: 0 = benign, 1 = attack)
    print("Evaluating on full validation set...")
    y_pred = (mse_val > threshold).astype(int)

    precision = precision_score(y_val_binary, y_pred, zero_division=0)
    recall    = recall_score(y_val_binary, y_pred, zero_division=0)
    f1        = f1_score(y_val_binary, y_pred, zero_division=0)

    metrics = {
        'threshold': round(threshold, 6),
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1_score':  round(f1,        4),
    }
    print(f"-> Precision: {metrics['precision']:.4f}")
    print(f"-> Recall:    {metrics['recall']:.4f}")
    print(f"-> F1-Score:  {metrics['f1_score']:.4f}")

    # 6. Save model
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_SAVE_PATH / f'ae_node{node_id}.keras'
    autoencoder.save(model_path)
    print(f"Model saved to: {model_path}")

    return metrics


def run_local_ae_training():
    """Run standard AE training for all nodes."""
    print("=" * 60)
    print("Starting Local Autoencoder Training (Anomaly Detection)")
    print("=" * 60)

    all_metrics = {}
    for node_id in range(1, config.NUM_NODES + 1):
        metrics = train_autoencoder_node(node_id)
        if metrics:
            all_metrics[f'node_{node_id}'] = metrics

    if not all_metrics:
        print("\n[ERROR] No models were trained successfully.")
        return

    # Summary table
    print("\n" + "=" * 60)
    print("Local Autoencoder Models Comparison (Binary Detection)")
    print("=" * 60)
    print(f"{'Node':<10} | {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    for node, m in all_metrics.items():
        name = node.replace('_', ' ').title()
        print(f"{name:<10} | {m['threshold']:<10.6f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
    print("=" * 60)

    _save_ae_metrics(all_metrics)


# ==============================================================================
# FAST MODE  (original local_autoencoder_fast.py — now fixed with benign-only
#             training and per-node 95th-percentile threshold)
# ==============================================================================

def train_autoencoder_direct(node_id: int):
    """
    Fast AE training:
      - Loads raw device CSV directly (no pre-split required).
      - Drops identifier/leakage columns before scaling.
      - Trains on benign-only rows (label == 0).
      - Regularized AE: 64→Dropout(0.2)→32→Dropout(0.2)→16→32→Dropout(0.2)→64.
      - EarlyStopping(patience=5), up to 50 epochs.
      - Adaptive threshold: sweeps 85th–99th pct, picks best F1.
      - Saves threshold as a JSON sidecar file next to the model.
    """
    print(f"\n--- Training Local Autoencoder for Node {node_id} [fast mode] ---")

    nodes_dir  = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
    train_path = nodes_dir / f'node{node_id}_train.csv'
    val_path   = nodes_dir / f'node{node_id}_val.csv'

    # Use pre-split files if available; otherwise load device CSV and split on-the-fly
    if train_path.exists() and val_path.exists():
        print("Loading pre-split data...")
        train_df = pd.read_csv(train_path, low_memory=False)
        val_df   = pd.read_csv(val_path,   low_memory=False)
    else:
        processed_dir = config.BASE_DIR / 'datasets' / 'processed'
        device_file   = processed_dir / f'device{node_id}.csv'
        if not device_file.exists():
            print(f"[ERROR] No data found for Node {node_id}.")
            return None

        print(f"Pre-split files not found; loading {device_file.name} and splitting on-the-fly...")
        df = pd.read_csv(device_file, low_memory=False)

        # Encode labels (benign → 0)
        unique_labels = sorted(df['label'].unique().tolist())
        if 'benign' in unique_labels:
            unique_labels = ['benign'] + [l for l in unique_labels if l != 'benign']
        mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
        df['label_encoded'] = df['label'].map(mapping)

        _LEAKAGE_COLS = ['ip', 'mac', 'timestamp', 'src_ip', 'dst_ip', 'port', 'Unnamed: 0']
        df = df.drop(columns=[c for c in _LEAKAGE_COLS if c in df.columns])
        drop_cols = ['label', 'label_encoded', 'device_id']
        X_all = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include='number').fillna(0)
        df_final = X_all.copy()
        df_final['label'] = df['label_encoded']

        val_test_ratio    = config.VAL_SIZE + config.TEST_SIZE
        test_ratio_of_temp = config.TEST_SIZE / val_test_ratio
        train_df, temp_df = train_test_split(df_final, test_size=val_test_ratio, random_state=config.RANDOM_SEED)
        val_df, _         = train_test_split(temp_df,  test_size=test_ratio_of_temp, random_state=config.RANDOM_SEED)

    print(f"Total train rows: {len(train_df)} | Val rows: {len(val_df)}")

    # Train on BENIGN ONLY (label == 0)
    benign_train = (
        train_df[train_df['label'] == 0]
        .drop(columns=['label'])
        .select_dtypes(include='number')
        .fillna(0)
    )
    print(f"Benign train samples used for AE: {len(benign_train)}")

    if len(benign_train) == 0:
        print(f"[ERROR] No benign training samples for Node {node_id}.")
        return None

    # Inline MinMax normalisation (scoped to this node — not global)
    X_min = benign_train.min()
    X_max = benign_train.max()
    X_train_norm = ((benign_train - X_min) / (X_max - X_min + 1e-8)).values.astype('float32')

    input_dim = X_train_norm.shape[1]
    print(f"Building denoising autoencoder: {input_dim}→GaussianNoise→64(L1)→Dropout(0.3)→32(L1)→Dropout(0.3)→8→32→Dropout(0.3)→64→{input_dim}")

    inp  = Input(shape=(input_dim,))
    x    = GaussianNoise(0.05)(inp)   # denoising: forces learning of broad patterns
    x    = Dense(64, activation='relu',
                 activity_regularizer=regularizers.l1(1e-4))(x)
    x    = Dropout(0.3)(x)
    x    = Dense(32, activation='relu',
                 activity_regularizer=regularizers.l1(1e-4))(x)
    x    = Dropout(0.3)(x)
    x    = Dense(8,  activation='relu')(x)   # bottleneck — 8 dims
    x    = Dense(32, activation='relu')(x)
    x    = Dropout(0.3)(x)
    x    = Dense(64, activation='relu')(x)
    out  = Dense(input_dim, activation='sigmoid')(x)
    ae   = Model(inp, out)
    ae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print("Training denoising autoencoder on benign samples — up to 50 epochs...")
    ae.fit(X_train_norm, X_train_norm,
           epochs=50, batch_size=256,
           validation_split=0.1, callbacks=[es], verbose=0)

    # Adaptive threshold — sweep 85th–99th pct, pick the percentile that maximises F1
    val_benign = (
        val_df[val_df['label'] == 0]
        .drop(columns=['label'])
        .select_dtypes(include='number')
        .fillna(0)
    )
    if len(val_benign) == 0:
        print(f"[WARNING] No benign val samples for Node {node_id}. Using train errors for threshold.")
        val_benign = benign_train

    val_benign_norm = ((val_benign - X_min) / (X_max - X_min + 1e-8)).values.astype('float32')
    benign_preds_t  = ae.predict(val_benign_norm, verbose=0)
    mse_benign_t    = np.mean(np.power(val_benign_norm - benign_preds_t, 2), axis=1)

    # Full-val errors for threshold search
    val_all = (
        val_df.drop(columns=['label'])
        .select_dtypes(include='number')
        .fillna(0)
    )
    val_all_norm  = ((val_all - X_min) / (X_max - X_min + 1e-8)).values.astype('float32')
    val_all_preds = ae.predict(val_all_norm, verbose=0)
    mse_val_all   = np.mean(np.power(val_all_norm - val_all_preds, 2), axis=1)
    y_val_binary  = (val_df['label'] > 0).astype(int).values

    print(f"Searching for best anomaly threshold (85th–99th percentile)...")
    threshold = _find_best_threshold(mse_benign_t, mse_val_all, y_val_binary)
    print(f"Node {node_id} best anomaly threshold: {threshold:.6f}")

    # Save model + threshold sidecar
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path     = config.MODEL_SAVE_PATH / f'ae_node{node_id}.keras'
    threshold_path = config.MODEL_SAVE_PATH / f'ae_node{node_id}_threshold.json'

    ae.save(model_path)
    with open(threshold_path, 'w') as f:
        json.dump({'node_id': node_id, 'threshold': threshold}, f, indent=4)

    print(f"Model saved: {model_path}")
    print(f"Threshold saved: {threshold_path}")

    return {'node_id': node_id, 'threshold': round(threshold, 6)}


def run_fast_ae_training():
    """Run fast AE training for all nodes."""
    print("=" * 60)
    print("Starting Autoencoder Training (Fast Mode — Benign-Only)")
    print("=" * 60)

    all_results = {}
    for node_id in range(1, config.NUM_NODES + 1):
        result = train_autoencoder_direct(node_id)
        if result:
            all_results[f'node_{node_id}'] = result
        else:
            print(f"[WARNING] AE training skipped for Node {node_id}")

    _save_ae_metrics(all_results, filename='local_ae_thresholds.json')

    print("\n" + "=" * 60)
    print("Autoencoder Training Complete")
    print("=" * 60)


# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def _find_best_threshold(
    mse_benign: np.ndarray,
    mse_full_val: np.ndarray,
    y_val_binary: np.ndarray,
    percentile_range: range = range(85, 100),
) -> float:
    """
    Sweep candidate thresholds derived from the benign validation MSE distribution
    (85th through 99th percentile).  For each candidate, compute the binary F1
    on the full validation set and return the threshold that maximises it.

    Args:
        mse_benign:    Reconstruction errors for benign-only validation samples.
        mse_full_val:  Reconstruction errors for the complete validation set.
        y_val_binary:  Ground-truth binary labels (0 = benign, 1 = attack).
        percentile_range: Percentiles to sweep (default 85–99 inclusive).

    Returns:
        Optimal threshold (float).
    """
    best_threshold = float(np.percentile(mse_benign, 95))  # safe fallback
    best_f1        = -1.0

    for pct in percentile_range:
        candidate = float(np.percentile(mse_benign, pct))
        y_pred    = (mse_full_val > candidate).astype(int)
        score     = f1_score(y_val_binary, y_pred, zero_division=0)
        if score > best_f1:
            best_f1        = score
            best_threshold = candidate

    return best_threshold


def _save_ae_metrics(all_metrics: dict, filename: str = 'local_ae_metrics.json'):
    if not all_metrics:
        return
    results_dir = config.RESULTS_PATH
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / filename
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[OK] Aggregated metrics saved to: {metrics_path}")


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local autoencoder training (anomaly detection)")
    parser.add_argument(
        '--mode',
        choices=['standard', 'fast'],
        default='standard',
        help="standard = pre-split CSVs, deeper AE, full eval (default) | fast = raw CSV, shallow AE"
    )
    args = parser.parse_args()

    if args.mode == 'fast':
        run_fast_ae_training()
    else:
        run_local_ae_training()
