"""
local_autoencoder.py
====================
Per-node autoencoder training for IoT botnet detection (PQC-IoT Sentinel).

Rules
-----
1. Each node's autoencoder is trained ONLY on its own device benign rows
   (label == 0 or label == 'benign') from nodeX_train.csv — never on
   combined data.
2. Reconstruction-error threshold is computed per device as:
       threshold_nodeX = np.percentile(benign_val_errors, 95)
3. Each threshold is saved to results/thresholds.json as:
       {"node1": float, "node2": float, "node3": float}
4. Evaluation runs on the full val set (benign + attack) using the
   per-device threshold.
5. Metrics (precision, recall, F1, AUC) are printed and saved to
       results/local_ae_metrics.json  per node.

Architecture
------------
  Encoder : 115 → Dense(64, relu) → Dense(32, relu)   [bottleneck]
  Decoder : Dense(64, relu) → Dense(115, sigmoid)
  Epochs  : 20
  Batch   : 256
  Loss    : MSE
  Optimizer: Adam(lr=0.001)

Usage
-----
  python local_autoencoder.py                  # train all nodes
  python local_autoencoder.py --node_id 1      # train a single node
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Environment — force CPU, silence TF logs
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
except ImportError as exc:
    print(f"[ERROR] TensorFlow import failed: {exc}")
    print("Run: pip install --upgrade tensorflow")
    raise

# Suppress noisy GPU/oneDNN messages that slip past TF_CPP_MIN_LOG_LEVEL
tf.get_logger().setLevel("ERROR")

import config  # project-level paths

# ---------------------------------------------------------------------------
# Constants matching the required spec
# ---------------------------------------------------------------------------
ENCODER_DIMS = [64, 32]          # encoder hidden layer widths
DECODER_DIMS = [64]              # decoder hidden layer widths (before output)
INPUT_DIM    = 115               # expected feature dimensionality
EPOCHS       = 20
BATCH_SIZE   = 256
THRESHOLD_PERCENTILE = 95

LEAKAGE_COLS = [
    "ip", "mac", "timestamp", "src_ip", "dst_ip", "port", "Unnamed: 0"
]


# ===========================================================================
# Helper: build autoencoder
# ===========================================================================

def _build_autoencoder(input_dim: int) -> Model:
    """
    Encoder : input_dim → 64(relu) → 32(relu)
    Decoder : 32        → 64(relu) → input_dim(sigmoid)
    """
    inp = Input(shape=(input_dim,), name="ae_input")

    # Encoder
    x = Dense(64, activation="relu", name="enc_1")(inp)
    x = Dense(32, activation="relu", name="enc_2")(x)

    # Decoder
    x = Dense(64, activation="relu", name="dec_1")(x)
    out = Dense(input_dim, activation="sigmoid", name="ae_output")(x)

    model = Model(inputs=inp, outputs=out, name=f"autoencoder_{input_dim}")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# ===========================================================================
# Core training function — callable per node or in batch
# ===========================================================================

def train_autoencoder_node(node_id: int) -> dict | None:
    """
    Train a per-device autoencoder for anomaly detection.

    Parameters
    ----------
    node_id : int
        Node identifier (1, 2, or 3).

    Returns
    -------
    dict with keys {threshold, precision, recall, f1_score, auc} or None on
    error.
    """
    print(f"\n{'='*60}")
    print(f"  Training Autoencoder — Node {node_id}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load pre-split CSVs
    # ------------------------------------------------------------------
    nodes_dir  = config.BASE_DIR / "datasets" / "processed" / "nodes"
    train_path = nodes_dir / f"node{node_id}_train.csv"
    val_path   = nodes_dir / f"node{node_id}_val.csv"

    if not train_path.exists() or not val_path.exists():
        print(f"[ERROR] Missing CSVs for Node {node_id}:")
        print(f"        train: {train_path}")
        print(f"        val  : {val_path}")
        return None

    print("  Loading data …")
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df   = pd.read_csv(val_path,   low_memory=False)

    # Drop leakage / identifier columns
    train_df = train_df.drop(columns=[c for c in LEAKAGE_COLS if c in train_df.columns])
    val_df   = val_df.drop(columns=[c for c in LEAKAGE_COLS if c in val_df.columns])

    # ------------------------------------------------------------------
    # 2. Normalise string labels → 0 (benign) / 1 (attack)
    # ------------------------------------------------------------------
    for df in (train_df, val_df):
        if df["label"].dtype == object:
            df["label"] = df["label"].str.strip().str.lower()
            df["label"] = df["label"].apply(lambda v: 0 if v == "benign" else 1)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)

    # ------------------------------------------------------------------
    # 3. Isolate benign-only rows for TRAINING (rule 1)
    # ------------------------------------------------------------------
    benign_train_df = train_df[train_df["label"] == 0].copy()
    n_total  = len(train_df)
    n_benign = len(benign_train_df)
    print(f"  Train rows total: {n_total:,}  |  Benign (used for AE): {n_benign:,}")

    if n_benign == 0:
        print(f"[ERROR] No benign training samples for Node {node_id}. Skipping.")
        return None

    # Feature matrix — numeric only, drop label
    X_train_benign = (
        benign_train_df
        .drop(columns=["label"])
        .select_dtypes(include="number")
        .fillna(0)
        .values.astype("float32")
    )

    input_dim = X_train_benign.shape[1]
    print(f"  Feature dimensionality: {input_dim}")

    # ------------------------------------------------------------------
    # 4. Build & train autoencoder
    # ------------------------------------------------------------------
    print(f"  Architecture: {input_dim}→64→32 (encoder) | 32→64→{input_dim} (decoder)")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")

    autoencoder = _build_autoencoder(input_dim)

    autoencoder.fit(
        X_train_benign,
        X_train_benign,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.1,
        verbose=0,
    )
    print("  Training complete.")

    # ------------------------------------------------------------------
    # 5. Compute threshold = 95th percentile of benign-val errors (rule 2)
    # ------------------------------------------------------------------
    val_benign_df = val_df[val_df["label"] == 0].copy()
    if len(val_benign_df) == 0:
        print(f"[WARNING] No benign val rows for Node {node_id}; "
              "using train benign errors for threshold.")
        X_val_benign = X_train_benign
    else:
        X_val_benign = (
            val_benign_df
            .drop(columns=["label"])
            .select_dtypes(include="number")
            .fillna(0)
            .values.astype("float32")
        )

    benign_val_preds  = autoencoder.predict(X_val_benign, verbose=0)
    benign_val_errors = np.mean(np.square(X_val_benign - benign_val_preds), axis=1)
    threshold         = float(np.percentile(benign_val_errors, THRESHOLD_PERCENTILE))
    print(f"  Threshold (95th pct of benign val MSE): {threshold:.8f}")

    # ------------------------------------------------------------------
    # 6. Evaluate on the full val set — benign + attack (rule 4)
    # ------------------------------------------------------------------
    X_val_all = (
        val_df
        .drop(columns=["label"])
        .select_dtypes(include="number")
        .fillna(0)
        .values.astype("float32")
    )
    y_val_true = val_df["label"].values.astype(int)

    val_preds  = autoencoder.predict(X_val_all, verbose=0)
    mse_val    = np.mean(np.square(X_val_all - val_preds), axis=1)
    y_pred     = (mse_val > threshold).astype(int)

    precision = float(precision_score(y_val_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_val_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_val_true, y_pred, zero_division=0))

    # AUC — use raw MSE scores as the anomaly probability
    unique_classes = np.unique(y_val_true)
    if len(unique_classes) == 2:
        auc = float(roc_auc_score(y_val_true, mse_val))
    else:
        auc = float("nan")
        print("  [WARNING] Only one class in val labels; AUC set to NaN.")

    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_SAVE_PATH / f"ae_node{node_id}.keras"
    autoencoder.save(model_path)
    print(f"  Model saved → {model_path}")

    return {
        "threshold": round(threshold, 8),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "auc":       round(auc, 4) if not np.isnan(auc) else None,
    }


# ===========================================================================
# Orchestration helpers
# ===========================================================================

def _save_thresholds(all_metrics: dict[str, dict]) -> None:
    """Save {node1: threshold, node2: threshold, node3: threshold} to
    results/thresholds.json (rule 3)."""
    thresholds = {
        key: val["threshold"]
        for key, val in all_metrics.items()
        if val and "threshold" in val
    }
    results_dir = config.RESULTS_PATH
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "thresholds.json"
    with open(out_path, "w") as fh:
        json.dump(thresholds, fh, indent=4)
    print(f"\n[OK] Thresholds saved → {out_path}")


def _save_metrics(all_metrics: dict[str, dict]) -> None:
    """Save full per-node metrics to results/local_ae_metrics.json (rule 5)."""
    results_dir = config.RESULTS_PATH
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "local_ae_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(all_metrics, fh, indent=4)
    print(f"[OK] Full metrics saved → {out_path}")


def run_local_ae_training(node_ids: list[int] | None = None) -> None:
    """
    Train autoencoders for the specified nodes (default: all 3).

    Parameters
    ----------
    node_ids : list[int] | None
        Which nodes to train.  ``None`` → [1, 2, 3].
    """
    if node_ids is None:
        node_ids = list(range(1, config.NUM_NODES + 1))

    print("\n" + "=" * 60)
    print("  PQC-IoT Sentinel — Local Autoencoder Training")
    print("=" * 60)

    all_metrics: dict[str, dict] = {}

    for nid in node_ids:
        result = train_autoencoder_node(nid)
        if result is not None:
            all_metrics[f"node{nid}"] = result
        else:
            print(f"[WARNING] Node {nid} training failed — excluded from results.")

    if not all_metrics:
        print("\n[ERROR] No models trained successfully.")
        return

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  {'Node':<8} {'Threshold':<14} {'Precision':<11} "
          f"{'Recall':<9} {'F1':<9} {'AUC':<9}")
    print("  " + "-" * 68)
    for node_key, m in all_metrics.items():
        auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A  "
        print(f"  {node_key:<8} {m['threshold']:<14.8f} {m['precision']:<11.4f} "
              f"{m['recall']:<9.4f} {m['f1_score']:<9.4f} {auc_str:<9}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Persist results (rules 3 & 5)
    # ------------------------------------------------------------------
    _save_thresholds(all_metrics)
    _save_metrics(all_metrics)

    print("\n  All done.\n")


# ===========================================================================
# CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local autoencoder training for PQC-IoT Sentinel"
    )
    parser.add_argument(
        "--node_id",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Train a single node (1, 2, or 3). Omit to train all nodes.",
    )
    args = parser.parse_args()

    if args.node_id is not None:
        node_ids = [args.node_id]
    else:
        node_ids = None   # run_local_ae_training defaults to all

    run_local_ae_training(node_ids)
