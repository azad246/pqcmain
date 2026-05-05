"""
federated/fl_client.py
======================
PQC-secured Flower FL Client with Differential Privacy via Opacus.

Key changes vs previous version:
  - MLPClassifier (sklearn) replaced by a PyTorch nn.Module MLP so that
    Opacus PrivacyEngine can hook into the autograd graph for per-sample
    gradient clipping + Gaussian noise addition.
  - PrivacyEngine is initialised once per client in __init__ and attached
    to the model, optimizer, and DataLoader each round inside fit().
  - After training, the spent privacy budget (epsilon) is extracted from
    the accountant and returned to the server inside the fit() metrics dict.
  - Weighted FedAvg metrics (local_f1, local_loss) are still reported.
  - PQC encryption/decryption unchanged.
"""

import argparse
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import flwr as fl
from sklearn.metrics import log_loss, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Opacus import — graceful fallback to manual DP if not installed
# ---------------------------------------------------------------------------
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("[WARNING] Opacus not found. DP will use manual gradient clipping fallback.")

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from crypto import pqc_crypto
from crypto.encrypt_weights import encrypt_weights, decrypt_weights


# ---------------------------------------------------------------------------
# DP hyper-parameters
# ---------------------------------------------------------------------------
DP_MAX_GRAD_NORM    = 1.0    # per-sample gradient clipping bound (C)
DP_NOISE_MULTIPLIER = 1.1    # Gaussian noise σ/C — updated to 1.1 per spec
DP_DELTA            = 1e-5   # δ for (ε, δ)-DP guarantee
DP_BATCH_SIZE       = 256    # must match DataLoader batch size for accountant
DP_LOCAL_EPOCHS     = 1      # local epochs per FL round

# Manual fallback DP constants (used when Opacus unavailable)
FALLBACK_CLIP_VAL   = 1.0    # clip weight arrays to [-1.0, 1.0]
FALLBACK_NOISE_STD  = 0.01   # std of Gaussian noise added to weights


# ==============================================================================
# PyTorch MLP — equivalent to sklearn MLPClassifier(128, 64)
# ==============================================================================

class MLP(nn.Module):
    """
    Simple feed-forward MLP compatible with Opacus.
    Opacus requires:
      - No in-place operations
      - No BatchNorm (use GroupNorm or no norm)
      - Single input tensor per forward pass
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# Weight extraction / injection helpers
# ==============================================================================

def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Return all trainable parameters as a flat list of numpy arrays."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]):
    """Load a flat list of numpy arrays back into the model in-place."""
    with torch.no_grad():
        for param, new_val in zip(model.parameters(), parameters):
            param.copy_(torch.tensor(new_val, dtype=param.dtype))


# ==============================================================================
# FL Client
# ==============================================================================

class PQC_FLClient(fl.client.NumPyClient):

    def __init__(self, node_id: int, use_differential_privacy: bool = True):
        self.node_id  = node_id
        self.use_dp   = use_differential_privacy   # --dp flag (spec §6)
        self.keys_dir = config.BASE_DIR / 'keys'
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        # Cumulative privacy budget across all rounds
        self.total_epsilon: float = 0.0

        # --- 1. PQC KEY GENERATION (KEM) ---
        print(f">>> [Node {self.node_id}] Generating Client KEM Keypair (Kyber512)...")
        self.client_pub, self.client_sec = pqc_crypto.generate_keypair()
        with open(self.keys_dir / f'client_{self.node_id}_pub.pem', 'wb') as f:
            f.write(self.client_pub)

        # --- 2. DILITHIUM2 SIGNING KEYPAIR ---
        print(f">>> [Node {self.node_id}] Generating Client Signing Keypair (Dilithium2)...")
        self.signing_pub, self.signing_sec = pqc_crypto.generate_signing_keypair()
        with open(self.keys_dir / f'client_{self.node_id}_signing_pub.bin', 'wb') as f:
            f.write(self.signing_pub)

        self.load_data()
        self.init_model()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        print(f"[Node {self.node_id}] Loading local training and validation data...")
        nodes_dir  = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
        train_path = nodes_dir / f'node{self.node_id}_train.csv'
        val_path   = nodes_dir / f'node{self.node_id}_val.csv'

        train_df = pd.read_csv(train_path, low_memory=False)
        val_df   = pd.read_csv(val_path,   low_memory=False)

        self.X_train = train_df.drop(columns=['label']).values.astype(np.float32)
        self.y_train = train_df['label'].values.astype(np.int64)
        self.X_val   = val_df.drop(columns=['label']).values.astype(np.float32)
        self.y_val   = val_df['label'].values.astype(np.int64)

        mapping_path = nodes_dir / f'node{self.node_id}_label_mapping.json'
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)

        self.num_classes   = len(mapping)
        self.total_classes = np.arange(self.num_classes)
        self.input_dim     = self.X_train.shape[1]

        val_ds = TensorDataset(torch.tensor(self.X_val), torch.tensor(self.y_val))
        self.val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def init_model(self):
        self.device = torch.device('cpu')
        self.model  = MLP(self.input_dim, self.num_classes).to(self.device)

        # PrivacyEngine only created when Opacus is available AND DP is enabled
        if self.use_dp and OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine()
            dp_mode = "Opacus (noise_mult={}, max_grad_norm={})".format(
                DP_NOISE_MULTIPLIER, DP_MAX_GRAD_NORM)
        elif self.use_dp:
            self.privacy_engine = None
            dp_mode = "Manual fallback (clip=±{}, noise_std={})".format(
                FALLBACK_CLIP_VAL, FALLBACK_NOISE_STD)
        else:
            self.privacy_engine = None
            dp_mode = "DISABLED"

        print(f"[Node {self.node_id}] MLP initialised  "
              f"(input={self.input_dim}  classes={self.num_classes}  DP={dp_mode})")

    # ------------------------------------------------------------------
    # Flower: get_properties
    # ------------------------------------------------------------------

    def get_properties(self, ins: fl.common.GetPropertiesIns) -> fl.common.GetPropertiesRes:
        return fl.common.GetPropertiesRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            properties={"signing_pub_key": self.signing_pub.hex()}
        )

    # ------------------------------------------------------------------
    # Flower: get_parameters
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    # ------------------------------------------------------------------
    # Flower: set_parameters  (PQC decryption intercept)
    # ------------------------------------------------------------------

    def set_parameters(self, parameters):
        if not parameters:
            return

        # Detect a PQC-encrypted bundle from the server
        is_encrypted_bundle = (
            len(parameters) == config.NUM_NODES
            and parameters[0].dtype == np.uint8
            and parameters[0].ndim == 1
        )

        if is_encrypted_bundle:
            print(f"[Node {self.node_id}] Decrypting PQC global weights from Server...")
            my_encrypted_bytes = parameters[self.node_id - 1].tobytes()
            decrypted_weights  = decrypt_weights(my_encrypted_bytes, self.client_sec)
            set_model_parameters(self.model, decrypted_weights)
        else:
            set_model_parameters(self.model, parameters)

    # ------------------------------------------------------------------
    # Flower: fit  (DP training + PQC encryption)
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        epsilon   = 0.0   # default when DP is off

        # ==============================================================
        # PATH A: Opacus DP (use_dp=True and Opacus installed)
        # ==============================================================
        if self.use_dp and OPACUS_AVAILABLE:

            # ── STEP 1: Fresh DataLoader (Opacus needs new one each round)
            train_ds     = TensorDataset(
                torch.tensor(self.X_train), torch.tensor(self.y_train)
            )
            train_loader = DataLoader(
                train_ds, batch_size=DP_BATCH_SIZE,
                shuffle=True, drop_last=True,
            )

            # ── STEP 2: Attach PrivacyEngine
            # make_private() wraps model, optimizer, and DataLoader.
            # It adds per-sample gradient clipping (max_grad_norm=1.0)
            # and Gaussian noise (noise_multiplier=1.1) automatically.
            dp_model, dp_optimizer, dp_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=DP_NOISE_MULTIPLIER,   # 1.1
                max_grad_norm=DP_MAX_GRAD_NORM,          # 1.0
            )

            # ── STEP 3: Local DP training
            dp_model.train()
            for epoch in range(DP_LOCAL_EPOCHS):
                epoch_loss, batches = 0.0, 0
                for X_b, y_b in dp_loader:
                    dp_optimizer.zero_grad()
                    loss = criterion(dp_model(X_b.to(self.device)), y_b.to(self.device))
                    loss.backward()
                    dp_optimizer.step()
                    epoch_loss += loss.item(); batches += 1
                if batches:
                    print(f"[Node {self.node_id}] DP Epoch {epoch+1}  "
                          f"avg_loss={epoch_loss/batches:.4f}")

            # ── STEP 4: Compute and print privacy budget (spec §4)
            epsilon = self.privacy_engine.get_epsilon(delta=DP_DELTA)
            self.total_epsilon = epsilon
            print(f"[Node {self.node_id}] Privacy budget spent: "
                  f"ε={epsilon:.4f}  δ={DP_DELTA}  (noise_mult={DP_NOISE_MULTIPLIER})")

            # ── STEP 5: Extract weights from wrapped model
            raw_weights = get_model_parameters(dp_model._module)

            # ── STEP 6: Val eval via unwrapped module
            eval_model = dp_model._module

        # ==============================================================
        # PATH B: Manual DP fallback (use_dp=True but Opacus missing)
        # ==============================================================
        elif self.use_dp:

            train_ds     = TensorDataset(
                torch.tensor(self.X_train), torch.tensor(self.y_train)
            )
            train_loader = DataLoader(
                train_ds, batch_size=DP_BATCH_SIZE, shuffle=True
            )

            self.model.train()
            for epoch in range(DP_LOCAL_EPOCHS):
                epoch_loss, batches = 0.0, 0
                for X_b, y_b in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(
                        self.model(X_b.to(self.device)), y_b.to(self.device)
                    )
                    loss.backward()

                    # ── Manual gradient clipping at max_grad_norm=1.0 (spec §2)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), DP_MAX_GRAD_NORM
                    )
                    optimizer.step()
                    epoch_loss += loss.item(); batches += 1
                if batches:
                    print(f"[Node {self.node_id}] Manual-DP Epoch {epoch+1}  "
                          f"avg_loss={epoch_loss/batches:.4f}")

            # ── Manual noise injection on weights (spec §7)
            # Clip all weight arrays to [-1.0, 1.0], then add Gaussian noise
            with torch.no_grad():
                for param in self.model.parameters():
                    param.clamp_(-FALLBACK_CLIP_VAL, FALLBACK_CLIP_VAL)
                    param.add_(
                        torch.tensor(
                            np.random.normal(0, FALLBACK_NOISE_STD, param.shape),
                            dtype=param.dtype
                        )
                    )

            epsilon = float("nan")   # no accountant in fallback mode
            self.total_epsilon = 0.0
            print(f"[Node {self.node_id}] Manual DP applied  "
                  f"(clip=±{FALLBACK_CLIP_VAL}, noise_std={FALLBACK_NOISE_STD})  "
                  f"ε=N/A (no Opacus accountant)")

            raw_weights = get_model_parameters(self.model)
            eval_model  = self.model

        # ==============================================================
        # PATH C: No DP (use_dp=False)
        # ==============================================================
        else:
            train_ds     = TensorDataset(
                torch.tensor(self.X_train), torch.tensor(self.y_train)
            )
            train_loader = DataLoader(
                train_ds, batch_size=DP_BATCH_SIZE, shuffle=True
            )
            self.model.train()
            for epoch in range(DP_LOCAL_EPOCHS):
                epoch_loss, batches = 0.0, 0
                for X_b, y_b in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(
                        self.model(X_b.to(self.device)), y_b.to(self.device)
                    )
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item(); batches += 1
                if batches:
                    print(f"[Node {self.node_id}] Epoch {epoch+1}  "
                          f"avg_loss={epoch_loss/batches:.4f}")

            print(f"[Node {self.node_id}] DP DISABLED — no privacy noise applied.")
            raw_weights = get_model_parameters(self.model)
            eval_model  = self.model

        # ── Shared: local validation eval ─────────────────────────────
        eval_model.eval()
        all_preds, all_probs, all_true = [], [], []
        with torch.no_grad():
            for X_b, y_b in self.val_loader:
                logits = eval_model(X_b.to(self.device))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_preds.extend(np.argmax(probs, axis=1).tolist())
                all_true.extend(y_b.numpy().tolist())

        local_f1 = float(f1_score(all_true, all_preds, average='weighted', zero_division=0))
        try:
            local_loss = float(log_loss(all_true, all_probs, labels=list(self.total_classes)))
        except Exception:
            local_loss = 1.0
        print(f"[Node {self.node_id}] Val F1={local_f1:.4f}  loss={local_loss:.4f}")

        # ── PQC encryption ────────────────────────────────────────────
        server_pub_path = self.keys_dir / 'server_pub.pem'
        while not server_pub_path.exists():
            time.sleep(1)
        with open(server_pub_path, 'rb') as f:
            server_pub = f.read()

        print(f"[Node {self.node_id}] Encrypting local updates (PQC)...")
        encrypted_bytes      = encrypt_weights(raw_weights, server_pub)

        # ── Dilithium2 signature ──────────────────────────────────────
        signature = pqc_crypto.sign_weights(encrypted_bytes, self.signing_sec)
        print(f"[Node {self.node_id}] Signed payload ({len(signature)}B)")

        # Return both encrypted weights and signature together in parameters
        encrypted_parameters = [
            np.frombuffer(encrypted_bytes, dtype=np.uint8),
            np.frombuffer(signature, dtype=np.uint8)
        ]

        # ── Metrics dict (spec §5: epsilon included) ──────────────────
        fit_metrics = {
            "local_f1":         local_f1,
            "local_loss":       local_loss,
            "dp_epsilon":       round(epsilon, 6) if not (isinstance(epsilon, float) and np.isnan(epsilon)) else -1.0,
            "dp_delta":         DP_DELTA,
            "dp_noise_mult":    DP_NOISE_MULTIPLIER if self.use_dp else 0.0,
            "dp_max_grad_norm": DP_MAX_GRAD_NORM,
            "dp_mode":          ("opacus" if OPACUS_AVAILABLE else "manual") if self.use_dp else "disabled",
        }
        return encrypted_parameters, len(self.X_train), fit_metrics

    # ------------------------------------------------------------------
    # Flower: evaluate
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        all_preds, all_probs, all_true = [], [], []
        with torch.no_grad():
            for X_b, y_b in self.val_loader:
                logits = self.model(X_b.to(self.device))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = np.argmax(probs, axis=1)
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_true.extend(y_b.numpy().tolist())

        try:
            loss = float(log_loss(all_true, all_probs, labels=list(self.total_classes)))
        except Exception:
            loss = 1.0

        accuracy = float(accuracy_score(all_true, all_preds))
        f1       = float(f1_score(all_true, all_preds, average='weighted', zero_division=0))

        metrics = {
            "accuracy":   accuracy,
            "f1":         f1,
            "dp_epsilon": round(self.total_epsilon, 6),
        }
        return loss, len(self.X_val), metrics


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PQC + DP Flower FL Client")
    parser.add_argument("--node_id",        type=int, required=True,
                        help="Node ID (e.g., 1, 2, 3)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080",
                        help="FL Server address")
    # ── DP toggle (spec §6) ─────────────────────────────────────────────────
    parser.add_argument("--dp", type=str, default="true",
                        choices=["true", "false"],
                        help="Enable differential privacy: true (default) | false")
    args = parser.parse_args()

    use_dp = args.dp.lower() == "true"

    print("=" * 65)
    print(f"  PQC-IoT Sentinel — FL Client  Node {args.node_id}")
    print(f"  DP enabled       : {use_dp}")
    if use_dp:
        print(f"  Opacus available : {OPACUS_AVAILABLE}")
        if OPACUS_AVAILABLE:
            print(f"  noise_multiplier : {DP_NOISE_MULTIPLIER}  "
                  f"max_grad_norm: {DP_MAX_GRAD_NORM}  delta: {DP_DELTA}")
        else:
            print(f"  Fallback mode    : clip=±{FALLBACK_CLIP_VAL}  "
                  f"noise_std={FALLBACK_NOISE_STD}")
    print("=" * 65)

    client = PQC_FLClient(
        node_id=args.node_id,
        use_differential_privacy=use_dp,
    )
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )
