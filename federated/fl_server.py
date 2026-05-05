"""
federated/fl_server.py
======================
PQC-IoT Sentinel — Secure Federated Learning Server

Strategy: F1×Reputation Weighted Aggregation
---------------------------------------------
Each node's aggregation weight is computed as:

    w_i = (local_f1_i × reputation_i) / Σ_j (local_f1_j × reputation_j)

Reputation System (cosine-similarity based)
-------------------------------------------
  - Each node starts with reputation_score = 1.0.
  - After every round the server computes the element-wise median update
    vector across all verified clients.
  - Any client whose weight-update cosine similarity to that median falls
    below COSINE_SIM_THRESHOLD (0.7) receives a penalty of −0.15.
  - Clients that pass the similarity check are NOT penalised (reputation
    is held constant; rewards were removed to keep the decay predictable).
  - Reputation is clamped to [0, 1.0].
  - Clients whose reputation falls below REPUTATION_FLOOR are EXCLUDED
    from aggregation that round (but still participate in future rounds).

Logging
-------
  results/fl_round_metrics.csv — one row per round:
    Round | Global_Accuracy | Global_F1 |
    Node1_Rep | Node1_LocalF1 |
    Node2_Rep | Node2_LocalF1 |
    Node3_Rep | Node3_LocalF1 |
    Clients_Aggregated | Clients_Penalised | Encryption_Overhead_ms

Compatibility
-------------
  - Server address unchanged: 0.0.0.0:8080
  - Reads the same PQC encrypted payloads and Dilithium2 signatures that
    fl_client.py sends.
  - Min clients before starting: 3
  - Total rounds: 10  (config.FL_ROUNDS)
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from crypto import pqc_crypto
from crypto.pqc_crypto import verify_weights
from crypto.encrypt_weights import encrypt_weights, decrypt_weights


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_FIT_CLIENTS        = 3       # must have ≥ 3 clients before any round starts
NUM_ROUNDS             = config.FL_ROUNDS   # 10

REPUTATION_INIT        = 1.0    # every node starts here
REPUTATION_PENALTY     = 0.15   # subtracted when cosine sim < threshold
REPUTATION_FLOOR       = 0.30   # below this → excluded from aggregation this round
COSINE_SIM_THRESHOLD   = 0.70   # minimum cosine similarity to median update


# ===========================================================================
# Cosine-similarity utility
# ===========================================================================

def _flatten(weight_list: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of weight arrays into a single 1-D float64 vector."""
    return np.concatenate([w.ravel().astype(np.float64) for w in weight_list])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity ∈ [−1, 1] between two flat vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ===========================================================================
# Reputation tracker  (cosine-similarity based)
# ===========================================================================

class ReputationTracker:
    """
    Tracks a reputation score in [0, 1] for every client node.

    Each round:
      1. Flatten each client's weight-update into a 1-D vector.
      2. Compute the element-wise median update vector.
      3. Flag any client whose cosine similarity to the median < 0.70.
      4. Penalise flagged clients by REPUTATION_PENALTY.
    """

    def __init__(self, node_ids: list[int]):
        self.scores: dict[int, float] = {nid: REPUTATION_INIT for nid in node_ids}

    def update(
        self,
        server_round: int,
        client_ids: list[int],
        weight_deltas: list[list[np.ndarray]],
    ) -> dict[int, bool]:
        """
        Compute cosine similarities, update scores, and return a
        {client_id → penalised?} dict.
        """
        # ── STEP A: Flatten each client's update ─────────────────────────
        flat_updates = [_flatten(delta) for delta in weight_deltas]

        # ── STEP B: Element-wise median update vector ────────────────────
        #   Stack all client vectors into a matrix, take the median across
        #   clients (axis=0) to get a robust reference direction.
        stacked = np.stack(flat_updates, axis=0)          # shape: (N_clients, D)
        median_vec = np.median(stacked, axis=0)           # shape: (D,)

        # ── STEP C: Cosine similarity of each client to the median ───────
        penalised: dict[int, bool] = {}

        print(f"\n  [Reputation] Round {server_round} — "
              f"cosine-similarity check ({len(client_ids)} clients):")
        print(f"  {'Node':<8} {'CosSim':<10} {'Threshold':<12} {'Penalised':<10} {'Score'}")
        print("  " + "-" * 55)

        for cid, flat_upd in zip(client_ids, flat_updates):
            cos_sim = cosine_similarity(flat_upd, median_vec)
            is_penalised = cos_sim < COSINE_SIM_THRESHOLD

            if is_penalised:
                # ── RULE: decrease reputation by 0.15 ────────────────────
                self.scores[cid] = max(0.0, self.scores[cid] - REPUTATION_PENALTY)

            penalised[cid] = is_penalised

            flag_str = "⚠ YES" if is_penalised else "  no"
            print(
                f"  Node {cid:<4} "
                f"{cos_sim:<10.4f} "
                f"{COSINE_SIM_THRESHOLD:<12.2f} "
                f"{flag_str:<10} "
                f"{self.scores[cid]:.4f}"
            )

        print()
        return penalised

    def is_trusted(self, client_id: int) -> bool:
        return self.scores.get(client_id, REPUTATION_INIT) >= REPUTATION_FLOOR


# ===========================================================================
# Weighted aggregation
# ===========================================================================

def f1_reputation_weighted_fedavg(
    weight_list: list[list[np.ndarray]],
    client_ids:  list[int],
    client_f1s:  list[float],
    reputation:  ReputationTracker,
) -> list[np.ndarray]:
    """
    Aggregate model weights using the formula:

        w_i = (local_f1_i × reputation_i) / Σ_j (local_f1_j × reputation_j)

    Parameters
    ----------
    weight_list  : list of per-client weight arrays
    client_ids   : node IDs matching weight_list order
    client_f1s   : local F1 scores from fl_client.py fit() metrics
    reputation   : ReputationTracker holding current scores

    Returns
    -------
    Aggregated model weights as a list of ndarrays.
    """
    # ── Compute raw weights ─────────────────────────────────────────────
    raw_weights = [
        f1 * reputation.scores.get(cid, REPUTATION_INIT)
        for cid, f1 in zip(client_ids, client_f1s)
    ]
    total = sum(raw_weights)

    if total < 1e-12:
        # Edge-case: all weights zero → fall back to equal weighting
        print("  [WeightedFedAvg] ⚠ All weights zero — using equal weighting.")
        raw_weights = [1.0] * len(weight_list)
        total = float(len(weight_list))

    # ── Normalise ───────────────────────────────────────────────────────
    normalised = [w / total for w in raw_weights]

    print("  [WeightedFedAvg] Aggregation weights (f1 × rep / Σ):")
    for cid, f1, rep, nw in zip(
        client_ids, client_f1s,
        [reputation.scores.get(c, REPUTATION_INIT) for c in client_ids],
        normalised,
    ):
        print(f"    Node {cid}:  local_f1={f1:.4f}  rep={rep:.4f}  → weight={nw:.4f}")

    # ── Weighted average layer by layer ─────────────────────────────────
    aggregated: list[np.ndarray] = []
    n_layers = len(weight_list[0])
    for layer_idx in range(n_layers):
        layer_acc = np.zeros_like(weight_list[0][layer_idx], dtype=np.float64)
        for client_weights, norm_w in zip(weight_list, normalised):
            layer_acc += norm_w * client_weights[layer_idx].astype(np.float64)
        aggregated.append(layer_acc)

    return aggregated


# ===========================================================================
# Strategy: F1×Reputation Weighted FedAvg  +  PQC encryption
# ===========================================================================

class SecureWeightedFedAvg(fl.server.strategy.Strategy):
    """
    Custom Flower strategy implementing:
      1. Dilithium2 signature verification on incoming updates.
      2. PQC (Kyber512) decryption of client weight payloads.
      3. Cosine-similarity reputation system (penalty = −0.15 if sim < 0.70).
      4. Weighted aggregation: w_i = (f1_i × rep_i) / Σ(f1_j × rep_j).
      5. PQC re-encryption of the global model before broadcasting.
      6. Per-round CSV logging to results/fl_round_metrics.csv.
    """

    def __init__(
        self,
        min_fit_clients: int       = MIN_FIT_CLIENTS,
        min_evaluate_clients: int  = MIN_FIT_CLIENTS,
        min_available_clients: int = MIN_FIT_CLIENTS,
        evaluate_metrics_aggregation_fn=None,
    ):
        self.min_fit_clients           = min_fit_clients
        self.min_evaluate_clients      = min_evaluate_clients
        self.min_available_clients     = min_available_clients
        self.eval_metrics_agg_fn       = evaluate_metrics_aggregation_fn

        # Paths
        config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        self.metrics_file = config.RESULTS_PATH / "fl_round_metrics.csv"

        # PQC server keypair
        self.keys_dir = config.BASE_DIR / "keys"
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        print(">>> [Server] Generating PQC Keypair (Kyber512)...")
        self.server_pub, self.server_sec = pqc_crypto.generate_keypair()
        with open(self.keys_dir / "server_pub.pem", "wb") as fh:
            fh.write(self.server_pub)

        # Reputation tracker — one entry per expected node (1, 2, 3)
        self.reputation = ReputationTracker(list(range(1, config.NUM_NODES + 1)))

        # Store last global weights for computing deltas
        self._last_global_weights: list[np.ndarray] | None = None

        # Cache per-round fit metrics for use in aggregate_evaluate
        self._round_fit_cache: dict = {}

        # Encryption overhead (set in aggregate_fit, read in aggregate_evaluate)
        self._last_overhead_ms: float = 0.0

        # ── Write CSV header ─────────────────────────────────────────────
        #   Columns are fixed regardless of NUM_NODES; nodes beyond 3 are
        #   simply omitted. Columns for nodes 1–3 are always written.
        with open(self.metrics_file, "w", newline="") as fh:
            csv.writer(fh).writerow([
                "Round",
                "Global_Accuracy", "Global_F1",
                "Node1_Rep", "Node1_LocalF1",
                "Node2_Rep", "Node2_LocalF1",
                "Node3_Rep", "Node3_LocalF1",
                "node1_verified", "node2_verified", "node3_verified",
                "Clients_Aggregated", "Clients_Penalised",
                "Encryption_Overhead_ms",
                "Epsilon",
            ])

        print(f">>> [Server] Metrics CSV: {self.metrics_file}")
        print(f">>> [Server] Min clients before start: {self.min_available_clients}")
        print(f">>> [Server] Reputation penalty: −{REPUTATION_PENALTY}  "
              f"if cosine_sim < {COSINE_SIM_THRESHOLD}")

    # ------------------------------------------------------------------
    # Required Strategy interface methods
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager):
        """No server-side initialisation — clients initialise their own models."""
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        sample = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        fit_ins = fl.common.FitIns(parameters or ndarrays_to_parameters([]), {})
        return [(client, fit_ins) for client in sample]

    def configure_evaluate(self, server_round, parameters, client_manager):
        sample = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_available_clients,
        )
        eval_ins = fl.common.EvaluateIns(parameters or ndarrays_to_parameters([]), {})
        return [(client, eval_ins) for client in sample]

    # ------------------------------------------------------------------
    # Core: aggregate_fit
    # Called by Flower after collecting fit() results from all clients.
    # ------------------------------------------------------------------

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        print(f"\n{'='*65}")
        print(f"  Round {server_round} — Weighted Aggregation (F1 × Reputation)")
        print(f"{'='*65}")

        # ── STEP 1: Signature verification + PQC decryption ─────────────
        t_dec_start = time.perf_counter()

        verified_weights:  list[list[np.ndarray]] = []
        client_ids:        list[int]               = []
        client_f1s:        list[float]             = []
        client_epsilons:   list[float]             = []
        sig_fail_ids:      list[int]               = []
        per_node_sig:      dict[int, bool]         = {}

        for idx, (client_proxy, fit_res) in enumerate(results):
            cid = idx + 1

            ndarrays        = parameters_to_ndarrays(fit_res.parameters)
            encrypted_bytes = ndarrays[0].tobytes()
            signature       = ndarrays[1].tobytes() if len(ndarrays) > 1 else b""

            # ── Fetch signing_pk from client properties ──────────────────
            try:
                prop_res = client_proxy.get_properties(fl.common.GetPropertiesIns(config={}), timeout=30)
                pub_hex = prop_res.properties.get("signing_pub_key", "")
                client_signing_pk = bytes.fromhex(pub_hex) if pub_hex else b""
            except Exception as e:
                print(f"  [Verify] ⚠ Node {cid} failed to fetch signing_pk: {e}")
                client_signing_pk = b""

            sig_ok = False
            if client_signing_pk:
                try:
                    sig_ok = verify_weights(encrypted_bytes, signature, client_signing_pk)
                except ValueError as e:
                    sig_ok = False
                except Exception as exc:
                    print(f"  [Verify] ⚠ Node {cid} signature decode error: {exc}")
            else:
                print(f"  [Verify] ⚠ Node {cid}: public-key missing")

            per_node_sig[cid] = sig_ok

            if sig_ok:
                print(f"  [Verify] ✓ Node {cid} update verified — accepted")
            else:
                print(f"  [Verify] ✗ Node {cid} signature verification FAILED — update rejected")
                sig_fail_ids.append(cid)
                # Immediate reputation hit for a failed signature (set to 0)
                self.reputation.scores[cid] = 0.0
                continue

            # ── PQC decryption (Kyber512) ────────────────────────────────
            dec_weights = decrypt_weights(encrypted_bytes, self.server_sec)

            local_f1   = float(fit_res.metrics.get("local_f1",  0.5))
            local_loss = float(fit_res.metrics.get("local_loss", 1.0))
            dp_epsilon = float(fit_res.metrics.get("dp_epsilon", 0.0))

            verified_weights.append(dec_weights)
            client_ids.append(cid)
            client_f1s.append(local_f1)
            client_epsilons.append(dp_epsilon)

            print(
                f"  [Node {cid}] Decrypted OK  "
                f"local_f1={local_f1:.4f}  local_loss={local_loss:.4f}  "
                f"examples={fit_res.num_examples}"
            )

        dec_ms = (time.perf_counter() - t_dec_start) * 1000
        print(f"\n>>> [Server] Verified+Decrypted {len(verified_weights)}/{len(results)} "
              f"updates  ({len(sig_fail_ids)} failed)  in {dec_ms:.1f} ms")

        if not verified_weights:
            print("[ERROR] No clients passed signature verification — aborting round.")
            return None, {}

        # ── STEP 2: Compute weight deltas for cosine-similarity check ────
        #   On round 1 we have no previous global weights, so we use the
        #   raw weights themselves as the "delta".
        if self._last_global_weights is not None:
            weight_deltas = [
                [w - g for w, g in zip(cw, self._last_global_weights)]
                for cw in verified_weights
            ]
        else:
            weight_deltas = verified_weights  # Round 1 fallback

        # ── STEP 3: Reputation update (cosine similarity check) ──────────
        #   This is the core reputation rule:
        #     "reputation_score decreases by 0.15 if that node's update
        #      cosine similarity to the median update is below 0.7"
        penalised = self.reputation.update(server_round, client_ids, weight_deltas)
        n_penalised = sum(penalised.values())

        # ── STEP 4: Filter out clients below the reputation floor ─────────
        trusted_weights: list[list[np.ndarray]] = []
        trusted_ids:     list[int]               = []
        trusted_f1s:     list[float]             = []

        for cid, weights, f1 in zip(client_ids, verified_weights, client_f1s):
            if not self.reputation.is_trusted(cid):
                print(
                    f"  [Reputation] ✗ Node {cid} EXCLUDED from aggregation "
                    f"(rep={self.reputation.scores[cid]:.4f} < floor={REPUTATION_FLOOR})"
                )
                continue
            trusted_weights.append(weights)
            trusted_ids.append(cid)
            trusted_f1s.append(f1)

        if not trusted_weights:
            print("[WARNING] All clients below reputation floor — including all as fallback.")
            trusted_weights = verified_weights
            trusted_ids     = client_ids
            trusted_f1s     = client_f1s

        # ── STEP 5: Weighted aggregation — w_i = (f1_i × rep_i) / Σ ─────
        #   This is the central aggregation step that replaces plain FedAvg.
        global_weights = f1_reputation_weighted_fedavg(
            trusted_weights, trusted_ids, trusted_f1s, self.reputation
        )
        self._last_global_weights = global_weights

        # ── STEP 6: Re-encrypt global model per client before broadcast ──
        t_enc_start = time.perf_counter()
        bundled_encrypted: list[np.ndarray] = []

        for node_id in range(1, config.NUM_NODES + 1):
            client_pub_path = self.keys_dir / f"client_{node_id}_pub.pem"
            # Wait for client key file (written by fl_client.py on startup)
            waited = 0
            while not client_pub_path.exists() and waited < 60:
                time.sleep(1)
                waited += 1
            with open(client_pub_path, "rb") as fh:
                client_pub = fh.read()
            enc_bytes = encrypt_weights(global_weights, client_pub)
            bundled_encrypted.append(np.frombuffer(enc_bytes, dtype=np.uint8))

        enc_ms = (time.perf_counter() - t_enc_start) * 1000
        self._last_overhead_ms = dec_ms + enc_ms
        print(f">>> [Server] Re-encrypted for {config.NUM_NODES} clients in {enc_ms:.1f} ms")

        # ── STEP 7: Cache per-node metrics for CSV logging ───────────────
        #   aggregate_evaluate() will write the CSV row; we cache fit data
        #   here so reputation/F1 is available at that point.
        per_node_f1 = {cid: f1 for cid, f1 in zip(client_ids, client_f1s)}
        round_epsilon = max(client_epsilons) if client_epsilons else 0.0
        self._round_fit_cache[server_round] = {
            "per_node_f1":  per_node_f1,
            "per_node_sig": per_node_sig,
            "n_aggregated": len(trusted_weights),
            "n_penalised":  n_penalised,
            "epsilon":      round_epsilon,
        }

        agg_metrics = {
            "clients_aggregated": len(trusted_weights),
            "clients_penalised":  n_penalised,
        }
        return ndarrays_to_parameters(bundled_encrypted), agg_metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate — called after clients return evaluate() results
    # ------------------------------------------------------------------

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        if total_examples == 0:
            return None, {}

        # Weighted loss
        loss_agg = sum(r.loss * r.num_examples for _, r in results) / total_examples

        # Aggregate accuracy and F1 using the custom fn, or simple weighted mean
        if self.eval_metrics_agg_fn:
            agg_metrics = self.eval_metrics_agg_fn(
                [(r.num_examples, r.metrics) for _, r in results]
            )
        else:
            agg_metrics = {}

        global_accuracy = float(agg_metrics.get("accuracy", 0.0))
        global_f1       = float(agg_metrics.get("f1",       0.0))

        # Pull fit-phase cache for this round
        fit_cache    = self._round_fit_cache.get(server_round, {})
        per_node_f1  = fit_cache.get("per_node_f1",  {})
        per_node_sig = fit_cache.get("per_node_sig", {})
        n_agg        = fit_cache.get("n_aggregated",  len(results))
        n_penalised  = fit_cache.get("n_penalised",   0)
        round_eps    = fit_cache.get("epsilon",       0.0)
        overhead_ms  = self._last_overhead_ms

        # ── Print round summary ──────────────────────────────────────────
        print(f"\n{'*'*65}")
        print(f"  Round {server_round} Complete")
        print(f"  Global Loss:          {loss_agg:.4f}")
        print(f"  Global Accuracy:      {global_accuracy:.4f}")
        print(f"  Global F1-Score:      {global_f1:.4f}")
        print(f"  Clients Aggregated:   {n_agg}")
        print(f"  Clients Penalised:    {n_penalised}")
        print(f"  Encryption Overhead:  {overhead_ms:.1f} ms")

        # ── Print per-node reputation scores (rule 4) ────────────────────
        print("\n  ┌─ Per-Node Reputation Scores (after this round) ─────────┐")
        for nid in sorted(self.reputation.scores):
            rep     = self.reputation.scores[nid]
            f1_val  = per_node_f1.get(nid, float("nan"))
            status  = "⚠ PENALISED" if nid in per_node_f1 and rep < REPUTATION_INIT else "OK"
            below   = " [BELOW FLOOR]" if rep < REPUTATION_FLOOR else ""
            print(f"  │  Node {nid}: rep={rep:.4f}  local_f1={f1_val:.4f}  {status}{below}")
        print("  └──────────────────────────────────────────────────────────┘")
        print(f"{'*'*65}\n")

        # ── Write CSV row ────────────────────────────────────────────────
        #   Fixed 3-node columns; missing nodes get 'N/A'
        node_cols: list = []
        sig_cols: list = []
        for nid in [1, 2, 3]:
            rep = self.reputation.scores.get(nid, float("nan"))
            f1v = per_node_f1.get(nid, float("nan"))
            node_cols += [
                round(rep, 4) if not np.isnan(rep) else "N/A",
                round(f1v, 4) if not np.isnan(f1v) else "N/A",
            ]
            if nid in per_node_sig:
                sig_cols.append("True" if per_node_sig[nid] else "False")
            else:
                sig_cols.append("N/A")

        with open(self.metrics_file, "a", newline="") as fh:
            csv.writer(fh).writerow([
                server_round,
                round(global_accuracy, 4),
                round(global_f1, 4),
                *node_cols,                     # Node1_Rep, Node1_F1, Node2_Rep, ...
                *sig_cols,                      # Node1_SigPass, ...
                n_agg,
                n_penalised,
                round(overhead_ms, 2),
                round(round_eps, 6),
            ])

        return loss_agg, agg_metrics

    # ------------------------------------------------------------------
    # evaluate — optional server-side evaluation (not used)
    # ------------------------------------------------------------------

    def evaluate(self, server_round, parameters):
        return None


# ===========================================================================
# Global metrics aggregation helper (passed to strategy)
# ===========================================================================

def evaluate_metrics_aggregation_fn(metrics):
    """Weighted average of accuracy and F1 across all clients."""
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0, "f1": 0.0}
    accuracy = sum(n * m["accuracy"] for n, m in metrics) / total_examples
    f1       = sum(n * m["f1"]       for n, m in metrics) / total_examples
    return {"accuracy": accuracy, "f1": f1}


# ===========================================================================
# Server entry point
# ===========================================================================

def start_server():
    print("=" * 65)
    print("  PQC-IoT Sentinel — Federated Learning Server")
    print("  Strategy : F1 × Reputation Weighted Aggregation")
    print(f"  Port     : 0.0.0.0:8080")
    print(f"  Rounds   : {NUM_ROUNDS}")
    print(f"  Min clients before start: {MIN_FIT_CLIENTS}")
    print(f"  Reputation penalty : −{REPUTATION_PENALTY} if cosine_sim < {COSINE_SIM_THRESHOLD}")
    print(f"  Reputation floor   : {REPUTATION_FLOOR} (excluded from aggregation)")
    print("=" * 65)

    strategy = SecureWeightedFedAvg(
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_FIT_CLIENTS,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
