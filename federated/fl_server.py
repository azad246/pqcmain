"""
federated/fl_server.py
======================
Secure FL Server with:
  1. Weighted FedAvg — aggregates client model updates weighted by each
     client's local F1 score (sent from fl_client.py via fit() metrics).
  2. Reputation System — each round, computes the median L2 norm of all
     client weight deltas.  Any client whose update deviates beyond a
     configurable z-score threshold gets a reputation penalty.
     Clients whose reputation falls below a floor threshold are excluded
     from the next round's aggregation.
  3. PQC encryption / decryption unchanged from original.
"""

import csv
import sys
import time
import json
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
# Reputation constants  (tune as needed)
# ---------------------------------------------------------------------------
REPUTATION_INIT          = 1.0   # starting reputation for every client
REPUTATION_PENALTY       = 0.15  # subtracted each round a client is flagged
REPUTATION_REWARD        = 0.05  # added each round a client behaves normally
REPUTATION_FLOOR         = 0.30  # below this → client excluded from aggregation
DEVIATION_ZSCORE_THRESH  = 2.5   # flag if |norm - median| > threshold * MAD


# ==============================================================================
# Weighted FedAvg aggregation helper (pure numpy — no Flower dependency)
# ==============================================================================

def weighted_fedavg(
    weight_list: list[list[np.ndarray]],
    scores: list[float],
) -> list[np.ndarray]:
    """
    Aggregate model weights by a weighted average driven by `scores`.

    Parameters
    ----------
    weight_list : list of weight arrays per client (each is a list of ndarrays)
    scores      : per-client quality score (e.g. local F1); must be >= 0

    Returns
    -------
    Aggregated weights as a list of ndarrays.
    """
    total = sum(scores)
    if total == 0:
        # Fall back to equal weighting if all scores are zero
        scores = [1.0] * len(scores)
        total  = float(len(scores))

    aggregated = []
    for layer_idx in range(len(weight_list[0])):
        layer_agg = np.zeros_like(weight_list[0][layer_idx], dtype=np.float64)
        for client_weights, score in zip(weight_list, scores):
            layer_agg += (score / total) * client_weights[layer_idx].astype(np.float64)
        aggregated.append(layer_agg)
    return aggregated


# ==============================================================================
# Reputation tracker
# ==============================================================================

class ReputationTracker:
    """
    Tracks a reputation score per client and flags outliers based on the
    L2 norm of their weight update relative to the median across all clients.
    """

    def __init__(self, client_ids: list[int]):
        self.scores: dict[int, float] = {cid: REPUTATION_INIT for cid in client_ids}
        self.history: list[dict]      = []   # one entry per round

    def update(
        self,
        server_round: int,
        client_ids: list[int],
        weight_updates: list[list[np.ndarray]],
    ) -> dict[int, bool]:
        """
        Compute per-client L2 norms, detect outliers, update scores.

        Returns
        -------
        flagged : dict mapping client_id → True if flagged this round
        """
        norms = np.array([
            float(np.sqrt(sum(np.sum(w ** 2) for w in upd)))
            for upd in weight_updates
        ])

        median_norm = float(np.median(norms))
        mad         = float(np.median(np.abs(norms - median_norm))) + 1e-9  # avoid /0

        flagged: dict[int, bool] = {}
        round_log = {"round": server_round, "clients": {}}

        for cid, norm in zip(client_ids, norms):
            z_score = abs(norm - median_norm) / mad
            is_flagged = z_score > DEVIATION_ZSCORE_THRESH

            if is_flagged:
                self.scores[cid] = max(0.0, self.scores[cid] - REPUTATION_PENALTY)
                print(f"  [Reputation] ⚠ Client {cid} FLAGGED  "
                      f"norm={norm:.4f}  z={z_score:.2f}  "
                      f"rep={self.scores[cid]:.2f}")
            else:
                self.scores[cid] = min(1.0, self.scores[cid] + REPUTATION_REWARD)

            flagged[cid] = is_flagged
            round_log["clients"][cid] = {
                "norm":      round(norm,           4),
                "z_score":   round(z_score,        4),
                "flagged":   is_flagged,
                "reputation": round(self.scores[cid], 4),
            }

        print(f"  [Reputation] Median norm={median_norm:.4f}  MAD={mad:.4f}")
        self.history.append(round_log)
        self._save_history()
        return flagged

    def is_trusted(self, client_id: int) -> bool:
        return self.scores.get(client_id, REPUTATION_INIT) >= REPUTATION_FLOOR

    def _save_history(self):
        out_path = config.RESULTS_PATH / 'reputation_log.json'
        config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, 'w') as f:
                json.dump(
                    {"reputation_floor": REPUTATION_FLOOR, "rounds": self.history},
                    f, indent=4
                )
        except Exception:
            pass


# ==============================================================================
# Strategy: Weighted FedAvg + PQC + Reputation
# ==============================================================================

class SecureWeightedFedAvg(fl.server.strategy.Strategy):
    """
    Custom FL strategy that:
      - Decrypts incoming weight updates (PQC / Kyber512)
      - Runs Weighted FedAvg using each client's reported local_f1
      - Penalises clients whose weight-update L2 norm is an outlier
      - Re-encrypts the global model per-client before broadcasting
    """

    def __init__(
        self,
        min_fit_clients: int    = config.NUM_NODES,
        min_evaluate_clients: int = config.NUM_NODES,
        min_available_clients: int = config.NUM_NODES,
        evaluate_metrics_aggregation_fn=None,
    ):
        self.min_fit_clients           = min_fit_clients
        self.min_evaluate_clients      = min_evaluate_clients
        self.min_available_clients     = min_available_clients
        self.eval_metrics_agg_fn       = evaluate_metrics_aggregation_fn
        self.metrics_file              = config.RESULTS_PATH / 'fl_round_metrics.csv'
        self.last_round_overhead       = 0.0

        config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)

        # PQC key generation
        self.keys_dir = config.BASE_DIR / 'keys'
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        print(">>> [Server] Generating Server PQC Keypair...")
        self.server_pub, self.server_sec = pqc_crypto.generate_keypair()
        with open(self.keys_dir / 'server_pub.pem', 'wb') as f:
            f.write(self.server_pub)

        # Reputation tracker — one entry per expected client node
        self.reputation = ReputationTracker(list(range(1, config.NUM_NODES + 1)))

        # CSV header
        with open(self.metrics_file, 'w', newline='') as f:
            csv.writer(f).writerow([
                'Round', 'Global_Loss', 'Global_Accuracy', 'Global_F1_Score',
                'Encryption_Overhead_ms', 'Clients_Aggregated', 'Clients_Flagged',
                'Mean_DP_Epsilon',
            ])

        # Track per-round mean DP epsilon across clients
        self.last_round_mean_epsilon: float = 0.0

        # Keep last global weights for delta computation
        self._last_global_weights: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Required Strategy interface methods
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager):
        return None   # clients initialise their own models

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
    # Core: Weighted FedAvg aggregation with reputation gating
    # ------------------------------------------------------------------

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        print(f"\n--- Round {server_round} Weighted FedAvg + PQC + Dilithium2 Verification ---")

        # ── STEP 1: Signature verification + Decryption ────────────────
        start_dec = time.perf_counter()
        decrypted_weights_per_client: list[list[np.ndarray]] = []
        client_ids:    list[int]   = []
        client_f1s:    list[float] = []
        raw_results_map: list[tuple] = []
        sig_failures:  list[int]   = []   # client IDs whose signature failed

        for idx, (client_proxy, fit_res) in enumerate(results):
            cid = idx + 1
            ndarrays        = parameters_to_ndarrays(fit_res.parameters)
            encrypted_bytes = ndarrays[0].tobytes()

            # ── Dilithium2 signature verification ────────────────────────
            sig_hex     = fit_res.metrics.get("signature",       "")
            pub_hex     = fit_res.metrics.get("signing_pub_key", "")

            sig_ok = False
            if sig_hex and pub_hex:
                try:
                    signature   = bytes.fromhex(sig_hex)
                    signing_pub = bytes.fromhex(pub_hex)
                    sig_ok      = verify_weights(encrypted_bytes, signature, signing_pub)
                except Exception as exc:
                    print(f"  [Verify] ⚠ Node {cid} signature decode error: {exc}")
            else:
                print(f"  [Verify] ⚠ Node {cid}: signature or public key missing from metrics")

            if sig_ok:
                print(f"  [Verify] ✓ Node {cid} signature VALID")
            else:
                print(f"  [Verify] ✗ Node {cid} signature INVALID — dropping update")
                sig_failures.append(cid)
                # Apply an immediate reputation penalty for failed verification
                self.reputation.scores[cid] = max(
                    0.0, self.reputation.scores[cid] - 0.25
                )
                continue   # skip this client entirely — do not decrypt or aggregate

            # ── PQC decryption (only for verified clients) ──────────────
            dec_weights = decrypt_weights(encrypted_bytes, self.server_sec)

            local_f1   = float(fit_res.metrics.get("local_f1",   0.5))
            local_loss = float(fit_res.metrics.get("local_loss",  1.0))
            dp_epsilon = float(fit_res.metrics.get("dp_epsilon",  0.0))
            dp_delta   = float(fit_res.metrics.get("dp_delta",    1e-5))

            decrypted_weights_per_client.append(dec_weights)
            client_ids.append(cid)
            client_f1s.append(local_f1)
            raw_results_map.append((client_proxy, fit_res, dec_weights, cid))

            print(f"  [Node {cid}] Decrypted  local_f1={local_f1:.4f}  "
                  f"local_loss={local_loss:.4f}  "
                  f"DP ε={dp_epsilon:.4f} (δ={dp_delta})  "
                  f"examples={fit_res.num_examples}")

        dec_time = (time.perf_counter() - start_dec) * 1000
        verified_count = len(client_ids)
        print(f">>> [Server] Verified+Decrypted {verified_count}/{len(results)} updates  "
              f"({len(sig_failures)} failed sig verification)  in {dec_time:.2f} ms")

        if not decrypted_weights_per_client:
            print("[ERROR] No clients passed signature verification — aborting round.")
            return None, {}

        # ── STEP 2: Compute weight deltas & run reputation check ───────
        if self._last_global_weights is not None:
            weight_deltas = [
                [w - g for w, g in zip(client_w, self._last_global_weights)]
                for client_w in decrypted_weights_per_client
            ]
        else:
            weight_deltas = decrypted_weights_per_client   # Round 1: use raw weights

        print(f"\n  [Reputation] Checking {len(client_ids)} clients (Round {server_round})...")
        flagged = self.reputation.update(server_round, client_ids, weight_deltas)

        # ── STEP 3: Filter out untrusted clients ───────────────────────
        trusted_weights: list[list[np.ndarray]] = []
        trusted_f1s:     list[float]            = []
        clients_flagged = 0

        for cid, weights, f1 in zip(client_ids, decrypted_weights_per_client, client_f1s):
            if not self.reputation.is_trusted(cid):
                print(f"  [Reputation] ✗ Client {cid} EXCLUDED "
                      f"(reputation={self.reputation.scores[cid]:.2f} < floor={REPUTATION_FLOOR})")
                clients_flagged += 1
                continue
            if flagged.get(cid, False):
                clients_flagged += 1   # flagged but still above floor — count but include
            trusted_weights.append(weights)
            trusted_f1s.append(f1)

        if not trusted_weights:
            print("[WARNING] All clients excluded by reputation system — using all clients as fallback.")
            trusted_weights = decrypted_weights_per_client
            trusted_f1s     = client_f1s

        # ── STEP 4: Weighted FedAvg ────────────────────────────────────
        print(f"\n  [WeightedFedAvg] Aggregating {len(trusted_weights)} trusted clients")
        for cid, f1 in zip(client_ids, trusted_f1s):
            print(f"    Node {cid}: weight={f1:.4f}  (local_f1 as aggregation weight)")

        global_weights = weighted_fedavg(trusted_weights, trusted_f1s)
        self._last_global_weights = global_weights

        # ── STEP 5: Re-encrypt for each client (broadcast) ─────────────
        start_enc = time.perf_counter()
        bundled_encrypted = []

        for node_id in range(1, config.NUM_NODES + 1):
            client_pub_path = self.keys_dir / f'client_{node_id}_pub.pem'
            while not client_pub_path.exists():
                time.sleep(1)
            with open(client_pub_path, 'rb') as f:
                client_pub = f.read()
            enc_bytes = encrypt_weights(global_weights, client_pub)
            bundled_encrypted.append(np.frombuffer(enc_bytes, dtype=np.uint8))

        enc_time = (time.perf_counter() - start_enc) * 1000
        print(f">>> [Server] Re-encrypted for {config.NUM_NODES} clients in {enc_time:.2f} ms")

        self.last_round_overhead = dec_time + enc_time

        # Compute mean DP epsilon across all clients this round
        epsilons = [float(r.metrics.get("dp_epsilon", 0.0)) for _, r in results]
        self.last_round_mean_epsilon = float(np.mean(epsilons)) if epsilons else 0.0
        if epsilons:
            print(f"  [DP] Mean ε this round: {self.last_round_mean_epsilon:.4f}  "
                  f"(min={min(epsilons):.4f}  max={max(epsilons):.4f})")

        agg_metrics = {
            "clients_aggregated": len(trusted_weights),
            "clients_flagged":    clients_flagged,
            "mean_dp_epsilon":    self.last_round_mean_epsilon,
        }
        return ndarrays_to_parameters(bundled_encrypted), agg_metrics

    # ------------------------------------------------------------------
    # Evaluation aggregation (unchanged logic, extra CSV column)
    # ------------------------------------------------------------------

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        if total_examples == 0:
            return None, {}

        loss_agg = sum(r.loss * r.num_examples for _, r in results) / total_examples

        if self.eval_metrics_agg_fn:
            agg_metrics = self.eval_metrics_agg_fn(
                [(r.num_examples, r.metrics) for _, r in results]
            )
        else:
            agg_metrics = {}

        accuracy = agg_metrics.get("accuracy", 0.0)
        f1       = agg_metrics.get("f1",       0.0)
        overhead = self.last_round_overhead

        clients_agg     = len(results)
        clients_flagged = sum(
            1 for cid, score in self.reputation.scores.items()
            if score < REPUTATION_INIT
        )

        print(f"\n{'*'*55}")
        print(f"--- Round {server_round} Complete ---")
        mean_epsilon = getattr(self, 'last_round_mean_epsilon', 0.0)

        print(f"Global Loss:            {loss_agg:.4f}")
        print(f"Global Accuracy:        {accuracy:.4f}")
        print(f"Global F1-Score:        {f1:.4f}")
        print(f"Encryption Overhead:    {overhead:.2f} ms")
        print(f"Clients aggregated:     {clients_agg}")
        print(f"Clients with rep < 1.0: {clients_flagged}")
        print(f"Mean DP ε (this round): {mean_epsilon:.4f}")
        print(f"Reputation scores:      {self.reputation.scores}")
        print(f"{'*'*55}\n")

        with open(self.metrics_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                server_round, loss_agg, accuracy, f1,
                overhead, clients_agg, clients_flagged,
                round(mean_epsilon, 6),
            ])

        return loss_agg, agg_metrics

    # ------------------------------------------------------------------
    # evaluate() — server-side model evaluation (optional, can be None)
    # ------------------------------------------------------------------

    def evaluate(self, server_round, parameters):
        return None


# ==============================================================================
# Global metrics aggregation function (for evaluate phase)
# ==============================================================================

def evaluate_metrics_aggregation_fn(metrics):
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0, "f1": 0.0}
    accuracies = [n * m["accuracy"] for n, m in metrics]
    f1s        = [n * m["f1"]       for n, m in metrics]
    return {
        "accuracy": sum(accuracies) / total_examples,
        "f1":       sum(f1s)        / total_examples,
    }


# ==============================================================================
# Server entry point
# ==============================================================================

def start_server():
    print("=" * 65)
    print("Starting Secure PQC Flower FL Server")
    print("Strategy: SecureWeightedFedAvg (Weighted FedAvg + Reputation + PQC KEM)")
    print(f"Expected Clients: {config.NUM_NODES}")
    print(f"Reputation Floor: {REPUTATION_FLOOR}  |  Deviation Threshold (z): {DEVIATION_ZSCORE_THRESH}")
    print("=" * 65)

    strategy = SecureWeightedFedAvg(
        min_fit_clients=config.NUM_NODES,
        min_evaluate_clients=config.NUM_NODES,
        min_available_clients=config.NUM_NODES,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config.FL_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()
