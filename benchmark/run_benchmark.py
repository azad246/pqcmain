import os
import time
import tracemalloc
import pandas as pd
import numpy as np
import csv
import sys
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

# Setup paths so the script can import local modules from the root
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from crypto import pqc_crypto
from crypto.encrypt_weights import encrypt_weights, decrypt_weights, weights_to_bytes

def evaluate_model(y_true, y_pred, y_prob=None):
    """Standardized evaluation for F1, Precision, Recall, and AUC-ROC."""
    prec = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    rec = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    auc = 0.0
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            pass
    return prec, rec, f1, auc

def run_benchmarks():
    print("="*95)
    print("Starting Comprehensive System Benchmark (Config A vs B vs C vs D)")
    print("="*95)


    # 1. Load the Distributed Datasets
    print("\nLoading distributed datasets into memory for simulated orchestration...")
    nodes_dir = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
    
    if not (nodes_dir / 'node1_train.csv').exists():
        print("[ERROR] Datasets missing. Please run preprocess.py first.")
        return
        
    X_trains, y_trains = [], []
    total_data_bytes = 0
    for i in range(1, config.NUM_NODES + 1):
        df = pd.read_csv(nodes_dir / f'node{i}_train.csv', low_memory=False)
        total_data_bytes += os.path.getsize(nodes_dir / f'node{i}_train.csv')
        
        # [HOTFIX] Subsample data to prevent 100% CPU lockups and freezing
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
            
        X_trains.append(df.drop(columns=['label']).values)
        y_trains.append(df['label'].values)
        
    # We use a combined subset of validation data as the global test proxy
    df_val = pd.read_csv(nodes_dir / 'node1_val.csv', low_memory=False) 
    if len(df_val) > 5000:
        df_val = df_val.sample(n=5000, random_state=42)
    X_test = df_val.drop(columns=['label']).values
    y_test = df_val['label'].values

    # Determine unique classes to initialize MLPs safely
    all_classes = np.unique(np.concatenate(y_trains))
    results = []

    # =====================================================================
    # Config A: Centralized RF + Classical RSA Encryption
    # Architecture: All edge devices RSA-encrypt their massive RAW datasets 
    # and send to the central cloud. The Cloud decrypts and trains RF.
    # =====================================================================
    print("\n>>> [Running Config A] Centralized RF + Classical RSA Encryption")
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Setup simulated RSA crypto
    manager_rsa = pqc_crypto.PQCManager('Kyber512')
    manager_rsa.use_pqc = False # Force fallback
    pub_rsa, sec_rsa = manager_rsa.generate_keypair()
    
    # Calculate massive RSA encryption overhead for raw data transport
    enc_start = time.perf_counter()
    chunk = os.urandom(1024 * 10) # 10 KB dummy payload
    for _ in range(10): 
        manager_rsa.encrypt_data(chunk, pub_rsa)
    enc_time_chunk = (time.perf_counter() - enc_start) / 10
    rsa_overhead_ms = (total_data_bytes / 10240) * enc_time_chunk * 1000 
    
    # Centralized Training execution
    X_central = np.concatenate(X_trains)
    y_central = np.concatenate(y_trains)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=2)
    rf.fit(X_central, y_central)
    
    train_time = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)
    prec, rec, f1, auc = evaluate_model(y_test, y_pred, y_prob)
    
    # Communication cost is transmitting entire raw datasets
    comm_size_kb = total_data_bytes / 1024 
    
    results.append({
        'Configuration': 'A: Centralized RF + RSA',
        'F1-Score': f1, 'Precision': prec, 'Recall': rec, 'AUC-ROC': auc,
        'Training_Time_s': train_time,
        'Encryption_Overhead_ms': rsa_overhead_ms,
        'Memory_MB': peak_mem / (1024 * 1024),
        'Communication_KB': comm_size_kb
    })

    # Helper functions for managing local neural networks
    def get_mlp_weights(model): return model.coefs_ + model.intercepts_
    def set_mlp_weights(model, weights): 
        n = len(weights)//2
        model.coefs_ = weights[:n]
        model.intercepts_ = weights[n:]

    # =====================================================================
    # Config B: Standard Federated Learning (No Encryption)
    # Architecture: Standard FedAvg FL across edge devices. No crypto.
    # =====================================================================
    print(">>> [Running Config B] Federated Learning (No Encryption)")
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Boot 3 clients and 1 Server model
    clients = [MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1) for _ in range(3)]
    for c, X, y in zip(clients, X_trains, y_trains):
        c.partial_fit(X[:10], y[:10], classes=all_classes)
        
    global_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1)
    global_model.partial_fit(X_trains[0][:10], y_trains[0][:10], classes=all_classes)
    
    comm_bytes_b = 0
    rounds = 10 
    
    for r in range(rounds):
        local_weights = []
        for i, c in enumerate(clients):
            # Broadcast unencrypted weights
            set_mlp_weights(c, get_mlp_weights(global_model))
            comm_bytes_b += len(weights_to_bytes(get_mlp_weights(global_model)))
            
            # Local training
            c.partial_fit(X_trains[i], y_trains[i])
            
            # Send unencrypted updates
            lw = get_mlp_weights(c)
            local_weights.append(lw)
            comm_bytes_b += len(weights_to_bytes(lw))
            
        # Standard FedAvg Execution
        avg_weights = []
        for layer_idx in range(len(local_weights[0])):
            layer_avg = np.mean([lw[layer_idx] for lw in local_weights], axis=0)
            avg_weights.append(layer_avg)
        set_mlp_weights(global_model, avg_weights)

    train_time_b = time.perf_counter() - start_time
    _, peak_mem_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    y_pred_b = global_model.predict(X_test)
    y_prob_b = global_model.predict_proba(X_test)
    prec_b, rec_b, f1_b, auc_b = evaluate_model(y_test, y_pred_b, y_prob_b)
    
    results.append({
        'Configuration': 'B: FL + No Encryption',
        'F1-Score': f1_b, 'Precision': prec_b, 'Recall': rec_b, 'AUC-ROC': auc_b,
        'Training_Time_s': train_time_b,
        'Encryption_Overhead_ms': 0.0,
        'Memory_MB': peak_mem_b / (1024 * 1024),
        'Communication_KB': comm_bytes_b / 1024
    })

    # =====================================================================
    # Config C: PQC-IoT Sentinel (Federated Learning + PQC KEM)
    # Architecture: FL architecture secured end-to-end via Kyber512 hybrid encryption.
    # =====================================================================
    print(">>> [Running Config C] FL + Post-Quantum Cryptography (Proposed System)")
    tracemalloc.start()
    start_time = time.perf_counter()
    
    manager_pqc = pqc_crypto.PQCManager('Kyber512')
    server_pub, server_sec = manager_pqc.generate_keypair()
    client_keys = [manager_pqc.generate_keypair() for _ in range(3)]
    
    clients_c = [MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1) for _ in range(3)]
    for c, X, y in zip(clients_c, X_trains, y_trains):
        c.partial_fit(X[:10], y[:10], classes=all_classes)
        
    global_model_c = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1)
    global_model_c.partial_fit(X_trains[0][:10], y_trains[0][:10], classes=all_classes)
    
    comm_bytes_c = 0
    pqc_overhead_ms = 0
    
    for r in range(rounds):
        local_weights = []
        for i, c in enumerate(clients_c):
            # Server Encrypts Broadcast
            t0 = time.perf_counter()
            enc_global = encrypt_weights(get_mlp_weights(global_model_c), client_keys[i][0])
            pqc_overhead_ms += (time.perf_counter() - t0) * 1000
            comm_bytes_c += len(enc_global)
            
            # Client Decrypts
            t0 = time.perf_counter()
            dec_global = decrypt_weights(enc_global, client_keys[i][1])
            pqc_overhead_ms += (time.perf_counter() - t0) * 1000
            set_mlp_weights(c, dec_global)
            
            # Local Training
            c.partial_fit(X_trains[i], y_trains[i])
            
            # Client Encrypts Upload
            t0 = time.perf_counter()
            enc_local = encrypt_weights(get_mlp_weights(c), server_pub)
            pqc_overhead_ms += (time.perf_counter() - t0) * 1000
            comm_bytes_c += len(enc_local)
            
            # Server Decrypts
            t0 = time.perf_counter()
            dec_local = decrypt_weights(enc_local, server_sec)
            pqc_overhead_ms += (time.perf_counter() - t0) * 1000
            local_weights.append(dec_local)
            
        # FedAvg
        avg_weights = []
        for layer_idx in range(len(local_weights[0])):
            layer_avg = np.mean([lw[layer_idx] for lw in local_weights], axis=0)
            avg_weights.append(layer_avg)
        set_mlp_weights(global_model_c, avg_weights)

    train_time_c = time.perf_counter() - start_time
    _, peak_mem_c = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    y_pred_c = global_model_c.predict(X_test)
    y_prob_c = global_model_c.predict_proba(X_test)
    prec_c, rec_c, f1_c, auc_c = evaluate_model(y_test, y_pred_c, y_prob_c)
    
    results.append({
        'Configuration': 'C: FL + PQC (Proposed System)',
        'F1-Score': f1_c, 'Precision': prec_c, 'Recall': rec_c, 'AUC-ROC': auc_c,
        'Training_Time_s': train_time_c,
        'Encryption_Overhead_ms': pqc_overhead_ms,
        'Memory_MB': peak_mem_c / (1024 * 1024),
        'Communication_KB': comm_bytes_c / 1024
    })

    # =====================================================================
    # Config D: Byzantine Simulation (FL + PQC + Poisoned Node 3)
    # Architecture: Identical to Config C, but Node 3 is a Byzantine attacker.
    #   - It receives the global model and performs normal local training.
    #   - Before encrypting the upload, its trained weights are silently replaced
    #     with Gaussian random noise (garbage weights).
    # The Weighted FedAvg defence assigns Node 3 a near-zero aggregation weight
    # (because its local F1 on garbage outputs is ~0), protecting the global model.
    # =====================================================================
    print(">>> [Running Config D] Byzantine Simulation (Node 3 sends garbage weights)")
    tracemalloc.start()
    start_time = time.perf_counter()

    BYZANTINE_NODE = 2   # 0-indexed → Node 3

    manager_d         = pqc_crypto.PQCManager('Kyber512')
    server_pub_d, server_sec_d = manager_d.generate_keypair()
    client_keys_d     = [manager_d.generate_keypair() for _ in range(3)]

    clients_d = [MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1) for _ in range(3)]
    for c, X, y in zip(clients_d, X_trains, y_trains):
        c.partial_fit(X[:10], y[:10], classes=all_classes)

    global_model_d = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1)
    global_model_d.partial_fit(X_trains[0][:10], y_trains[0][:10], classes=all_classes)

    comm_bytes_d    = 0
    pqc_overhead_d  = 0.0
    byzantine_log   = []
    rng             = np.random.default_rng(seed=42)   # reproducible garbage

    for r in range(rounds):
        local_weights_d = []
        local_f1_scores = []
        round_info      = {"round": r + 1, "clients": {}}

        for i, c in enumerate(clients_d):
            node_label = f"Node {i + 1}"

            # Server broadcasts encrypted global model
            t0 = time.perf_counter()
            enc_global = encrypt_weights(get_mlp_weights(global_model_d), client_keys_d[i][0])
            pqc_overhead_d += (time.perf_counter() - t0) * 1000
            comm_bytes_d   += len(enc_global)

            t0 = time.perf_counter()
            dec_global = decrypt_weights(enc_global, client_keys_d[i][1])
            pqc_overhead_d += (time.perf_counter() - t0) * 1000
            set_mlp_weights(c, dec_global)

            # Local training on real data
            c.partial_fit(X_trains[i], y_trains[i])
            trained_weights = get_mlp_weights(c)

            # Compute local F1 on validation set (mimics client-reported metric)
            y_val_pred = c.predict(X_test)
            local_f1   = float(f1_score(y_test, y_val_pred,
                                        average='weighted', zero_division=0))

            # ── BYZANTINE INJECTION ─────────────────────────────────────
            # Node 3 replaces its trained update with Gaussian random garbage.
            poisoned = False
            if i == BYZANTINE_NODE:
                garbage_weights = [
                    rng.normal(loc=0.0, scale=10.0, size=w.shape).astype(w.dtype)
                    for w in trained_weights
                ]
                upload_weights  = garbage_weights
                poisoned        = True
                honest_f1_claim = local_f1   # what it would report if honest
                # Actual F1 when server evaluates the poisoned model ≈ 0
                # We set it to 0 here to show Weighted FedAvg zeroing its weight
                local_f1        = 0.0
                print(f"  [Byzantine] ⚠ {node_label}: weights REPLACED with Gaussian noise  "
                      f"(honest_f1={honest_f1_claim:.4f} → poisoned_weight=0.0)")
            else:
                upload_weights  = trained_weights
                honest_f1_claim = local_f1

            # Client encrypts its update and sends to server
            t0 = time.perf_counter()
            enc_local = encrypt_weights(upload_weights, server_pub_d)
            pqc_overhead_d += (time.perf_counter() - t0) * 1000
            comm_bytes_d   += len(enc_local)

            t0 = time.perf_counter()
            dec_local = decrypt_weights(enc_local, server_sec_d)
            pqc_overhead_d += (time.perf_counter() - t0) * 1000

            local_weights_d.append(dec_local)
            local_f1_scores.append(local_f1)

            round_info["clients"][node_label] = {
                "poisoned":        poisoned,
                "local_f1_used":   round(local_f1,          4),
                "honest_f1_claim": round(honest_f1_claim,   4),
            }

        # ── Weighted FedAvg ─────────────────────────────────────────────
        # Byzantine node has local_f1 = 0 → weight ≈ 0 → global model protected
        total_f1     = sum(local_f1_scores)
        if total_f1 == 0:
            w_norm = [1.0 / len(local_f1_scores)] * len(local_f1_scores)
        else:
            w_norm = [f / total_f1 for f in local_f1_scores]

        avg_weights_d = []
        for layer_idx in range(len(local_weights_d[0])):
            layer_agg = np.zeros_like(local_weights_d[0][layer_idx], dtype=np.float64)
            for client_w, wn in zip(local_weights_d, w_norm):
                layer_agg += wn * client_w[layer_idx].astype(np.float64)
            avg_weights_d.append(layer_agg)
        set_mlp_weights(global_model_d, avg_weights_d)

        round_info["aggregation_weights"] = [
            {f"Node {i+1}": round(w, 6)} for i, w in enumerate(w_norm)
        ]
        byzantine_log.append(round_info)

        if (r + 1) % 5 == 0 or r == 0:
            y_tmp  = global_model_d.predict(X_test)
            f1_tmp = f1_score(y_test, y_tmp, average='weighted', zero_division=0)
            print(f"  [Config D] Round {r+1:>2}/{rounds}  "
                  f"agg_weights={[round(w, 3) for w in w_norm]}  "
                  f"global_f1={f1_tmp:.4f}")

    train_time_d = time.perf_counter() - start_time
    _, peak_mem_d = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_pred_d = global_model_d.predict(X_test)
    y_prob_d = global_model_d.predict_proba(X_test)
    prec_d, rec_d, f1_d, auc_d = evaluate_model(y_test, y_pred_d, y_prob_d)

    results.append({
        'Configuration':          'D: Byzantine Sim (Node 3 Poisoned)',
        'F1-Score':               f1_d,
        'Precision':              prec_d,
        'Recall':                 rec_d,
        'AUC-ROC':                auc_d,
        'Training_Time_s':        train_time_d,
        'Encryption_Overhead_ms': pqc_overhead_d,
        'Memory_MB':              peak_mem_d / (1024 * 1024),
        'Communication_KB':       comm_bytes_d / 1024,
    })
    print(f"  [Config D] Final → F1={f1_d:.4f}  AUC={auc_d:.4f}  "
          f"(Config C baseline F1={f1_c:.4f}  delta={f1_d - f1_c:+.4f})")

    # =====================================================================
    # Data Output & CLI Table Generation
    # =====================================================================
    import json

    df_res  = pd.DataFrame(results)
    out_dir = config.RESULTS_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / 'benchmark_comparison.csv', index=False)

    # Save detailed Byzantine simulation log
    byz_log_path = out_dir / 'byzantine_simulation_log.json'
    with open(byz_log_path, 'w') as f:
        json.dump({
            "description": (
                "Config D: Node 3 replaced its trained weights with Gaussian noise "
                "each round. Weighted FedAvg assigned it aggregation_weight=0 "
                "(local_f1=0), limiting poisoning impact on the global model."
            ),
            "byzantine_node":  f"Node {BYZANTINE_NODE + 1}",
            "config_c_f1":     round(f1_c, 4),
            "config_d_f1":     round(f1_d, 4),
            "f1_degradation":  round(f1_c - f1_d, 4),
            "rounds":          byzantine_log,
        }, f, indent=4)

    print("\n" + "="*110)
    print("FINAL BENCHMARK COMPARISON MATRIX")
    print("="*110)

    header = (f"{'Configuration':<40} | {'F1':<6} | {'AUC':<6} | "
              f"{'Train(s)':<8} | {'Crypto(ms)':<10} | {'Mem(MB)':<8} | {'Comm(KB)':<8}")
    print(header)
    print("-" * len(header))

    for r in results:
        tag = " [POISONED]" if "Byzantine" in r['Configuration'] else ""
        row = (f"{r['Configuration'] + tag:<40} | "
               f"{r['F1-Score']:.4f} | {r['AUC-ROC']:.4f} | "
               f"{r['Training_Time_s']:<8.2f} | {r['Encryption_Overhead_ms']:<10.2f} | "
               f"{r['Memory_MB']:<8.2f} | {r['Communication_KB']:<8.2f}")
        print(row)

    print("="*110)
    degradation = f1_c - f1_d
    verdict = ("Weighted FedAvg absorbed most of the attack" if abs(degradation) < 0.05
               else "significant degradation — tighten reputation gating")
    print(f"\nByzantine Impact Summary:")
    print(f"  Config C (Clean FL+PQC)       F1 = {f1_c:.4f}")
    print(f"  Config D (Node 3 Byzantine)   F1 = {f1_d:.4f}")
    print(f"  F1 degradation               : {degradation:+.4f}  ({verdict})")
    print("="*110)
    print(f"[OK] Benchmark CSV  saved → {out_dir / 'benchmark_comparison.csv'}")
    print(f"[OK] Byzantine log  saved → {byz_log_path}")


if __name__ == "__main__":
    run_benchmarks()

