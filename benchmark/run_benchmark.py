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
    print("Starting Comprehensive System Benchmark (Config A vs B vs C)")
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
        X_trains.append(df.drop(columns=['label']).values)
        y_trains.append(df['label'].values)
        
    # We use a combined subset of validation data as the global test proxy
    df_val = pd.read_csv(nodes_dir / 'node1_val.csv', low_memory=False) 
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
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
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
    clients = [MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True) for _ in range(3)]
    for c, X, y in zip(clients, X_trains, y_trains):
        c.partial_fit(X[:10], y[:10], classes=all_classes)
        
    global_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True)
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
    
    clients_c = [MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True) for _ in range(3)]
    for c, X, y in zip(clients_c, X_trains, y_trains):
        c.partial_fit(X[:10], y[:10], classes=all_classes)
        
    global_model_c = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True)
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
    # Data Output & CLI Table Generation
    # =====================================================================
    df_res = pd.DataFrame(results)
    
    out_dir = config.RESULTS_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / 'benchmark_comparison.csv', index=False)

    print("\n" + "="*105)
    print("FINAL BENCHMARK COMPARISON MATRIX")
    print("="*105)
    
    header = f"{'Configuration':<32} | {'F1':<6} | {'AUC':<6} | {'Train(s)':<8} | {'Crypto(ms)':<10} | {'Mem(MB)':<8} | {'Comm(KB)':<8}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        row = f"{r['Configuration']:<32} | {r['F1-Score']:.4f} | {r['AUC-ROC']:.4f} | {r['Training_Time_s']:<8.2f} | {r['Encryption_Overhead_ms']:<10.2f} | {r['Memory_MB']:<8.2f} | {r['Communication_KB']:<8.2f}"
        print(row)
        
    print("="*105)
    print(f"[OK] Full CSV matrix saved successfully to: {out_dir / 'benchmark_comparison.csv'}")

if __name__ == "__main__":
    run_benchmarks()
