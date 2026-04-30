import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

# Setup system path so the script can import local modules
sys.path.append(str(Path(__file__).resolve().parents[1]))
from crypto.pqc_crypto import PQCManager
import config

# Set paper-quality context for seaborn/matplotlib
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def run_benchmarks():
    print("="*65)
    print("Running Cryptography Benchmarks: PQC vs Classical (RSA)")
    print("="*65)
    
    results_dir = config.RESULTS_PATH
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Crypto Managers
    # Native liboqs PQC instance
    manager_pqc = PQCManager('Kyber512')
    
    # Forced Fallback RSA instance
    manager_rsa = PQCManager('Kyber512')
    manager_rsa.use_pqc = False
    manager_rsa.algorithm = "RSA-2048"
    
    # Generate exactly 10 KB of random dummy data (Simulating FL weights)
    payload_size = 10 * 1024 
    dummy_payload = os.urandom(payload_size)
    print(f"\nPayload Size: {payload_size} bytes (10 KB)")
    
    num_runs = 10
    print(f"Iterations per algorithm: {num_runs}\n")
    
    results = {
        "PQC (Kyber512)": {},
        "Classical (RSA-2048)": {}
    }
    
    managers = {
        "PQC (Kyber512)": manager_pqc,
        "Classical (RSA-2048)": manager_rsa
    }
    
    # 2. Execute Benchmark Iterations
    for name, mgr in managers.items():
        print(f"Benchmarking {name}...")
        
        keygen_times = []
        enc_times = []
        dec_times = []
        
        # Track sizes strictly once (since they are constant per algorithm)
        pub_keys = []
        ciphertexts = []
        
        for i in range(num_runs):
            # Measure Key Generation
            t0 = time.perf_counter()
            pub, sec = mgr.generate_keypair()
            t1 = time.perf_counter()
            keygen_times.append((t1 - t0) * 1000) # Convert to ms
            
            # Measure Encryption
            t0 = time.perf_counter()
            ct = mgr.encrypt_data(dummy_payload, pub)
            t1 = time.perf_counter()
            enc_times.append((t1 - t0) * 1000) # Convert to ms
            
            # Measure Decryption
            t0 = time.perf_counter()
            dec = mgr.decrypt_data(ct, sec)
            t1 = time.perf_counter()
            dec_times.append((t1 - t0) * 1000) # Convert to ms
            
            if i == 0:
                pub_keys.append(len(pub))
                ciphertexts.append(len(ct))
                
        # Store aggregations
        results[name] = {
            "keygen_time_ms_mean": float(np.mean(keygen_times)),
            "keygen_time_ms_std": float(np.std(keygen_times)),
            "encrypt_time_ms_mean": float(np.mean(enc_times)),
            "encrypt_time_ms_std": float(np.std(enc_times)),
            "decrypt_time_ms_mean": float(np.mean(dec_times)),
            "decrypt_time_ms_std": float(np.std(dec_times)),
            "public_key_size_bytes": pub_keys[0],
            "ciphertext_size_bytes": ciphertexts[0]
        }
        
        print(f"  Keygen (ms):  Mean = {results[name]['keygen_time_ms_mean']:.3f}, Std = {results[name]['keygen_time_ms_std']:.3f}")
        print(f"  Encrypt (ms): Mean = {results[name]['encrypt_time_ms_mean']:.3f}, Std = {results[name]['encrypt_time_ms_std']:.3f}")
        print(f"  Decrypt (ms): Mean = {results[name]['decrypt_time_ms_mean']:.3f}, Std = {results[name]['decrypt_time_ms_std']:.3f}")
        print(f"  Pub Key Size: {results[name]['public_key_size_bytes']} bytes")
        print(f"  Cipher Size:  {results[name]['ciphertext_size_bytes']} bytes\n")

    # 3. Save JSON Metrics
    json_path = results_dir / 'crypto_benchmark.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[OK] Raw metrics JSON saved to -> {json_path}")
    
    # 4. Generate Paper-Quality Subplot
    plot_path = results_dir / 'crypto_comparison.png'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ----- Plot 1: Operational Timings -----
    labels = ['Key Generation', 'Encryption', 'Decryption']
    
    pqc_times = [
        results["PQC (Kyber512)"]['keygen_time_ms_mean'],
        results["PQC (Kyber512)"]['encrypt_time_ms_mean'],
        results["PQC (Kyber512)"]['decrypt_time_ms_mean']
    ]
    pqc_errs = [
        results["PQC (Kyber512)"]['keygen_time_ms_std'],
        results["PQC (Kyber512)"]['encrypt_time_ms_std'],
        results["PQC (Kyber512)"]['decrypt_time_ms_std']
    ]
    
    rsa_times = [
        results["Classical (RSA-2048)"]['keygen_time_ms_mean'],
        results["Classical (RSA-2048)"]['encrypt_time_ms_mean'],
        results["Classical (RSA-2048)"]['decrypt_time_ms_mean']
    ]
    rsa_errs = [
        results["Classical (RSA-2048)"]['keygen_time_ms_std'],
        results["Classical (RSA-2048)"]['encrypt_time_ms_std'],
        results["Classical (RSA-2048)"]['decrypt_time_ms_std']
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    # We use upward-only error bars to prevent standard deviation lines from crashing the logarithmic scale if they cross zero
    pqc_errs_up = np.vstack([np.zeros(len(pqc_errs)), pqc_errs])
    rsa_errs_up = np.vstack([np.zeros(len(rsa_errs)), rsa_errs])
    
    axes[0].bar(x - width/2, pqc_times, width, yerr=pqc_errs_up, label='PQC (Kyber512)', capsize=5, color='#2c3e50')
    axes[0].bar(x + width/2, rsa_times, width, yerr=rsa_errs_up, label='Classical (RSA-2048)', capsize=5, color='#e74c3c')
    axes[0].set_ylabel('Time (ms) - Log Scale', fontweight='bold')
    axes[0].set_title('Cryptographic Operation Timings', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].legend()
    
    # ----- Plot 2: Size & Communication Overhead -----
    size_labels = ['Public Key Size', 'Total Payload Size']
    pqc_sizes = [
        results["PQC (Kyber512)"]['public_key_size_bytes'],
        results["PQC (Kyber512)"]['ciphertext_size_bytes']
    ]
    rsa_sizes = [
        results["Classical (RSA-2048)"]['public_key_size_bytes'],
        results["Classical (RSA-2048)"]['ciphertext_size_bytes']
    ]
    
    x2 = np.arange(len(size_labels))
    axes[1].bar(x2 - width/2, pqc_sizes, width, label='PQC (Kyber512)', color='#2c3e50')
    axes[1].bar(x2 + width/2, rsa_sizes, width, label='Classical (RSA-2048)', color='#e74c3c')
    axes[1].set_ylabel('Size (Bytes)', fontweight='bold')
    axes[1].set_title('Communication Overhead (10KB Payload)', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(size_labels, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[OK] High-resolution benchmark comparison plot saved to -> {plot_path}")

if __name__ == "__main__":
    run_benchmarks()
