import pickle
import time
import sys
import numpy as np
from pathlib import Path

# Setup system path so the script can import local modules when run directly
sys.path.append(str(Path(__file__).resolve().parents[1]))
from crypto import pqc_crypto

def weights_to_bytes(model_weights_list):
    """
    Serializes a list of numpy arrays representing model weights into raw bytes.
    Using pickle is the standard and fastest method for Python arrays.
    """
    return pickle.dumps(model_weights_list)

def bytes_to_weights(data_bytes):
    """
    Deserializes raw bytes back into a list of numpy arrays.
    """
    return pickle.loads(data_bytes)

def encrypt_weights(weights, public_key):
    """
    Converts raw model weights into bytes and hybrid-encrypts them using Post-Quantum Crypto.
    """
    data_bytes = weights_to_bytes(weights)
    encrypted_bytes = pqc_crypto.encrypt_data(data_bytes, public_key)
    return encrypted_bytes

def decrypt_weights(encrypted_bytes, secret_key):
    """
    Decrypts the hybrid payload and deserializes the bytes back into the original numpy weights.
    """
    decrypted_bytes = pqc_crypto.decrypt_data(encrypted_bytes, secret_key)
    weights = bytes_to_weights(decrypted_bytes)
    return weights

def run_test():
    print("="*65)
    print("PQC-IoT Sentinel: Federated Weights Encryption Self-Test")
    print("="*65)
    
    # 1. Generate PQC Keypair
    print("Generating PQC Keypair...")
    pub_key, sec_key = pqc_crypto.generate_keypair()
    
    # 2. Create sample numpy arrays representing local model weights
    # Simulating a multi-layer neural network with random uniform weights
    print("\nGenerating simulated model weights...")
    sample_weights = [
        np.random.rand(115, 64).astype(np.float32),  # Hidden Layer 1 Weights
        np.random.rand(64).astype(np.float32),       # Hidden Layer 1 Biases
        np.random.rand(64, 32).astype(np.float32),   # Hidden Layer 2 Weights
        np.random.rand(32).astype(np.float32)        # Hidden Layer 2 Biases
    ]
    
    # Serialize to measure true original size
    original_bytes = weights_to_bytes(sample_weights)
    original_size = len(original_bytes)
    print(f"-> Original Serialized Size: {original_size} bytes")
    
    # 3. Test Encryption Speed
    print("\nStarting Encryption Pipeline...")
    start_enc = time.time()
    encrypted_payload = encrypt_weights(sample_weights, pub_key)
    end_enc = time.time()
    
    enc_time = end_enc - start_enc
    encrypted_size = len(encrypted_payload)
    
    # 4. Test Decryption Speed
    print("Starting Decryption Pipeline...")
    start_dec = time.time()
    decrypted_weights = decrypt_weights(encrypted_payload, sec_key)
    end_dec = time.time()
    
    dec_time = end_dec - start_dec
    
    # 5. Verify Mathematical Match
    print("\nVerifying Decrypted Weights Integrity...")
    all_match = True
    for i, (orig, dec) in enumerate(zip(sample_weights, decrypted_weights)):
        if not np.array_equal(orig, dec):
            print(f"[ERROR] Array {i} mismatch detected!")
            all_match = False
            break
            
    if all_match:
        print("[OK] Decrypted weights mathematically match the original weights perfectly!")
    else:
        print("[ERROR] Verification Failed.")
        return
        
    # 6. Output Final Overhead Metrics
    print("\n" + "-"*65)
    print("Performance & Overhead Metrics")
    print("-"*65)
    print(f"Encryption Time: {enc_time:.6f} seconds")
    print(f"Decryption Time: {dec_time:.6f} seconds")
    print(f"Encrypted Size:  {encrypted_size} bytes")
    
    size_ratio = encrypted_size / original_size
    print(f"Size Ratio (Encrypted / Original): {size_ratio:.4f}x")
    
    overhead_bytes = encrypted_size - original_size
    print(f"Network Overhead Added: {overhead_bytes} bytes (KEM Payload + AES Auth Tag)")
    print("="*65 + "\n")

if __name__ == "__main__":
    run_test()
