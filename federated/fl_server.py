import flwr as fl
import csv
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from crypto import pqc_crypto
from crypto.encrypt_weights import encrypt_weights, decrypt_weights

class SecureMetricsStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_file = config.RESULTS_PATH / 'fl_round_metrics.csv'
        config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        
        self.keys_dir = config.BASE_DIR / 'keys'
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 1. SERVER PQC KEY GENERATION ---
        print(">>> [Server] Generating Server PQC Keypair...")
        self.server_pub, self.server_sec = pqc_crypto.generate_keypair()
        
        with open(self.keys_dir / 'server_pub.pem', 'wb') as f:
            f.write(self.server_pub)
            
        self.last_round_overhead = 0.0
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Global_Loss', 'Global_Accuracy', 'Global_F1_Score', 'Encryption_Overhead_ms'])

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return super().aggregate_fit(server_round, results, failures)
            
        print(f"\n--- Round {server_round} Secure Aggregation ---")
        
        # --- 2. SERVER DECRYPTION PHASE ---
        start_dec = time.perf_counter()
        decrypted_results = []
        
        for client_proxy, fit_res in results:
            # Extract the raw byte stream from the NumPy wrapper
            ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            encrypted_bytes = ndarrays[0].tobytes()
            
            # Decrypt using Server's Secret Key
            decrypted_weights = decrypt_weights(encrypted_bytes, self.server_sec)
            decrypted_parameters = fl.common.ndarrays_to_parameters(decrypted_weights)
            
            # Reconstruct Flower FitRes object natively
            decrypted_fit_res = fl.common.FitRes(
                status=fit_res.status,
                parameters=decrypted_parameters,
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics
            )
            decrypted_results.append((client_proxy, decrypted_fit_res))
            
        dec_time = (time.perf_counter() - start_dec) * 1000
        print(f">>> [Server] Decrypted {len(results)} client updates in {dec_time:.2f} ms")
        
        # Native FedAvg Math Execution
        aggregated_parameters, metrics = super().aggregate_fit(server_round, decrypted_results, failures)
        
        if aggregated_parameters is None:
            return None, {}
            
        global_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        # --- 3. SERVER ENCRYPTION PHASE (BROADCAST) ---
        start_enc = time.perf_counter()
        bundled_encrypted_weights = []
        
        # Loop through each node's public key and encrypt the global weights uniquely for them
        for node_id in range(1, config.NUM_NODES + 1):
            client_pub_path = self.keys_dir / f'client_{node_id}_pub.pem'
            
            while not client_pub_path.exists():
                time.sleep(1)
                
            with open(client_pub_path, 'rb') as f:
                client_pub = f.read()
                
            encrypted_bytes = encrypt_weights(global_weights, client_pub)
            bundled_encrypted_weights.append(np.frombuffer(encrypted_bytes, dtype=np.uint8))
            
        enc_time = (time.perf_counter() - start_enc) * 1000
        print(f">>> [Server] Encrypted global model sequentially for {config.NUM_NODES} clients in {enc_time:.2f} ms")
        
        # Cache overhead for logging during evaluation
        self.last_round_overhead = dec_time + enc_time
        
        # Bundle the encrypted arrays safely back into Flower Parameters
        secure_parameters = fl.common.ndarrays_to_parameters(bundled_encrypted_weights)
        return secure_parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_loss is not None and aggregated_metrics is not None:
            accuracy = aggregated_metrics.get("accuracy", 0.0)
            f1 = aggregated_metrics.get("f1", 0.0)
            overhead = getattr(self, 'last_round_overhead', 0.0)
            
            print(f"\n{'*'*45}")
            print(f"--- Round {server_round} Complete ---")
            print(f"Global Loss:         {aggregated_loss:.4f}")
            print(f"Global Accuracy:     {accuracy:.4f}")
            print(f"Global F1-Score:     {f1:.4f}")
            print(f"Encryption Overhead: {overhead:.2f} ms")
            print(f"{'*'*45}\n")
            
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([server_round, aggregated_loss, accuracy, f1, overhead])
                
        return aggregated_loss, aggregated_metrics

def evaluate_metrics_aggregation_fn(metrics):
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {"accuracy": 0.0, "f1": 0.0}
        
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]
    
    return {
        "accuracy": sum(accuracies) / total_examples,
        "f1": sum(f1s) / total_examples,
    }

def start_server():
    print("="*65)
    print("Starting Secure PQC Flower FL Server")
    print(f"Strategy: SecureMetricsStrategy (FedAvg + Post-Quantum KEM)")
    print(f"Expected Clients: {config.NUM_NODES}")
    print("="*65)

    strategy = SecureMetricsStrategy(
        fraction_fit=1.0, 
        fraction_evaluate=1.0, 
        min_fit_clients=config.NUM_NODES,
        min_evaluate_clients=config.NUM_NODES,
        min_available_clients=config.NUM_NODES,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config.FL_ROUNDS),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server()
