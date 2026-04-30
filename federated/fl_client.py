import argparse
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import flwr as fl
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from crypto import pqc_crypto
from crypto.encrypt_weights import encrypt_weights, decrypt_weights

def get_model_parameters(model):
    if hasattr(model, 'coefs_'):
        return model.coefs_ + model.intercepts_
    return []

def set_model_parameters(model, parameters):
    if not parameters: return
    n_layers = len(parameters) // 2
    model.coefs_ = parameters[:n_layers]
    model.intercepts_ = parameters[n_layers:]

class PQC_FLClient(fl.client.NumPyClient):
    def __init__(self, node_id):
        self.node_id = node_id
        self.keys_dir = config.BASE_DIR / 'keys'
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 1. PQC KEY GENERATION ---
        print(f">>> [Node {self.node_id}] Generating Client PQC Keypair...")
        self.client_pub, self.client_sec = pqc_crypto.generate_keypair()
        
        with open(self.keys_dir / f'client_{self.node_id}_pub.pem', 'wb') as f:
            f.write(self.client_pub)
            
        self.load_data()
        self.init_model()

    def load_data(self):
        print(f"[Node {self.node_id}] Loading local training and validation data...")
        nodes_dir = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
        train_path = nodes_dir / f'node{self.node_id}_train.csv'
        val_path = nodes_dir / f'node{self.node_id}_val.csv'
        
        train_df = pd.read_csv(train_path, low_memory=False)
        val_df = pd.read_csv(val_path, low_memory=False)
        
        self.X_train = train_df.drop(columns=['label']).values
        self.y_train = train_df['label'].values
        self.X_val = val_df.drop(columns=['label']).values
        self.y_val = val_df['label'].values
        
        mapping_path = nodes_dir / f'node{self.node_id}_label_mapping.json'
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            
        self.total_classes = np.arange(len(mapping))

    def init_model(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True, random_state=config.RANDOM_SEED
        )
        self.model.partial_fit(self.X_train[:10], self.y_train[:10], classes=self.total_classes)

    def get_parameters(self, config_dict):
        # Allow initial parameter probing by the server in raw format
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        if not parameters: return
        
        # --- 2. PQC DECRYPTION INTERCEPT ---
        # Identify if this is a secure PQC bundle from the Server
        is_encrypted_bundle = (
            len(parameters) == config.NUM_NODES and 
            parameters[0].dtype == np.uint8 and 
            parameters[0].ndim == 1
        )
        
        if is_encrypted_bundle:
            print(f"[Node {self.node_id}] Decrypting PQC global weights from Server...")
            
            # Extract the specific encrypted payload meant exclusively for this node
            my_encrypted_bytes = parameters[self.node_id - 1].tobytes()
            
            decrypted_weights = decrypt_weights(my_encrypted_bytes, self.client_sec)
            set_model_parameters(self.model, decrypted_weights)
        else:
            set_model_parameters(self.model, parameters)

    def fit(self, parameters, fit_config):
        self.set_parameters(parameters)
        self.model.partial_fit(self.X_train, self.y_train)
        
        raw_weights = get_model_parameters(self.model)
        
        # --- 3. PQC ENCRYPTION INTERCEPT ---
        server_pub_path = self.keys_dir / 'server_pub.pem'
        while not server_pub_path.exists():
            time.sleep(1)
            
        with open(server_pub_path, 'rb') as f:
            server_pub = f.read()
            
        print(f"[Node {self.node_id}] Encrypting local updates for Server using PQC...")
        encrypted_bytes = encrypt_weights(raw_weights, server_pub)
        
        # Wrap the byte stream natively into a NumPy wrapper so Flower can transport it over gRPC
        encrypted_parameters = [np.frombuffer(encrypted_bytes, dtype=np.uint8)]
        
        return encrypted_parameters, len(self.X_train), {}

    def evaluate(self, parameters, eval_config):
        self.set_parameters(parameters)
        
        y_pred = self.model.predict(self.X_val)
        y_prob = self.model.predict_proba(self.X_val)
        
        try:
            loss = log_loss(self.y_val, y_prob, labels=self.total_classes)
        except Exception:
            loss = 1.0
            
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted', zero_division=0)
        
        metrics = {"accuracy": float(accuracy), "f1": float(f1)}
        return float(loss), len(self.X_val), metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PQC Flower FL Client")
    parser.add_argument("--node_id", type=int, required=True, help="Node ID (e.g., 1, 2, 3)")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="FL Server Address")
    args = parser.parse_args()
    
    print("="*60)
    print(f"Starting Secure PQC Flower Client for Node {args.node_id}")
    print("="*60)
    
    client = PQC_FLClient(node_id=args.node_id)
    
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
