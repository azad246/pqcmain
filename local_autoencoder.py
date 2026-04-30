import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Fix TensorFlow DLL issues on Windows by forcing CPU-only mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
except ImportError as e:
    print(f"[ERROR] TensorFlow import failed: {e}")
    print("Please reinstall: pip install --upgrade tensorflow")
    exit(1)

import config

# Configure TensorFlow to avoid allocating all GPU memory instantly
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def train_autoencoder_node(node_id):
    print(f"\n--- Training Local Autoencoder for Node {node_id} ---")
    
    # Define data paths
    nodes_dir = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
    train_path = nodes_dir / f'node{node_id}_train.csv'
    val_path = nodes_dir / f'node{node_id}_val.csv'
    
    if not train_path.exists() or not val_path.exists():
        print(f"[ERROR] Missing data for Node {node_id}.")
        return None
        
    print("Loading datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    
    # 1. Prepare Training Data (Benign Only)
    # Our encode_labels logic ensures benign == 0
    benign_train_df = train_df[train_df['label'] == 0]
    X_train_benign = benign_train_df.drop(columns=['label']).values
    
    print(f"Total training samples: {len(train_df)}. Benign samples used for AE: {len(benign_train_df)}")
    
    # 2. Build Autoencoder Model
    input_dim = X_train_benign.shape[1]
    
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    # Output layer uses sigmoid since data was MinMax scaled to [0, 1]
    decoded = Dense(input_dim, activation='sigmoid')(decoded) 
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 3. Train Autoencoder
    print(f"Training Autoencoder (input_dim={input_dim}) for 20 epochs...")
    autoencoder.fit(
        X_train_benign, X_train_benign,
        epochs=20,
        batch_size=256,
        shuffle=True,
        validation_split=0.1,
        verbose=0  # Set to 1 if you want to see the progress bar
    )
    
    # 4. Determine Threshold using Benign Validation Data
    val_benign_df = val_df[val_df['label'] == 0]
    X_val_benign = val_benign_df.drop(columns=['label']).values
    
    benign_preds = autoencoder.predict(X_val_benign, verbose=0)
    mse_benign = np.mean(np.power(X_val_benign - benign_preds, 2), axis=1)
    
    # Set anomaly threshold at 95th percentile of benign validation errors
    threshold = np.percentile(mse_benign, 95)
    print(f"Calculated Anomaly Threshold (95th percentile): {threshold:.6f}")
    
    # 5. Evaluate on Full Validation Set
    X_val = val_df.drop(columns=['label']).values
    
    # Convert multi-class labels to binary anomaly detection (0 = Benign, 1 = Attack)
    y_val_binary = (val_df['label'] > 0).astype(int).values
    
    print("Evaluating on full validation set...")
    val_preds = autoencoder.predict(X_val, verbose=0)
    mse_val = np.mean(np.power(X_val - val_preds, 2), axis=1)
    
    # Predictions: 1 (Anomaly) if MSE > threshold else 0 (Benign)
    y_pred = (mse_val > threshold).astype(int)
    
    # Calculate Metrics
    precision = precision_score(y_val_binary, y_pred, zero_division=0)
    recall = recall_score(y_val_binary, y_pred, zero_division=0)
    f1 = f1_score(y_val_binary, y_pred, zero_division=0)
    
    metrics = {
        'threshold': round(float(threshold), 6),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }
    
    print(f"-> Precision: {metrics['precision']:.4f}")
    print(f"-> Recall:    {metrics['recall']:.4f}")
    print(f"-> F1-Score:  {metrics['f1_score']:.4f}")
    
    # 6. Save Model
    model_save_path = config.MODEL_SAVE_PATH / f'ae_node{node_id}.keras'
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    autoencoder.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    return metrics

def run_local_ae_training():
    print("="*60)
    print("Starting Local Autoencoder Training (Anomaly Detection)")
    print("="*60)
    
    all_metrics = {}
    
    for node_id in range(1, config.NUM_NODES + 1):
        metrics = train_autoencoder_node(node_id)
        if metrics:
            all_metrics[f'node_{node_id}'] = metrics
            
    if not all_metrics:
        print("\n[ERROR] No models were trained successfully.")
        return
        
    # Print Comparison Table
    print("\n" + "="*60)
    print("Local Autoencoder Models Comparison (Binary Detection)")
    print("="*60)
    print(f"{'Node':<10} | {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    
    for node, m in all_metrics.items():
        node_name = node.replace('_', ' ').title()
        print(f"{node_name:<10} | {m['threshold']:<10.6f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
        
    print("="*60)
    
    # Save metrics to JSON
    results_dir = config.RESULTS_PATH
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / 'local_ae_metrics.json'
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"\n[OK] Aggregated metrics successfully saved to: {metrics_path}")

if __name__ == "__main__":
    run_local_ae_training()
