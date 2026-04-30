import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

# Force TensorFlow CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    print(f"[ERROR] TensorFlow import failed: {e}")
    exit(1)

import config

def train_autoencoder_direct(node_id):
    print(f"\n--- Training Local Autoencoder for Node {node_id} ---")
    
    # Load device CSV directly
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    device_file = processed_dir / f'device{node_id}.csv'
    
    if not device_file.exists():
        print(f"[ERROR] Device file not found: {device_file}")
        return None
    
    print(f"Loading data from {device_file.name}...")
    df = pd.read_csv(device_file, low_memory=False)
    print(f"Loaded {len(df)} rows")
    
    # Prepare features (no label needed for autoencoder)
    X = df.drop(columns=['label', 'device_id']).select_dtypes(include='number')
    X = X.fillna(0)
    
    # Normalize
    print("Normalizing...")
    X_min, X_max = X.min(), X.max()
    X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
    
    # Simple autoencoder
    input_dim = X_normalized.shape[1]
    encoding_dim = max(10, input_dim // 2)
    
    print(f"Building autoencoder: {input_dim} -> {encoding_dim} -> {input_dim}")
    
    # Encoder
    encoder_input = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(encoder_input)
    
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder
    ae = Model(encoder_input, decoded)
    ae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Train
    print("Training autoencoder...")
    ae.fit(X_normalized, X_normalized, epochs=10, batch_size=256, verbose=0)
    
    # Save
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_SAVE_PATH / f'ae_node{node_id}.h5'
    ae.save(model_path)
    print(f"Autoencoder saved: {model_path}")
    
    return {"status": "trained"}

def run_autoencoder_training():
    print("="*60)
    print("Starting Autoencoder Training (Fast Mode)")
    print("="*60)
    
    for node_id in range(1, config.NUM_NODES + 1):
        result = train_autoencoder_direct(node_id)
        if not result:
            print(f"[WARNING] Failed to train autoencoder for Node {node_id}")
    
    print("\n" + "="*60)
    print("Autoencoder Training Complete")
    print("="*60)

if __name__ == "__main__":
    run_autoencoder_training()
