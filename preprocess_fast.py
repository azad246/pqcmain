import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import config
import warnings

warnings.filterwarnings('ignore')

def encode_labels(df):
    """Encode labels with benign as 0."""
    unique_labels = sorted(df['label'].unique().tolist())
    if 'benign' in unique_labels:
        unique_labels.remove('benign')
        unique_labels = ['benign'] + unique_labels
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_encoded'] = df['label'].map(mapping)
    return df, mapping

def preprocess_node_fast(device_id):
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    file_path = processed_dir / f'device{device_id}.csv'
    
    if not file_path.exists():
        print(f"[WARNING] Processed file not found: {file_path}")
        return
        
    print(f"\n{'='*50}")
    print(f"Preprocessing Node {device_id}")
    print(f"{'='*50}")
    
    # Read CSV
    print("Loading data...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df)} rows")
    
    # Encode labels
    df, label_mapping = encode_labels(df)
    print(f"Label Mapping: {label_mapping}")
    
    # Create nodes directory
    nodes_dir = processed_dir / 'nodes'
    nodes_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label mapping
    mapping_path = nodes_dir / f'node{device_id}_label_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=4)
    
    # Prepare features
    y = df['label_encoded']
    drop_cols = ['label', 'label_encoded', 'device_id']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).select_dtypes(include=[np.number])
    
    print(f"Features: {X.shape[1]}")
    
    # Fill NaN
    X = X.fillna(0)
    
    # Create DataFrame with label
    final_df = X.copy()
    final_df['label'] = y
    
    # Stratified Split (70-15-15)
    print("Creating train/val/test splits...")
    val_test_ratio = config.VAL_SIZE + config.TEST_SIZE
    test_ratio_of_temp = config.TEST_SIZE / val_test_ratio
    
    try:
        train_df, temp_df = train_test_split(
            final_df, 
            test_size=val_test_ratio, 
            stratify=final_df['label'], 
            random_state=config.RANDOM_SEED
        )
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=test_ratio_of_temp, 
            stratify=temp_df['label'], 
            random_state=config.RANDOM_SEED
        )
    except:
        print("Stratified split failed, using random split")
        train_df, temp_df = train_test_split(final_df, test_size=val_test_ratio, random_state=config.RANDOM_SEED)
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio_of_temp, random_state=config.RANDOM_SEED)
    
    # Print stats
    print(f"\nTrain: {train_df.shape}")
    print(f"Val: {val_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Save splits
    print("Saving splits...")
    train_path = nodes_dir / f'node{device_id}_train.csv'
    val_path = nodes_dir / f'node{device_id}_val.csv'
    test_path = nodes_dir / f'node{device_id}_test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Node {device_id} complete!\n")

if __name__ == "__main__":
    print("Starting Fast Preprocessing Pipeline...")
    for dev_id in [1, 2, 3]:
        preprocess_node_fast(dev_id)
    print("All nodes preprocessed!")
