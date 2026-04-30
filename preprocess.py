import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import config
import warnings

warnings.filterwarnings('ignore')

def encode_labels(df):
    """
    Custom encoder to ensure 'benign' is always mapped to 0, 
    and other attack types are mapped to subsequent integers.
    """
    unique_labels = sorted(df['label'].unique().tolist())
    
    # Ensure benign is at index 0
    if 'benign' in unique_labels:
        unique_labels.remove('benign')
        unique_labels = ['benign'] + unique_labels
        
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_encoded'] = df['label'].map(mapping)
    return df, mapping

def preprocess_node(device_id):
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    file_path = processed_dir / f'device{device_id}.csv'
    
    if not file_path.exists():
        print(f"[WARNING] Processed file not found: {file_path}. Skipping Node {device_id}.")
        return
        
    print(f"\n{'='*50}")
    print(f"Preprocessing Node {device_id} Data")
    print(f"{'='*50}")
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # 1. Encode Labels
    df, label_mapping = encode_labels(df)
    print(f"Label Mapping: {label_mapping}")
    
    nodes_dir = processed_dir / 'nodes'
    nodes_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the label mapping for inverse transform later in dashboard/results
    mapping_path = nodes_dir / f'node{device_id}_label_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=4)
        
    # 2. Separate Features and Target
    y = df['label_encoded']
    drop_cols = ['label', 'label_encoded', 'device_id']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).select_dtypes(include=[np.number])
    
    print(f"Initial numeric features: {X.shape[1]}")
    
    # Impute missing values with 0 to prevent errors during scaling/variance threshold
    X = X.fillna(0)
    
    # 3. Remove features with variance below 0.01 (optimized for large datasets)
    print("Computing feature variances...")
    variances = np.var(X.values, axis=0)
    var_mask = variances >= 0.01
    X_var_filtered = X.loc[:, var_mask]
    print(f"Features remaining after VarianceThreshold(0.01): {X_var_filtered.shape[1]}")
    
    # 4. Apply MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_var_filtered), columns=X_var_filtered.columns)
    
    # Re-attach target column for splitting
    final_df = X_scaled.copy()
    final_df['label'] = y
    
    # 5. Stratified Split (70-15-15)
    val_test_ratio = config.VAL_SIZE + config.TEST_SIZE
    test_ratio_of_temp = config.TEST_SIZE / val_test_ratio
    
    try:
        # Split 1: Train (70%) and Temp (30%)
        train_df, temp_df = train_test_split(
            final_df, 
            test_size=val_test_ratio, 
            stratify=final_df['label'], 
            random_state=config.RANDOM_SEED
        )
        
        # Split 2: Temp into Val (15%) and Test (15%)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=test_ratio_of_temp, 
            stratify=temp_df['label'], 
            random_state=config.RANDOM_SEED
        )
    except ValueError as e:
        print(f"[ERROR] Stratified split failed (possibly rare classes): {e}")
        print("Falling back to random split...")
        train_df, temp_df = train_test_split(final_df, test_size=val_test_ratio, random_state=config.RANDOM_SEED)
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio_of_temp, random_state=config.RANDOM_SEED)

    def print_split_stats(split_name, split_df):
        print(f"\n--- {split_name} Set ---")
        print(f"Shape: {split_df.shape}")
        print("Class Distribution:")
        print(split_df['label'].value_counts().sort_index())

    print_split_stats("Train", train_df)
    print_split_stats("Validation", val_df)
    print_split_stats("Test", test_df)
    
    # 6. Save Splits
    train_path = nodes_dir / f'node{device_id}_train.csv'
    val_path = nodes_dir / f'node{device_id}_val.csv'
    test_path = nodes_dir / f'node{device_id}_test.csv'
    
    print(f"\nSaving processed splits to {nodes_dir}...")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[OK] Node {device_id} preprocessing complete.")

def run_preprocessing():
    for device_id in range(1, config.NUM_NODES + 1):
        preprocess_node(device_id)

if __name__ == "__main__":
    print("Starting Preprocessing Pipeline...")
    run_preprocessing()
