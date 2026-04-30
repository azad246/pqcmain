import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import config
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def prepare_unsw():
    print("="*60)
    print("Preparing UNSW-NB15 Dataset for Cross-Dataset Validation")
    print("="*60)

    unsw_path = config.DATA_PATHS['test'] / 'testing_set.csv'
    
    if not unsw_path.exists():
        print(f"[ERROR] UNSW dataset not found at {unsw_path}")
        print("Please ensure you have placed 'testing_set.csv' in datasets/test/")
        return

    print(f"Loading {unsw_path}...")
    df = pd.read_csv(unsw_path, low_memory=False)
    
    # 1. Print all column names and dtypes
    print("\n--- UNSW-NB15 Columns and DataTypes ---")
    with pd.option_context('display.max_rows', None):
        print(df.dtypes)
    
    # 2. Identify Label Column and Encode Binary
    # In UNSW-NB15, 'label' is typically binary (0 for normal, 1 for attack).
    # 'attack_cat' contains the specific attack strings.
    target_col = 'label'
    if target_col not in df.columns and 'Label' in df.columns:
        df.rename(columns={'Label': 'label'}, inplace=True)
        
    if 'label' not in df.columns:
        print(f"\n[ERROR] Could not find 'label' column. Found columns: {df.columns.tolist()}")
        return

    print("\n--- Encoding Labels ---")
    df['label'] = df['label'].astype(int) # Ensure binary
    
    print("Label distribution:")
    print("0 = Normal, 1 = Attack")
    print(df['label'].value_counts())
    
    # Drop non-numeric and target columns before scaling
    # UNSW specific categorical columns
    cols_to_drop = ['id', 'attack_cat', 'proto', 'service', 'state', 'label']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Filter strictly to numeric
    X = X.select_dtypes(include=[np.number]).fillna(0)
    
    # 3. Apply MinMaxScaler
    print("\n--- Applying MinMaxScaler ---")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 4. Align Feature Columns to Match N-BaIoT
    print("\n--- Aligning Features with N-BaIoT ---")
    
    # To accurately match, we read the columns from the processed N-BaIoT training set
    n_baiot_train = config.BASE_DIR / 'datasets' / 'processed' / 'nodes' / 'node1_train.csv'
    
    if n_baiot_train.exists():
        print(f"Reading target schema from {n_baiot_train.name}...")
        n_df = pd.read_csv(n_baiot_train, nrows=0) # Read only header
        n_features = [col for col in n_df.columns if col != 'label']
    else:
        print("[WARNING] Processed N-BaIoT node1_train.csv not found!")
        print("Please run preprocess.py first to establish the baseline N-BaIoT features.")
        print("Falling back to generic 115 zero-padded columns...")
        n_features = [f'n_baiot_feature_{i}' for i in range(115)]

    # Align Data
    X_aligned = pd.DataFrame(index=X_scaled.index)
    
    mapped_count = 0
    padded_count = 0
    
    for feat in n_features:
        # If exact column name matches, map it. Otherwise, pad with zero.
        if feat in X_scaled.columns:
            X_aligned[feat] = X_scaled[feat]
            mapped_count += 1
        else:
            X_aligned[feat] = 0.0
            padded_count += 1
            
    print(f"Original UNSW numeric features: {X_scaled.shape[1]}")
    print(f"Target N-BaIoT features: {len(n_features)}")
    print(f"Successfully mapped by exact name: {mapped_count} features")
    print(f"Zero-padded missing features: {padded_count} features")

    # Add binary label back
    X_aligned['label'] = df['label'].values
    
    # 5. Save the aligned dataset
    out_path = config.BASE_DIR / 'datasets' / 'processed' / 'unsw_test_ready.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving final cross-dataset validation file to:")
    print(f"-> {out_path}")
    X_aligned.to_csv(out_path, index=False)
    print("\n[OK] UNSW-NB15 preparation complete!")
    print("NOTE: This dataset should be used ONLY for cross-dataset validation.")

if __name__ == "__main__":
    prepare_unsw()
