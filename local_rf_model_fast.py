import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import config
import warnings

warnings.filterwarnings('ignore')

def train_rf_node_direct(node_id):
    print(f"\n--- Training Local RF Model for Node {node_id} ---")
    
    # Load device CSV directly
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    device_file = processed_dir / f'device{node_id}.csv'
    
    if not device_file.exists():
        print(f"[ERROR] Device file not found: {device_file}")
        return None
    
    print(f"Loading data from {device_file.name}...")
    df = pd.read_csv(device_file, low_memory=False)
    print(f"Loaded {len(df)} rows")
    
    # Encode labels
    labels = df['label'].unique().tolist()
    labels = sorted([l for l in labels if l != 'benign'])
    labels = ['benign'] + labels
    label_map = {l: i for i, l in enumerate(labels)}
    df['label_encoded'] = df['label'].map(label_map)
    
    # Prepare features
    X = df.drop(columns=['label', 'label_encoded', 'device_id']).select_dtypes(include='number')
    y = df['label_encoded']
    
    X = X.fillna(0)
    
    # Train/Val split (70/30)
    print("Splitting into train/val (70/30)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=config.RANDOM_SEED
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train
    print(f"Training RF (n_estimators=50, max_depth=10)...")
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)
    
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    metrics = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4)
    }
    
    print(f"→ F1-Score: {metrics['f1_score']:.4f}")
    print(f"→ Precision: {metrics['precision']:.4f}")
    print(f"→ Recall: {metrics['recall']:.4f}")
    
    # Save
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_SAVE_PATH / f'rf_node{node_id}.pkl'
    joblib.dump(clf, model_path)
    print(f"Model saved: {model_path}")
    
    return metrics

def run_local_training():
    print("="*60)
    print("Starting Local Model Training (Fast Mode)")
    print("="*60)
    
    all_metrics = {}
    for node_id in range(1, config.NUM_NODES + 1):
        metrics = train_rf_node_direct(node_id)
        if metrics:
            all_metrics[f'node_{node_id}'] = metrics
    
    if not all_metrics:
        print("\n[ERROR] No models trained")
        return
    
    # Save results
    results_path = config.RESULTS_PATH / 'local_rf_metrics.json'
    config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print("\n" + "="*60)
    print("Local RF Training Complete")
    print("="*60)

if __name__ == "__main__":
    run_local_training()
