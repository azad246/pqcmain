import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import config
import warnings

warnings.filterwarnings('ignore')

def train_rf_node(node_id):
    print(f"\n--- Training Local RF Model for Node {node_id} ---")
    
    # Define data paths
    nodes_dir = config.BASE_DIR / 'datasets' / 'processed' / 'nodes'
    train_path = nodes_dir / f'node{node_id}_train.csv'
    val_path = nodes_dir / f'node{node_id}_val.csv'
    
    if not train_path.exists() or not val_path.exists():
        print(f"[ERROR] Missing training or validation data for Node {node_id} at {train_path.parent}.")
        print("Please ensure preprocess.py has been run successfully.")
        return None
        
    print("Loading datasets...")
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    
    # Separate features and target
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    
    X_val = val_df.drop(columns=['label'])
    y_val = val_df['label']
    
    print("Training RandomForestClassifier(n_estimators=100, max_depth=15)...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=config.RANDOM_SEED, 
        n_jobs=-1 # Utilize all CPU cores for faster training
    )
    clf.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)
    
    # Calculate weighted metrics suitable for multi-class datasets
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    try:
        # AUC-ROC for multi-class
        auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"  [WARNING] Could not calculate AUC-ROC: {e}")
        auc = 0.0
        
    metrics = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4)
    }
    
    print(f"-> Precision: {metrics['precision']:.4f}")
    print(f"-> Recall:    {metrics['recall']:.4f}")
    print(f"-> F1-Score:  {metrics['f1_score']:.4f}")
    print(f"-> AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Save the trained model
    model_save_path = config.MODEL_SAVE_PATH / f'rf_node{node_id}.pkl'
    config.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    return metrics

def run_local_training():
    print("="*60)
    print("Starting Local Model Training for PQC-IoT Sentinel Nodes")
    print("="*60)
    
    all_metrics = {}
    
    for node_id in range(1, config.NUM_NODES + 1):
        metrics = train_rf_node(node_id)
        if metrics:
            all_metrics[f'node_{node_id}'] = metrics
            
    if not all_metrics:
        print("\n[ERROR] No models were trained successfully.")
        return
        
    # Print Comparison Table
    print("\n" + "="*60)
    print("Local Random Forest Models Comparison")
    print("="*60)
    print(f"{'Node':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
    print("-" * 60)
    
    for node, m in all_metrics.items():
        node_name = node.replace('_', ' ').title()
        print(f"{node_name:<10} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['auc_roc']:<10.4f}")
        
    print("="*60)
    
    # Save metrics to JSON
    results_dir = config.RESULTS_PATH
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / 'local_rf_metrics.json'
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"\n[OK] Aggregated metrics successfully saved to: {metrics_path}")

if __name__ == "__main__":
    run_local_training()
