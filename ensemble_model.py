import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import config
import warnings

warnings.filterwarnings('ignore')

# Limit TensorFlow GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

def ensemble_predict(node_id, X_test):
    """
    Combines Random Forest and Autoencoder predictions using the requested logic.
    Returns binary predictions for fair comparison across all three approaches.
    """
    # 1. Load Models
    rf_path = config.MODEL_SAVE_PATH / f'rf_node{node_id}.pkl'
    ae_path = config.MODEL_SAVE_PATH / f'ae_node{node_id}.keras'
    
    if not rf_path.exists() or not ae_path.exists():
        raise FileNotFoundError(f"Missing models for Node {node_id}. Run local_rf_model.py and local_autoencoder.py first.")
        
    rf_model = joblib.load(rf_path)
    ae_model = load_model(ae_path)
    
    # 2. Load AE Threshold
    metrics_path = config.RESULTS_PATH / 'local_ae_metrics.json'
    with open(metrics_path, 'r') as f:
        ae_metrics = json.load(f)
    ae_threshold = ae_metrics[f'node_{node_id}']['threshold']
    
    # 3. Get RF Predictions
    rf_multi_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)
    # Convert multi-class RF prediction to binary (0=benign, 1=attack)
    rf_binary = (rf_multi_pred > 0).astype(int)
    # Probability of being an attack is 1.0 minus the probability of being class 0 (benign)
    rf_attack_prob = 1.0 - rf_prob[:, 0]
    
    # 4. Get AE Predictions
    ae_reconstructed = ae_model.predict(X_test, verbose=0)
    mse = np.mean(np.power(X_test - ae_reconstructed, 2), axis=1)
    ae_binary = (mse > ae_threshold).astype(int)
    
    # 5. Combine using Voting Logic
    ensemble_binary = np.zeros_like(rf_binary)
    for i in range(len(rf_binary)):
        if rf_binary[i] == 1 and ae_binary[i] == 1:
            # Both agree it is attack -> flag as attack
            ensemble_binary[i] = 1
        elif rf_binary[i] != ae_binary[i]:
            # Only one flags -> use RF decision
            ensemble_binary[i] = rf_binary[i]
        else:
            # Both agree it is benign -> flag as benign
            ensemble_binary[i] = 0
            
    # Note: Mathematically, this specific logic defaults to always trusting the RF model.
    
    return rf_binary, ae_binary, ensemble_binary, rf_attack_prob

def get_binary_metrics(y_true, y_pred, y_prob=None):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    else:
        # If no probability is provided (like raw AE predictions), calculate AUC from strict binary classes
        auc = roc_auc_score(y_true, y_pred)
        
    return {
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4)
    }

def run_ensemble_evaluation():
    print("="*80)
    print("Starting Ensemble Model Evaluation on Test Sets (Binary Anomaly Detection)")
    print("="*80)
    
    all_results = {}
    
    for node_id in range(1, config.NUM_NODES + 1):
        print(f"\n--- Evaluating Node {node_id} ---")
        test_path = config.BASE_DIR / 'datasets' / 'processed' / 'nodes' / f'node{node_id}_test.csv'
        
        if not test_path.exists():
            print(f"[ERROR] Test data for Node {node_id} not found.")
            continue
            
        df = pd.read_csv(test_path, low_memory=False)
        X_test = df.drop(columns=['label']).values
        y_test_binary = (df['label'] > 0).astype(int).values
        
        try:
            rf_binary, ae_binary, ensemble_binary, rf_attack_prob = ensemble_predict(node_id, X_test)
            
            # Since ensemble strictly trusts RF in this logic, its probabilities map to RF probabilities
            metrics_rf = get_binary_metrics(y_test_binary, rf_binary, rf_attack_prob)
            metrics_ae = get_binary_metrics(y_test_binary, ae_binary)
            metrics_ens = get_binary_metrics(y_test_binary, ensemble_binary, rf_attack_prob)
            
            all_results[f'node_{node_id}'] = {
                'RF': metrics_rf,
                'AE': metrics_ae,
                'Ensemble': metrics_ens
            }
            
            # Print localized comparison
            print(f"{'Model':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
            print("-" * 65)
            print(f"{'RF Alone':<12} | {metrics_rf['precision']:<10.4f} | {metrics_rf['recall']:<10.4f} | {metrics_rf['f1_score']:<10.4f} | {metrics_rf['auc_roc']:<10.4f}")
            print(f"{'AE Alone':<12} | {metrics_ae['precision']:<10.4f} | {metrics_ae['recall']:<10.4f} | {metrics_ae['f1_score']:<10.4f} | {metrics_ae['auc_roc']:<10.4f}")
            print(f"{'Ensemble':<12} | {metrics_ens['precision']:<10.4f} | {metrics_ens['recall']:<10.4f} | {metrics_ens['f1_score']:<10.4f} | {metrics_ens['auc_roc']:<10.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate Node {node_id}: {e}")

    # Save to JSON
    if all_results:
        results_dir = config.RESULTS_PATH
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / 'ensemble_metrics.json'
        
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        print(f"\n[OK] Final ensemble comparison metrics saved to: {out_path}")

if __name__ == "__main__":
    run_ensemble_evaluation()
