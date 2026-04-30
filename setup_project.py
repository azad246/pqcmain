import os
import pandas as pd
from pathlib import Path

def setup_project():
    base_path = Path(__file__).parent.resolve()
    print(f"Setting up project in: {base_path}")
    
    # 1. Create missing folders and 2. Create __init__.py
    folders = ['models', 'results', 'logs', 'dashboard', 'crypto', 'federated']
    for f in folders:
        folder_path = base_path / f
        folder_path.mkdir(parents=True, exist_ok=True)
        init_file = folder_path / '__init__.py'
        init_file.touch(exist_ok=True)
        print(f"Created/Verified directory: {f}/ with __init__.py")

def print_tree(directory, prefix=''):
    path = Path(directory)
    try:
        items = sorted([p for p in path.iterdir() if p.name not in ('.git', '__pycache__')])
    except PermissionError:
        return
        
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = '\\--- ' if is_last else '+--- '
        print(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")
        if item.is_dir():
            extension = '    ' if is_last else '|   '
            print_tree(item, prefix + extension)

def verify_datasets():
    base_path = Path(__file__).parent.resolve()
    print("\nVerifying datasets...")
    
    # Check Train datasets
    train_dir = base_path / 'datasets' / 'train'
    train_devices = ['1', '2', '3']
    for dev in train_devices:
        dev_path = train_dir / dev
        if dev_path.exists() and dev_path.is_dir():
            csv_files = list(dev_path.glob('*.csv'))
            if not csv_files:
                 print(f"[WARNING] No CSV files found in datasets/train/{dev}")
            for csv_file in csv_files:
                 try:
                     df = pd.read_csv(csv_file, low_memory=False)
                     print(f"[OK] datasets/train/{dev}/{csv_file.name} - Rows: {len(df)}")
                 except Exception as e:
                     print(f"[ERROR] Failed to read {csv_file.name}: {e}")
        else:
            print(f"[MISSING] Directory not found: datasets/train/{dev}/")

    # Check Test datasets
    test_dir = base_path / 'datasets' / 'test'
    test_files = ['testing_set.csv', 'training_set.csv']
    for t_file in test_files:
        t_path = test_dir / t_file
        if t_path.exists():
             try:
                 df = pd.read_csv(t_path, low_memory=False)
                 print(f"[OK] datasets/test/{t_file} - Rows: {len(df)}")
             except Exception as e:
                 print(f"[ERROR] Failed to read {t_file}: {e}")
        else:
             print(f"[MISSING] File not found: datasets/test/{t_file}")

if __name__ == '__main__':
    setup_project()
    print("\nProject Structure:")
    print(f"{Path(__file__).parent.name}/")
    print_tree(Path(__file__).parent)
    verify_datasets()
