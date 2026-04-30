import pandas as pd
import config

def extract_label_from_filename(filename):
    """
    Extracts the label ('benign' or specific attack type) from the CSV filename.
    """
    name = filename.lower()
    if 'benign' in name:
        return 'benign'
    else:
        # Return the filename without the .csv extension as the attack label
        return filename.replace('.csv', '').replace('.CSV', '')

def process_device_data(device_id):
    device_dir = config.DATA_PATHS.get(f"node_{device_id}")
    
    if not device_dir or not device_dir.exists():
        print(f"[ERROR] Directory for Device {device_id} not found at {device_dir}")
        return
        
    csv_files = list(device_dir.rglob('*.csv'))
    if not csv_files:
        print(f"[WARNING] No CSV files found for Device {device_id} in {device_dir}")
        return
        
    df_list = []
    for file_path in csv_files:
        print(f"  - Loading {file_path.name}...")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Add required columns
            df['label'] = extract_label_from_filename(file_path.name)
            df['device_id'] = device_id
            
            df_list.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path.name}: {e}")
            
    if not df_list:
        return
        
    # Merge all DataFrames for the device
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Print requested statistics
    print(f"\n--- Statistics for Device {device_id} ---")
    print(f"Shape: {merged_df.shape} (Rows, Columns)")
    
    print(f"\nColumn Names ({len(merged_df.columns)} columns):")
    print(", ".join(merged_df.columns.tolist()))
    
    print("\nClass Distribution:")
    print(merged_df['label'].value_counts())
    
    missing_count = merged_df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {missing_count}")
    
    if missing_count > 0:
        print("Missing values per column:")
        missing_series = merged_df.isnull().sum()
        print(missing_series[missing_series > 0])
        
    # Ensure processed directory exists
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to processed directory
    save_path = processed_dir / f"device{device_id}.csv"
    print(f"\nSaving merged data to {save_path}...")
    merged_df.to_csv(save_path, index=False)
    print(f"Device {device_id} data successfully saved!\n")

if __name__ == "__main__":
    print("Starting Data Loading Process...")
    for dev_id in [1, 2, 3]:
        print("=" * 50)
        print(f"Processing Device {dev_id}")
        print("=" * 50)
        process_device_data(dev_id)
