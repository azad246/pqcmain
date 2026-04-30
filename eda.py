import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import config
from sklearn.feature_selection import f_classif
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set aesthetic style for research paper plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def get_numeric_features(df):
    """Filter out non-numeric columns like label and device_id."""
    return df.select_dtypes(include=[np.number]).drop(columns=['device_id'], errors='ignore')

def plot_class_distribution(df, device_id, save_dir):
    plt.figure(figsize=(10, 6))
    
    # Count classes and sort
    class_counts = df['label'].value_counts()
    
    ax = sns.barplot(x=class_counts.values, y=class_counts.index, palette='viridis')
    plt.title(f'Class Distribution for Device {device_id}', fontsize=16, fontweight='bold')
    plt.xlabel('Count', fontsize=12, fontweight='bold')
    plt.ylabel('Class Label', fontsize=12, fontweight='bold')
    
    # Add exact count annotations to the bars
    for i, v in enumerate(class_counts.values):
        ax.text(v + (v * 0.01), i, str(v), color='black', va='center', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(save_dir / f'class_distribution_device_{device_id}.png')
    plt.close()

def plot_correlation_heatmap(df, device_id, save_dir):
    numeric_df = get_numeric_features(df)
    
    # Find top 20 features with highest variance to prevent massive, unreadable heatmaps
    variances = numeric_df.var()
    top_20_features = variances.nlargest(20).index
    
    corr_matrix = numeric_df[top_20_features].corr()
    
    plt.figure(figsize=(14, 12))
    # Use a diverging colormap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .75})
    plt.title(f'Correlation Heatmap of Top 20 Features (by variance) - Device {device_id}', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'correlation_heatmap_device_{device_id}.png')
    plt.close()

def plot_boxplots(df, device_id, save_dir):
    numeric_df = get_numeric_features(df)
    
    # Impute missing values with mean for ANOVA F-value calculation
    X = numeric_df.fillna(numeric_df.mean())
    y = df['label']
    
    # Calculate ANOVA F-value to find most discriminative features
    f_values, _ = f_classif(X, y)
    f_series = pd.Series(f_values, index=X.columns)
    
    # Get top 5 most discriminative features
    top_5_features = f_series.nlargest(5).index
    
    for i, feature in enumerate(top_5_features):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='label', y=feature, palette='Set2')
        plt.title(f'Boxplot of Most Discriminative Feature #{i+1}: {feature} - Device {device_id}', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=12, fontweight='bold')
        plt.ylabel(feature, fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Replace special characters in feature name for safe file saving
        safe_feature_name = feature.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(save_dir / f'boxplot_top{i+1}_{safe_feature_name}_device_{device_id}.png')
        plt.close()

def print_basic_stats(df, device_id, save_dir):
    print(f"\n{'='*50}")
    print(f"Basic Statistics for Device {device_id}")
    print(f"{'='*50}")
    
    numeric_df = get_numeric_features(df)
    
    # Calculate stats for all numeric features
    stats_df = df.groupby('label')[numeric_df.columns].agg(['mean', 'std', 'min', 'max'])
    
    # Save the full statistics to CSV for the paper
    stats_csv_path = save_dir / f'basic_stats_device_{device_id}.csv'
    stats_df.to_csv(stats_csv_path)
    print(f"[OK] Full descriptive statistics saved to: {stats_csv_path}")
    
    # Calculate ANOVA F-value to get top 5 features for console preview
    X = numeric_df.fillna(numeric_df.mean())
    y = df['label']
    f_values, _ = f_classif(X, y)
    top_5_features = pd.Series(f_values, index=X.columns).nlargest(5).index
    
    print(f"\nPreview: Statistics for top 5 most discriminative features by class:\n")
    top_5_stats = df.groupby('label')[top_5_features].agg(['mean', 'std', 'min', 'max'])
    print(top_5_stats)

def run_eda():
    processed_dir = config.BASE_DIR / 'datasets' / 'processed'
    
    # Store EDA plots in results/eda_plots as requested
    eda_plots_dir = config.RESULTS_PATH / 'eda_plots'
    eda_plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Plots and statistics will be saved to: {eda_plots_dir}\n")
    
    for device_id in [1, 2, 3]:
        file_path = processed_dir / f'device{device_id}.csv'
        if not file_path.exists():
            print(f"[WARNING] Processed file not found: {file_path}. Skipping EDA for Device {device_id}.")
            continue
            
        print(f"Processing Exploratory Data Analysis for Device {device_id}...")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Plot Class Distribution
            print("  - Generating class distribution plot...")
            plot_class_distribution(df, device_id, eda_plots_dir)
            
            # Plot Correlation Heatmap
            print("  - Generating correlation heatmap (Top 20 features)...")
            plot_correlation_heatmap(df, device_id, eda_plots_dir)
            
            # Plot Boxplots
            print("  - Generating discriminative feature boxplots (Top 5 features)...")
            plot_boxplots(df, device_id, eda_plots_dir)
            
            # Print Basic Stats
            print_basic_stats(df, device_id, eda_plots_dir)
            
        except Exception as e:
            print(f"[ERROR] Failed during EDA for Device {device_id}: {e}")

if __name__ == "__main__":
    run_eda()
