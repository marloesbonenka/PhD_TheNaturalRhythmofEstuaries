"""Discharge analysis functions for global data analysis"""

#%% --- IMPORTS ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#%%

def analyze_discharge_metrics(estuary_discharge_data):
    """
    Calculate discharge variability, intermittency, and flashiness metrics for estuaries.
    
    Parameters:
        estuary_discharge_data (dict): Dictionary of estuary discharge time series
        
    Returns:
        pd.DataFrame: DataFrame containing calculated metrics
    """
    results = []
    
    for estuary, q in estuary_discharge_data.items():
        # Handle both pd.Series and np.ndarray
        if hasattr(q, 'values'):
            q = q.values  # extract numpy array from Series, dropping datetime index
            q = np.array(q, dtype=float)
            q = q[~np.isnan(q)]
        
        if len(q) == 0:
            print(f"  WARNING: {estuary} has no valid data. Skipping.")
            continue

        mean_q = np.mean(q)
        max_q = np.max(q)
        min_q = np.min(q)
        std_q = np.std(q)
        cv = std_q / mean_q if mean_q != 0 else np.nan

        # Actual zero-flow intermittency
        zero_flow_intermittency = np.sum(q == 0) / len(q)

        # Relative low-flow (below 5th percentile) intermittency
        q5 = np.percentile(q, 5)
        relative_zero_flow_intermittency = np.sum(q < q5) / len(q)

        # Flashiness: P90 / P10
        p90 = np.percentile(q, 90)
        p10 = np.percentile(q, 10)
        flashiness = p90 / p10 if p10 != 0 else np.nan

        # New metric: peak relative to mean
        p95 = np.percentile(q, 95)
        peak_to_mean = p95 / mean_q if mean_q != 0 else np.nan

        results.append({
            'Estuary': estuary,
            'Mean': mean_q,
            'Max': max_q,
            'Min': min_q,
            'Std': std_q,
            'CV': cv,
            'Zero-Flow Intermittency': zero_flow_intermittency,
            'Relative Zero-Flow Intermittency (Q < Q5)': relative_zero_flow_intermittency,
            'P95': p95, 
            'P90': p90,
            'P10': p10,
            'Flashiness (P90/P10)': flashiness,
            'Peak-to-Mean (P95/Mean)': peak_to_mean
        })
    
    return pd.DataFrame(results)

def visualize_discharge_metrics(df, output_dir="04_Metrics_per_estuary"):
    """
    Visualizes discharge metrics using bar and scatter plots with improved formatting.
    Ensures all axes start at zero, clearly identifies estuaries, and has proper spacing
    to prevent text from being cut off.

    Args:
        df (pd.DataFrame): DataFrame containing estuary discharge metrics.
        output_dir (str): Directory to save the plots.
    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a custom color palette for consistent estuary colors across plots
    num_estuaries = len(df)
    # Use tab20 for up to 20 unique colors, repeat if more
    base_cmap = plt.cm.get_cmap('tab20')
    colors = [base_cmap(i % base_cmap.N) for i in range(num_estuaries)]
    estuary_colors = dict(zip(df['Estuary'], colors))
    
    # 1. Bar Chart for Mean Discharge
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Estuary'], df['Mean'], color=[estuary_colors[e] for e in df['Estuary']])
    plt.xlabel('Estuary', fontsize=12)
    plt.ylabel('Mean Discharge (m³/s)', fontsize=12)
    plt.title('Mean Discharge by Estuary', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Add more space at the bottom and top of the plot
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(str(Path(output_dir) / 'mean_discharge_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Scatter Plot: Mean vs. CV with estuary labels
    plt.figure(figsize=(12, 10))
    for estuary in df['Estuary']:
        estuary_data = df[df['Estuary'] == estuary]
        plt.scatter(estuary_data['Mean'], estuary_data['CV'], 
                   color=estuary_colors[estuary], s=100, label=estuary)
        
        # Adjust text positions to avoid overlap
        x_pos = estuary_data['Mean'].values[0]
        y_pos = estuary_data['CV'].values[0]
        # Add some offset for text to avoid overlapping with points
        plt.annotate(estuary, (x_pos, y_pos),
                    xytext=(7, 7), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.xlabel('Mean Discharge (m³/s)', fontsize=12)
    plt.ylabel('Coefficient of Variation (CV)', fontsize=12)
    plt.title('Mean Discharge vs. Coefficient of Variation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)  # Ensure x-axis starts at zero
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add a legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.75)  # Make room for the legend
    
    plt.savefig(str(Path(output_dir) / 'mean_vs_cv_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Bar Chart for CV
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Estuary'], df['CV'], color=[estuary_colors[e] for e in df['Estuary']])
    plt.xlabel('Estuary', fontsize=12)
    plt.ylabel('Coefficient of Variation (CV)', fontsize=12)
    plt.title('Coefficient of Variation by Estuary', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(str(Path(output_dir) / 'cv_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4a. Bar Chart for Flashiness (P90/P10)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Estuary'], df['Flashiness (P90/P10)'], 
                  color=[estuary_colors[e] for e in df['Estuary']])
    plt.xlabel('Estuary', fontsize=12)
    plt.ylabel('Flashiness (P90/P10)', fontsize=12)
    plt.title('Flashiness by Estuary', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(str(Path(output_dir) / 'flashiness_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4b. Bar Chart for peak to mean (P95/Mean)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Estuary'], df['Peak-to-Mean (P95/Mean)'], 
                  color=[estuary_colors[e] for e in df['Estuary']])
    plt.xlabel('Estuary', fontsize=12)
    plt.ylabel('Peak-to-Mean (P95/Mean)', fontsize=12)
    plt.title('Peak-to-Mean Ratio by Estuary', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(str(Path(output_dir) / 'peak_to_mean_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Scatter Plot: Mean vs. Flashiness with estuary labels
    plt.figure(figsize=(12, 10))
    for estuary in df['Estuary']:
        estuary_data = df[df['Estuary'] == estuary]
        plt.scatter(estuary_data['Mean'], estuary_data['Flashiness (P90/P10)'], 
                   color=estuary_colors[estuary], s=100, label=estuary)
        
        # Adjust text positions to avoid overlap
        x_pos = estuary_data['Mean'].values[0]
        y_pos = estuary_data['Flashiness (P90/P10)'].values[0]
        plt.annotate(estuary, (x_pos, y_pos),
                    xytext=(7, 7), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.xlabel('Mean Discharge (m³/s)', fontsize=12)
    plt.ylabel('Flashiness (P90/P10)', fontsize=12)
    plt.title('Mean Discharge vs. Flashiness', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)  # Ensure x-axis starts at zero
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add a legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.75)
    
    plt.savefig(str(Path(output_dir) / 'mean_vs_flashiness_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation heatmap between metrics
    metrics_for_corr = ['Mean', 'Max', 'Min', 'Std', 'CV', 
                        'Zero-Flow Intermittency', 
                        'Relative Zero-Flow Intermittency (Q < Q5)', 
                        'Flashiness (P90/P10)']
    
    corr_matrix = df[metrics_for_corr].corr()
    
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                         linewidths=0.5, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Between Discharge Metrics', fontsize=14)
    
    # Ensure heatmap labels are visible
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Adjust layout for heatmap
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / 'metrics_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_discharge_metrics_comparison(df_raw, df_moving_avg, output_dir):
    """
    Compare metrics between raw and moving average discharge data.
    
    Parameters:
        df_raw (pd.DataFrame): Metrics from raw discharge data
        df_moving_avg (pd.DataFrame): Metrics from moving average discharge data
        output_dir (str or Path): Directory to save figures
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    estuaries = df_raw['Estuary'].tolist()
    x = np.arange(len(estuaries))
    width = 0.35

    for metric, ylabel, filename in [
        ('Mean',               'Mean Discharge (m³/s)', 'comparison_mean.png'),
        ('CV',                 'Coefficient of Variation (-)', 'comparison_cv.png'),
        ('Flashiness (P90/P10)', 'Flashiness (P90/P10) (-)', 'comparison_flashiness.png'),
    ]:
        fig, ax = plt.subplots(figsize=(16, 7))
        
        bars1 = ax.bar(x - width/2, df_raw[metric],       width, label='Raw',            color='steelblue')
        bars2 = ax.bar(x + width/2, df_moving_avg[metric], width, label='Moving Average', color='coral')

        ax.set_xlabel('Estuary', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{metric}: Raw vs. Moving Average', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(estuaries, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Comparison plots saved to {output_dir}")