import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load the CSV file
df = pd.read_csv('comparison_results/consolidated_results.csv')

# ✅ Define the batch sizes you want to plot
batch_sizes = [1000, 1500, 2000, 2500]  # <-- Change this list as needed

# Filter for the selected batch sizes
filtered_df = df[df['batch_size'].isin(batch_sizes)]

# Set plot style
sns.set(style="whitegrid")

# Define unique markers for techniques
markers = ['o', 's', 'D', '^', 'P', 'X', '*', 'v']
techniques = filtered_df['technique'].unique()
marker_map = {tech: markers[i % len(markers)] for i, tech in enumerate(techniques)}

# ✅ Dynamically set subplot layout
n = len(batch_sizes)
fig, axes = plt.subplots(n, 2, figsize=(14, 6 * n), squeeze=False)

# Plot for each batch size
for row, batch in enumerate(batch_sizes):
    subset = filtered_df[filtered_df['batch_size'] == batch]
    
    # F1 Plot
    ax_f1 = axes[row, 0]
    for tech in techniques:
        tech_data = subset[subset['technique'] == tech]
        ax_f1.scatter(tech_data['num_drifts'], tech_data['f1'],
                      label=tech, marker=marker_map[tech], s=100)
    ax_f1.set_title(f'F1 Score vs Number of Detections (Batch Size {batch})')
    ax_f1.set_xlabel('Number of Detections')
    ax_f1.set_ylabel('F1 Score')
    ax_f1.legend(title="Technique")

    # AUC Plot
    ax_auc = axes[row, 1]
    for tech in techniques:
        tech_data = subset[subset['technique'] == tech]
        ax_auc.scatter(tech_data['num_drifts'], tech_data['auc'],
                       label=tech, marker=marker_map[tech], s=100)
    ax_auc.set_title(f'AUC vs Number of Detections (Batch Size {batch})')
    ax_auc.set_xlabel('Number of Detections')
    ax_auc.set_ylabel('AUC')
    ax_auc.legend(title="Technique")

plt.tight_layout()
plt.show()
