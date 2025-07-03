# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load CSV
# df = pd.read_csv('comparison_results/consolidated_results_from_git.csv')

# # Filter desired batch sizes
# batch_sizes = [1000, 1500, 2000, 2500]
# filtered_df = df[df['batch_size'].isin(batch_sizes)]

# # Aggregate: mean F1 per technique and batch size
# agg = filtered_df.groupby(['technique', 'batch_size']).agg({
#     'f1': 'mean',
#     'auc': 'mean'
# }).reset_index()

# # # Plot (F1 example)
# # plt.figure(figsize=(8, 6))
# # sns.barplot(
# #     data=agg,
# #     x='technique', y='f1',
# #     hue='batch_size',
# #     palette='pastel'
# # )

# # plt.title('F1 Score per Technique (by Batch Size)')
# # plt.xlabel('Drift Detection Technique')
# # plt.ylabel('F1 Score')
# # plt.legend(title='Batch Size')
# # plt.tight_layout()
# # plt.show()


# # Plot (AUC)
# plt.figure(figsize=(8, 6))
# sns.barplot(
#     data=agg,
#     x='technique', y='auc',
#     hue='batch_size',
#     palette='pastel'
# )

# plt.title('AUC per Technique (by Batch Size)')
# plt.xlabel('Drift Detection Technique')
# plt.ylabel('AUC')
# plt.legend(title='Batch Size')
# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv('comparison_results/consolidated_results.csv')

# Filter desired batch sizes
batch_sizes = [1000, 1500, 2000, 2500]
filtered_df = df[df['batch_size'].isin(batch_sizes)]

# Aggregate metrics
agg = filtered_df.groupby(['technique', 'batch_size']).agg({
    'f1': 'mean',
    'auc': 'mean'
}).reset_index()

# Set seaborn style
sns.set(style="whitegrid")

# --- Plot F1 ---
fig_f1, ax_f1 = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=agg,
    x='technique', y='f1',
    hue='batch_size',
    palette='pastel',
    ax=ax_f1
)
ax_f1.set_title('Mean F1 Score per Technique (by Batch Size)', fontsize=20)
ax_f1.set_xlabel('Drift Detection Technique', fontsize=20)
ax_f1.set_ylabel('F1 Score', fontsize=20)
ax_f1.tick_params(axis='both', which='major', labelsize=15)
ax_f1.legend_.remove()  # Remove legend for separate file
fig_f1.tight_layout()
fig_f1.savefig('f1_per_technique.png', bbox_inches='tight')

# --- Plot AUC ---
fig_auc, ax_auc = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=agg,
    x='technique', y='auc',
    hue='batch_size',
    palette='pastel',
    ax=ax_auc
)
ax_auc.set_title('Mean AUC per Technique (by Batch Size)', fontsize=20)
ax_auc.set_xlabel('Drift Detection Technique', fontsize=20)
ax_auc.set_ylabel('AUC', fontsize=20)
ax_auc.tick_params(axis='both', which='major', labelsize=15)
ax_auc.legend_.remove()  # Remove legend for separate file
fig_auc.tight_layout()
fig_auc.savefig('auc_per_technique.png', bbox_inches='tight')


# --- Save Legend Separately (Horizontal) ---
fig_legend = plt.figure(figsize=(8, 1))

# Get handles and labels from the axis (not from the legend object)
handles, labels = ax_f1.get_legend_handles_labels()

# Create legend figure
fig_legend = plt.figure(figsize=(8, 1))
fig_legend.legend(
    handles, labels,
    loc='center',
    ncol=len(labels),  # Horizontal layout
    frameon=False
)
fig_legend.tight_layout()
fig_legend.savefig('legend_horizontal.png', dpi=300)  # or .pdf
plt.close(fig_legend)

# Close main plots to clean memory
plt.close(fig_f1)
plt.close(fig_auc)

print("Plots and legend saved successfully!")
