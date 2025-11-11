# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from matplotlib.colors import ListedColormap
# import os

# # For reproducibility
# np.random.seed(0)

# # --- Utility Functions ---

# def plot_feature_streams_subplots(fig, axes, X_stream, drift_point, title="Data Streams Over Time", highlight_range=None, range_color='green'):
#     """Plots Feature 1 and Feature 2 on separate subplots."""
#     ax1, ax2 = axes
#     fig.suptitle(title, fontsize=18, y=0.96)

#     # Plot for Feature 1
#     ax1.plot(X_stream[:, 0], color='darkcyan', alpha=0.8, linewidth=1.5, label='Feature 1')
#     ax1.axvline(x=drift_point, color='red', linestyle='--', linewidth=3, label='Drift Occurs')
#     ax1.set_ylabel('Feature 1 Value', fontsize=12)
#     ax1.grid(True, linestyle='--', alpha=0.5)
#     if highlight_range:
#         ax1.axvspan(highlight_range[0], highlight_range[1], color=range_color, alpha=0.1, label='Active Concept')
#     ax1.legend(loc='upper right')
#     ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#     # Plot for Feature 2
#     ax2.plot(X_stream[:, 1], color='purple', alpha=0.8, linewidth=1.5, label='Feature 2')
#     ax2.axvline(x=drift_point, color='red', linestyle='--', linewidth=3)
#     ax2.set_xlabel('Sample Index', fontsize=12)
#     ax2.set_ylabel('Feature 2 Value', fontsize=12)
#     ax2.grid(True, linestyle='--', alpha=0.5)
#     if highlight_range:
#         ax2.axvspan(highlight_range[0], highlight_range[1], color=range_color, alpha=0.1)
#     ax2.legend(loc='upper right')


# def plot_decision_boundary_full_page(ax, X, y, clf, title_suffix="", concept_rule_text="", rule_text_loc='upper left', rule_text_color='black'):
#     """Utility function to plot a single decision boundary to a full-page figure."""
#     cmap_light = ListedColormap(['#FFCCCC', '#CCE5FF'])
#     cmap_bold = ['#CC0000', '#0066CC']

#     h = .05 # Adjusted step for potentially wider data range
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
#     ax.scatter(X[y==0, 0], X[y==0, 1], c=cmap_bold[0], edgecolor='k', s=60, label='Class 0')
#     ax.scatter(X[y==1, 0], X[y==1, 1], c=cmap_bold[1], edgecolor='k', s=60, label='Class 1')
#     ax.set_title(title_suffix, fontsize=18)
#     ax.set_xlabel("Feature 1", fontsize=14)
#     ax.set_ylabel("Feature 2", fontsize=14)
#     ax.legend(loc='upper right', fontsize=12)
#     ax.grid(True, linestyle='--', alpha=0.6)

#     if concept_rule_text:
#         bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="0.5", alpha=0.8)
#         # Place text based on plot corners
#         x_range = x_max - x_min
#         y_range = y_max - y_min
#         if 'upper left' in rule_text_loc:
#             x_pos, y_pos, ha, va = (x_min + 0.05 * x_range, y_max - 0.05 * y_range, 'left', 'top')
#         elif 'lower right' in rule_text_loc:
#             x_pos, y_pos, ha, va = (x_max - 0.05 * x_range, y_min + 0.05 * y_range, 'right', 'bottom')
#         else:
#             x_pos, y_pos, ha, va = (x_min + 0.05 * x_range, y_max - 0.05 * y_range, 'left', 'top')
#         ax.text(x_pos, y_pos, concept_rule_text, fontsize=12, color=rule_text_color, bbox=bbox_props,
#                 horizontalalignment=ha, verticalalignment=va, transform=ax.transData)


# def generate_obvious_drift_images():
#     """
#     Generates seven images for a presentation on concept drift combined with data drift.
#     """
#     # *** Updated directory name as requested ***
#     output_dir = "data_drift_images"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     print(f"Saving images to: {os.path.abspath(output_dir)}")

#     # === 1. Generate Data with Data Drift ===
#     n_samples_per_concept = 500
#     drift_point = n_samples_per_concept
    
#     # Concept 1 Data: Centered around [0, 0]
#     mean1 = [0, 0]
#     cov1 = [[2, 0.8], [0.8, 2]]
#     X_concept1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_concept)
    
#     # Concept 2 Data: Centered around [4, 4] - THIS IS THE DATA DRIFT
#     mean2 = [4, 4]
#     cov2 = [[2, -0.8], [-0.8, 2]] # Also change covariance for more visual difference
#     X_concept2 = np.random.multivariate_normal(mean2, cov2, n_samples_per_concept)

#     # Apply different rules to each data distribution
#     y_concept1 = (X_concept1[:, 0] + X_concept1[:, 1] > 0.5).astype(int)
#     concept1_rule_text = "Concept 1 Rule:\nF1 + F2 > 0.5"
    
#     y_concept2 = (X_concept2[:, 0] - X_concept2[:, 1] > 0.5).astype(int)
#     concept2_rule_text = "Concept 2 Rule:\nF1 - F2 > 0.5"
    
#     # Create the full stream for plotting
#     X_stream_full = np.vstack([X_concept1, X_concept2])
#     y_stream_full = np.hstack([y_concept1, y_concept2])

#     # === 2. Train Models on Their Respective Concepts ===
#     model_concept1 = LogisticRegression(solver='lbfgs', random_state=42)
#     model_concept1.fit(X_concept1, y_concept1)
    
#     model_concept2 = LogisticRegression(solver='lbfgs', random_state=42)
#     model_concept2.fit(X_concept2, y_concept2)

#     # === 3. Generate and Save Separate Images ===

#     # --- Image 1 & 2: Before Drift ---
#     fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     plot_feature_streams_subplots(fig1, axes1, X_stream_full, drift_point, "1. Data Streams (Before Data Drift)",
#                                   highlight_range=[0, drift_point], range_color='lightgreen')
#     fig1.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(os.path.join(output_dir, "1_data_streams_before_drift.png"), dpi=200)
#     plt.close(fig1)
#     print("Generated '1_data_streams_before_drift.png'")

#     fig2, ax2 = plt.subplots(figsize=(10, 9))
#     plot_decision_boundary_full_page(ax2, X_concept1, y_concept1, model_concept1,
#                                      "2. Original Concept: Correct Model", 
#                                      concept_rule_text=concept1_rule_text, rule_text_loc='upper left', rule_text_color='green')
#     fig2.tight_layout()
#     plt.savefig(os.path.join(output_dir, "2_original_concept_boundary.png"), dpi=200)
#     plt.close(fig2)
#     print("Generated '2_original_concept_boundary.png'")

#     # --- Image 3 & 4: Drift Impact ---
#     fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     plot_feature_streams_subplots(fig3, axes3, X_stream_full, drift_point, "3. Data Streams (Obvious Data Drift Occurs)",
#                                   highlight_range=[drift_point, 2*drift_point], range_color='lightcoral')
#     fig3.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(os.path.join(output_dir, "3_data_streams_drift_occurs.png"), dpi=200)
#     plt.close(fig3)
#     print("Generated '3_data_streams_drift_occurs.png'")

#     fig4, ax4 = plt.subplots(figsize=(10, 9))
#     # Note: we plot the OLD model on the NEW data
#     plot_decision_boundary_full_page(ax4, X_concept2, y_concept2, model_concept1,
#                                      "4. Drift Impact: Old Model Fails on New Data",
#                                      concept_rule_text=f"Old Model Applied to New Data\nActual New Rule: {concept2_rule_text.replace(chr(10), ' ')}",
#                                      rule_text_loc='upper left', rule_text_color='red')
#     fig4.tight_layout()
#     plt.savefig(os.path.join(output_dir, "4_drift_impact_old_model_failing.png"), dpi=200)
#     plt.close(fig4)
#     print("Generated '4_drift_impact_old_model_failing.png'")

#     # --- Image 5 & 6: After Adaptation ---
#     fig5, axes5 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#     plot_feature_streams_subplots(fig5, axes5, X_stream_full, drift_point, "5. Data Streams (After Adaptation)",
#                                   highlight_range=[drift_point, 2*drift_point], range_color='lightblue')
#     fig5.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(os.path.join(output_dir, "5_data_streams_after_adaptation.png"), dpi=200)
#     plt.close(fig5)
#     print("Generated '5_data_streams_after_adaptation.png'")
    
#     fig6, ax6 = plt.subplots(figsize=(10, 9))
#     plot_decision_boundary_full_page(ax6, X_concept2, y_concept2, model_concept2,
#                                      "6. After Adaptation: Retrained Model is Correct",
#                                      concept_rule_text=concept2_rule_text, rule_text_loc='upper left', rule_text_color='blue')
#     fig6.tight_layout()
#     plt.savefig(os.path.join(output_dir, "6_after_adaptation_new_model_correct.png"), dpi=200)
#     plt.close(fig6)
#     print("Generated '6_after_adaptation_new_model_correct.png'")

#     # --- Image 7: Performance Degradation Over Time ---
#     fig7, ax7 = plt.subplots(figsize=(12, 7))
#     window_size = 50
#     accuracies = []
#     for i in range(0, 2*n_samples_per_concept - window_size + 1, window_size):
#         X_window = X_stream_full[i:i+window_size]
#         y_window = y_stream_full[i:i+window_size]
#         predictions = model_concept1.predict(X_window) 
#         accuracies.append(accuracy_score(y_window, predictions))

#     window_indices = np.arange(len(accuracies))
#     ax7.plot(window_indices, accuracies, marker='o', linestyle='-', color='#0066CC', linewidth=2, markersize=8, label='Model Accuracy')
#     drift_window_index = (drift_point / window_size) - 0.5 
#     ax7.axvline(x=drift_window_index, color='red', linestyle='--', linewidth=3, label='Concept & Data Drift Occur')
#     ax7.set_title("7. Performance Degradation (With Data Drift)", fontsize=16)
#     ax7.set_xlabel("Time (Data Window Index)", fontsize=12)
#     ax7.set_ylabel("Accuracy", fontsize=12)
#     ax7.set_ylim(0, 1.1)
#     ax7.grid(True, linestyle='--', alpha=0.6)
#     ax7.legend(fontsize=10)
#     fig7.tight_layout()
#     plt.savefig(os.path.join(output_dir, "7_performance_degradation.png"), dpi=200)
#     plt.close(fig7)
#     print("Generated '7_performance_degradation.png'")

# # Run the full image generation process
# generate_obvious_drift_images()



#####

import numpy as np
import matplotlib.pyplot as plt

# Number of samples
n_samples = 1000
drift_point = n_samples // 2  # point of abrupt drift

# Time axis
time = np.arange(n_samples)

# Feature 1: incremental (linear) virtual drift
feature_1 = np.random.normal(0, 1, n_samples) + (time * 0.01)

# Feature 2: abrupt virtual drift
feature_2 = np.random.normal(0, 1, n_samples)
feature_2[drift_point:] += 3

# Labels: fixed rule (no concept drift)
labels = (feature_1 + feature_2 > 0).astype(int)

# ---- Visualization ----
plt.figure(figsize=(14, 10))

# Stream of Feature 1
plt.subplot(3, 2, 1)
plt.scatter(time, feature_1, c=labels, cmap="coolwarm", s=10, alpha=0.7)
plt.axvline(drift_point, color="black", linestyle="--", label="Abrupt Drift Point")
plt.title("Feature 1 (Incremental Drift)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Stream of Feature 2
plt.subplot(3, 2, 2)
plt.scatter(time, feature_2, c=labels, cmap="coolwarm", s=10, alpha=0.7)
plt.axvline(drift_point, color="black", linestyle="--", label="Abrupt Drift Point")
plt.title("Feature 2 (Abrupt Drift)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

# Class stream
plt.subplot(3, 2, 3)
plt.scatter(time, labels, c=labels, cmap="coolwarm", s=10, alpha=0.7)
plt.axvline(drift_point, color="black", linestyle="--")
plt.title("Class Stream (Fixed Rule)")
plt.xlabel("Time")
plt.ylabel("Class")

# Distribution of Feature 1
plt.subplot(3, 2, 4)
plt.hist(feature_1[:drift_point], bins=30, alpha=0.5, label="Before Drift")
plt.hist(feature_1[drift_point:], bins=30, alpha=0.5, label="After Drift")
plt.title("Distribution of Feature 1 (Incremental Drift)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

# Distribution of Feature 2
plt.subplot(3, 2, 5)
plt.hist(feature_2[:drift_point], bins=30, alpha=0.5, label="Before Drift")
plt.hist(feature_2[drift_point:], bins=30, alpha=0.5, label="After Drift")
plt.title("Distribution of Feature 2 (Abrupt Drift)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()

# Joint feature space (2D scatter)
plt.subplot(3, 2, 6)
plt.scatter(feature_1, feature_2, c=labels, cmap="coolwarm", s=10, alpha=0.7)
plt.title("Joint Feature Space (Virtual Drift)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
