import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from codes.river_datasets import sea_concept_drift_dataset, generate_dataset_from_river_generator

# For reproducibility
np.random.seed(42)


def plot_concept_drift_sea():
    """
    Plot concept drift using SEA dataset from River library.
    
    This function creates a visualization showing how concept drift affects
    the relationship between features and target classes, while keeping
    the input features relatively stable.
    
    The plot shows:
    - Top 3 subplots: SEA features over time (relatively stable)
    - Bottom subplot: Class labels over time (showing concept drift)
    """
    # Parameters matching the style of drift_generation.py
    DF_SIZE = 3000
    drift_central_position = DF_SIZE // 2  # 1500
    drift_width = 750
    
    # Calculate drift boundaries
    drift_start = drift_central_position - drift_width // 2  # ~1125
    drift_end = drift_central_position + drift_width // 2    # ~1875
    
    print(f"Dataset size: {DF_SIZE}")
    print(f"Drift central position: {drift_central_position}")
    print(f"Drift width: {drift_width}")
    print(f"Drift start: {drift_start}")
    print(f"Drift end: {drift_end}")
    
    # Generate SEA dataset with concept drift
    sea_generator = sea_concept_drift_dataset(
        seed=42,
        drift_central_position=drift_central_position,
        drift_width=drift_width,
        stream_variant=0,  # Initial concept
        drift_variant=1    # Target concept after drift
    )
    
    # Convert to DataFrame
    sea_df = generate_dataset_from_river_generator(sea_generator, DF_SIZE)
    
    print(f"Generated dataset shape: {sea_df.shape}")
    print(f"Dataset columns: {sea_df.columns.tolist()}")
    print(f"Class distribution: {sea_df['class'].value_counts().to_dict()}")
    
    # Get feature columns (excluding class)
    feature_columns = [col for col in sea_df.columns if col != 'class']
    
    # Create figure with equal-height subplots (matching drift_generation.py style)
    fig, axes = plt.subplots(
        nrows=4,  # 3 features + 1 class
        ncols=1,
        figsize=(5, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1]},
    )
    
    # Plot the 3 SEA features (these remain relatively stable)
    colors = ['blue', 'blue', 'blue']  # Keep features in blue like original data
    
    for i, feature in enumerate(feature_columns[:3]):
        axes[i].plot(sea_df[feature], color=colors[i], linewidth=0.8)
        axes[i].set_ylabel(f'Feature {i+1}')
        
        # Set consistent y-axis limits for features
        feature_min = sea_df[feature].min()
        feature_max = sea_df[feature].max()
        feature_range = feature_max - feature_min
        axes[i].set_ylim(feature_min - 0.1 * feature_range, 
                        feature_max + 0.1 * feature_range)
    
    # Plot class labels (showing concept drift)
    # Use line plot for better visibility of concept drift
    axes[3].plot(sea_df.index, sea_df['class'], color='red', linewidth=0.8, alpha=0.8)
    
    # Add drift markers to the class subplot
    axes[3].axvline(x=drift_start, color="black", linestyle="--")
    axes[3].axvline(x=drift_end, color="black", linestyle="--", label="Concept Drift")
    axes[3].set_ylabel('Class')
    axes[3].set_ylim(-0.1, 1.1)  # Binary classes 0 and 1
    
    # Add some debugging info about class distribution in different regions
    pre_drift_classes = sea_df.loc[:drift_start, 'class'].value_counts()
    drift_classes = sea_df.loc[drift_start:drift_end, 'class'].value_counts()
    post_drift_classes = sea_df.loc[drift_end:, 'class'].value_counts()
    
    print(f"Pre-drift class distribution: {pre_drift_classes.to_dict()}")
    print(f"During-drift class distribution: {drift_classes.to_dict()}")
    print(f"Post-drift class distribution: {post_drift_classes.to_dict()}")
    
    # Set x-axis label for the last subplot
    axes[-1].set_xlabel('Index')
    
    # Align y-labels across all subplots
    fig.align_ylabels(axes)
    
    # Create a single legend outside the subplots
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Only add legend if we have the concept drift label
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    # Adjust spacing to reduce blank space at the top
    plt.subplots_adjust(
        hspace=0.2, top=0.98, bottom=0.18
    )  # Less top margin, keep bottom for legend
    
    # Save figure with extra space for the legend
    plt.savefig("concept_drift_sea_visualization.png", bbox_inches="tight", dpi=300)
    plt.show()


def plot_concept_drift_comparison():
    """
    Plot comparison between data drift and concept drift side by side.
    
    This creates a more comprehensive visualization showing the difference
    between data drift (features change) and concept drift (concept changes).
    """
    # Parameters
    DF_SIZE = 3000
    drift_central_position = DF_SIZE // 2
    drift_width = 750
    
    drift_start = drift_central_position - drift_width // 2
    drift_end = drift_central_position + drift_width // 2
    
    # Generate SEA dataset with concept drift
    sea_generator = sea_concept_drift_dataset(
        seed=42,
        drift_central_position=drift_central_position,
        drift_width=drift_width,
        stream_variant=0,
        drift_variant=1
    )
    
    sea_df = generate_dataset_from_river_generator(sea_generator, DF_SIZE)
    feature_columns = [col for col in sea_df.columns if col != 'class']
    
    # Create figure with 2 columns: Data Drift vs Concept Drift
    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1]},
    )
    
    # Left column: Simulated Data Drift (features change, concept stable)
    # For demonstration, we'll add artificial drift to features
    sea_df_data_drift = sea_df.copy()
    
    # Add artificial data drift to features
    for i, feature in enumerate(feature_columns[:3]):
        # Add step change to simulate data drift
        sea_df_data_drift.loc[drift_start:drift_end, feature] += 2.0
        
        axes[i, 0].plot(sea_df_data_drift[feature], color='orange', linewidth=0.8)
        axes[i, 0].axvline(x=drift_start, color="black", linestyle="--")
        axes[i, 0].axvline(x=drift_end, color="black", linestyle="--")
        axes[i, 0].set_ylabel(f'Feature {i+1}')
        if i == 0:
            axes[i, 0].set_title('Data Drift\n(Features Change)')
    
    # Left column: Class (no concept drift, just original mapping)
    original_sea_generator = sea_concept_drift_dataset(
        seed=42, drift_central_position=drift_central_position, drift_width=0,  # No drift
        stream_variant=0, drift_variant=0  # Same variant
    )
    original_sea_df = generate_dataset_from_river_generator(original_sea_generator, DF_SIZE)
    
    class_colors_orig = original_sea_df['class'].map({0: 'red', 1: 'blue'})
    axes[3, 0].scatter(original_sea_df.index, original_sea_df['class'], 
                      c=class_colors_orig, s=0.5, alpha=0.7)
    axes[3, 0].set_ylabel('Class')
    axes[3, 0].set_ylim(-0.1, 1.1)
    
    # Right column: Concept Drift (features stable, concept changes)
    for i, feature in enumerate(feature_columns[:3]):
        axes[i, 1].plot(sea_df[feature], color='blue', linewidth=0.8)
        axes[i, 1].set_ylabel(f'Feature {i+1}')
        if i == 0:
            axes[i, 1].set_title('Concept Drift\n(Concept Changes)')
    
    # Right column: Class (with concept drift)
    class_colors = sea_df['class'].map({0: 'red', 1: 'blue'})
    axes[3, 1].scatter(sea_df.index, sea_df['class'], 
                      c=class_colors, s=0.5, alpha=0.7)
    axes[3, 1].axvline(x=drift_start, color="black", linestyle="--")
    axes[3, 1].axvline(x=drift_end, color="black", linestyle="--", label="Concept Drift")
    axes[3, 1].set_ylabel('Class')
    axes[3, 1].set_ylim(-0.1, 1.1)
    
    # Set x-axis labels
    axes[-1, 0].set_xlabel('Index')
    axes[-1, 1].set_xlabel('Index')
    
    plt.tight_layout()
    plt.savefig("data_vs_concept_drift_comparison.png", bbox_inches="tight", dpi=300)
    plt.show()


def main():
    """Main function to run the concept drift visualization."""
    print("Generating concept drift visualization using SEA dataset...")
    plot_concept_drift_sea()
    
    print("\nGenerating comparison between data drift and concept drift...")
    plot_concept_drift_comparison()


if __name__ == "__main__":
    main()
