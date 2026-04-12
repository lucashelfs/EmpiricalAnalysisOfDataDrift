import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# For reproducibility
np.random.seed(42)


def plot_decision_boundary(X, y, clf, ax, title):
    """Plot decision boundary for a given classifier and dataset."""
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    ax.set_title(title, fontsize=10)


def plot_concept_drift_streams():
    """
    Plot concept drift streams showing feature and class evolution over time,
    with decision boundaries at the bottom to demonstrate concept drift.
    
    This visualization demonstrates how the same feature distributions can have
    different class mappings due to concept drift. Shows side-by-side comparison
    of original concept vs drifted concept as time series streams, plus decision boundaries.
    """
    
    # Generate original dataset (concept 1) - using your exact parameters
    X1, y1 = make_classification(n_samples=500, n_features=2, n_informative=2,
                                 n_redundant=0, n_clusters_per_class=1,
                                 class_sep=1.5, flip_y=0.01, random_state=42)
    
    # Generate drift dataset (concept 2) - using your exact parameters  
    X2, y2 = make_classification(n_samples=500, n_features=2, n_informative=2,
                                 n_redundant=0, n_clusters_per_class=2,
                                 class_sep=0.7, flip_y=0.10, random_state=12)
    
    print(f"Concept 1 dataset shape: {X1.shape}")
    print(f"Concept 1 class distribution: {np.bincount(y1)}")
    print(f"Concept 2 dataset shape: {X2.shape}")
    print(f"Concept 2 class distribution: {np.bincount(y2)}")
    
    # Train classifier on Concept 1
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X1, y1)
    
    # Create figure with side-by-side layout (4 rows now)
    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(12, 12),
        gridspec_kw={"height_ratios": [1, 1, 1, 1.2]},  # Make decision boundary plots slightly larger
    )
    
    # Column titles
    axes[0, 0].set_title('Concept 1 (Original)\nclass_sep=1.5, flip_y=0.01', fontsize=12, pad=20)
    axes[0, 1].set_title('Concept 2 (Drift)\nclass_sep=0.7, flip_y=0.10', fontsize=12, pad=20)
    
    # Left column: Concept 1 streams
    # Feature 1 stream
    axes[0, 0].plot(X1[:, 0], color='blue', linewidth=0.8, alpha=0.8)
    axes[0, 0].set_ylabel('Feature 1')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature 2 stream  
    axes[1, 0].plot(X1[:, 1], color='blue', linewidth=0.8, alpha=0.8)
    axes[1, 0].set_ylabel('Feature 2')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Class stream
    axes[2, 0].plot(y1, 'ro', markersize=2, alpha=0.7)
    axes[2, 0].set_ylabel('Class')
    axes[2, 0].set_ylim(-0.1, 1.1)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Right column: Concept 2 streams
    # Feature 1 stream
    axes[0, 1].plot(X2[:, 0], color='blue', linewidth=0.8, alpha=0.8)
    axes[0, 1].set_ylabel('Feature 1')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature 2 stream
    axes[1, 1].plot(X2[:, 1], color='blue', linewidth=0.8, alpha=0.8)
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Class stream
    axes[2, 1].plot(y2, 'ro', markersize=2, alpha=0.7)
    axes[2, 1].set_ylabel('Class')
    axes[2, 1].set_ylim(-0.1, 1.1)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Bottom row: Decision boundaries
    # Left: Classifier trained on Concept 1, applied to Concept 1 (good fit)
    plot_decision_boundary(X1, y1, clf, axes[3, 0], 
                          'Decision Boundary\n(Trained on Concept 1)')
    axes[3, 0].set_xlabel('Feature 1')
    axes[3, 0].set_ylabel('Feature 2')
    
    # Right: Same classifier applied to Concept 2 (poor fit - shows concept drift)
    plot_decision_boundary(X2, y2, clf, axes[3, 1], 
                          'Same Decision Boundary\n(Applied to Concept 2)')
    axes[3, 1].set_xlabel('Feature 1')
    axes[3, 1].set_ylabel('Feature 2')
    
    # Add x-axis labels for stream plots
    axes[2, 0].set_xlabel('Sample Index')
    axes[2, 1].set_xlabel('Sample Index')
    
    # Align y-labels across all subplots
    fig.align_ylabels(axes)
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.94, bottom=0.06)
    
    # Save figure
    plt.savefig("concept_drift_streams_with_boundaries.png", bbox_inches="tight", dpi=300)
    plt.show()
    
    # Print some statistics for analysis
    print(f"\nConcept 1 Feature Statistics:")
    print(f"Feature 1 - Mean: {X1[:, 0].mean():.3f}, Std: {X1[:, 0].std():.3f}")
    print(f"Feature 2 - Mean: {X1[:, 1].mean():.3f}, Std: {X1[:, 1].std():.3f}")
    
    print(f"\nConcept 2 Feature Statistics:")
    print(f"Feature 1 - Mean: {X2[:, 0].mean():.3f}, Std: {X2[:, 0].std():.3f}")
    print(f"Feature 2 - Mean: {X2[:, 1].mean():.3f}, Std: {X2[:, 1].std():.3f}")


def plot_combined_stream():
    """
    Plot a combined stream showing concept drift transition over time.
    
    This creates a single timeline where concept drift occurs at a specific point,
    similar to real-world streaming scenarios.
    """
    
    # Generate datasets
    X1, y1 = make_classification(n_samples=250, n_features=2, n_informative=2,
                                 n_redundant=0, n_clusters_per_class=1,
                                 class_sep=1.5, flip_y=0.01, random_state=42)
    
    X2, y2 = make_classification(n_samples=250, n_features=2, n_informative=2,
                                 n_redundant=0, n_clusters_per_class=2,
                                 class_sep=0.7, flip_y=0.10, random_state=12)
    
    # Combine into single stream
    X_combined = np.vstack([X1, X2])
    y_combined = np.hstack([y1, y2])
    drift_point = len(X1)
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )
    
    # Feature 1 stream
    axes[0].plot(X_combined[:, 0], color='blue', linewidth=0.8, alpha=0.8)
    axes[0].axvline(x=drift_point, color='red', linestyle='--', linewidth=2, label='Concept Drift')
    axes[0].set_ylabel('Feature 1')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Feature 2 stream
    axes[1].plot(X_combined[:, 1], color='blue', linewidth=0.8, alpha=0.8)
    axes[1].axvline(x=drift_point, color='red', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True, alpha=0.3)
    
    # Class stream
    axes[2].plot(y_combined, 'ro', markersize=2, alpha=0.7)
    axes[2].axvline(x=drift_point, color='red', linestyle='--', linewidth=2)
    axes[2].set_ylabel('Class')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Sample Index')
    
    # Add annotations
    axes[0].text(drift_point/2, axes[0].get_ylim()[1]*0.9, 'Concept 1', 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[0].text(drift_point + (len(X_combined)-drift_point)/2, axes[0].get_ylim()[1]*0.9, 'Concept 2', 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    plt.suptitle('Concept Drift in Streaming Data', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig("concept_drift_combined_stream.png", bbox_inches="tight", dpi=300)
    plt.show()


def main():
    """Main function to run the concept drift stream visualizations."""
    print("Generating concept drift streams visualization...")
    plot_concept_drift_streams()
    
    print("\nGenerating combined stream with concept drift...")
    plot_combined_stream()


if __name__ == "__main__":
    main()
