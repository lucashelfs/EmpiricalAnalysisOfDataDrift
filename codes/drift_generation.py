import math
import os
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from codes.config import comparisons_output_dir as output_dir
from codes.drift_measures import DataFrameComparator

# For reproducibility
np.random.seed(42)


def create_simple_dataframe(dataframe_size: int) -> pd.DataFrame:
    """Create a simple default dataframe."""
    data = {
        "feature1": np.random.normal(loc=0, scale=1, size=dataframe_size),
        "feature2": np.random.normal(loc=5, scale=2, size=dataframe_size),
    }
    return pd.DataFrame(data)


def add_abrupt_drift(
    df: pd.DataFrame, column: str, start_index: int, end_index: int, change: float
) -> pd.DataFrame:
    """
    Apply a sudden drift to a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the drift will be applied.
    column (str): The name of the column where the drift will be applied.
    start_index (int): The index from which the drift will start.
    change (float): The amount of change to apply to the column values.

    Returns:
    pd.DataFrame: The DataFrame with the applied sudden drift.
    """
    df.loc[start_index:end_index, column] += change
    return df


def add_gradual_drift(
    df: pd.DataFrame, column: str, start_index: int, end_index: int, max_change: float
) -> pd.DataFrame:
    """
    Apply a gradual drift to a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the drift will be applied.
    column (str): The name of the column where the drift will be applied.
    start_index (int): The index from which the drift will start.
    end_index (int): The index at which the drift will end.
    max_change (float): The maximum amount of change to apply to the column values.

    Returns:
    pd.DataFrame: The DataFrame with the applied gradual drift.
    """
    # Probability of drift starts low and increases linearly
    drift_probabilities = np.linspace(0, 1, end_index - start_index)

    for i, prob in zip(range(start_index, end_index), drift_probabilities):
        if np.random.rand() < prob:
            drift_amount = np.random.uniform(0, max_change)
            df.at[i, column] += drift_amount

    return df


def add_incremental_drift(
    df: pd.DataFrame,
    column: str,
    start_index: int,
    end_index: int,
    change: float,
    step: float,
) -> pd.DataFrame:
    """
    Apply an incremental drift to a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the drift will be applied.
    column (str): The name of the column where the drift will be applied.
    start_index (int): The index from which the drift will start.
    change (float): The initial amount of change to apply to the column values.
    step (float): The incremental step to increase the change.

    Returns:
    pd.DataFrame: The DataFrame with the applied incremental drift.
    """
    for i in range(start_index, end_index):
        df.loc[i, column] += change
        change += step
    return df


def calculate_index(df: pd.DataFrame, drift_value: float) -> int:
    """
    Calculate the index based on the drift value.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the drift will be applied.
    drift_value (float): The drift value which can be an integer or a float.

    Returns:
    int: The calculated index.
    """
    if isinstance(drift_value, int):
        return drift_value
    else:
        return int(df.shape[0] * drift_value)


def print_dataframe_heads(df, df_sudden, df_gradual, df_incremental):
    print("Original DataFrame:\n", df.head())
    print("DataFrame with Sudden Drift:\n", df_sudden.head())
    print("DataFrame with Gradual Drift:\n", df_gradual.head())
    print("DataFrame with Incremental Drift:\n", df_incremental.head())


# def plot_data_disturbance():
#     """
#     Plot data disturbance by applying different types of drifts to a DataFrame.
#
#     This function creates a simple DataFrame and applies sudden, gradual, and incremental drifts
#     to one of its columns. It then plots the original and drifted data for comparison.
#
#     The function performs the following steps:
#     1. Creates a simple DataFrame with random data.
#     2. Applies a abrupt drift to the 'feature1' column.
#     3. Applies a gradual drift to the 'feature1' column.
#     4. Applies an incremental drift to the 'feature1' column.
#     5. Plots the original data and the data with each type of drift in separate subplots.
#
#     Parameters:
#     None
#
#     Returns:
#     None
#     """
#     DF_SIZE = 3000
#     df = create_simple_dataframe(dataframe_size=DF_SIZE)
#
#     # Apply gradual drift
#     df_gradual = df.copy()
#     df_gradual = add_gradual_drift(
#         df_gradual, "feature1", start_index=0, end_index=(DF_SIZE // 4), max_change=5
#     )
#
#     # Apply incremental drift
#     df_incremental = df.copy()
#     df_incremental = add_incremental_drift(
#         df_incremental,
#         "feature1",
#         start_index=(DF_SIZE // 4),
#         end_index=(DF_SIZE // 2),
#         change=0.1,
#         step=0.1,
#     )
#
#     # Apply sudden drift
#     df_sudden = df.copy()
#     df_sudden = add_abrupt_drift(
#         df_sudden,
#         "feature1",
#         start_index=(DF_SIZE // 2),
#         end_index=(DF_SIZE // 2 + DF_SIZE // 4),
#         change=5,
#     )
#
#     # Plot the original and drifted data
#     plt.figure(figsize=(12, 8))
#
#     # Original data
#     plt.subplot(4, 1, 1)
#     plt.plot(df["feature1"], label="Original")
#     plt.title("Original Data")
#     plt.xlabel("Index")
#     plt.ylabel("feature1")
#     plt.legend()
#
#     # Sudden drift
#     plt.subplot(4, 1, 2)
#     plt.plot(df_sudden["feature1"], label="Sudden Drift", color="orange")
#     plt.title("Sudden Drift")
#     plt.xlabel("Index")
#     plt.ylabel("feature1")
#     plt.legend()
#
#     # Gradual drift
#     plt.subplot(4, 1, 3)
#     plt.plot(df_gradual["feature1"], label="Gradual Drift", color="green")
#     plt.title("Gradual Drift")
#     plt.xlabel("Index")
#     plt.ylabel("feature1")
#     plt.legend()
#
#     # Incremental drift
#     plt.subplot(4, 1, 4)
#     plt.plot(df_incremental["feature1"], label="Incremental Drift", color="red")
#     plt.title("Incremental Drift")
#     plt.xlabel("Index")
#     plt.ylabel("feature1")
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()

## This one is good

# def plot_data_disturbance():
#     """
#     Plot data disturbance by applying different types of drifts to a DataFrame.
#
#     This function creates a simple DataFrame and applies sudden and incremental drifts
#     to one of its columns. It then plots the original and drifted data for comparison.
#
#     Parameters:
#     None
#
#     Returns:
#     None
#     """
#     DF_SIZE = 3000
#     start_drift = DF_SIZE // 2 - DF_SIZE // 4  # Start index of drifts
#     end_drift = DF_SIZE // 2 + DF_SIZE // 4  # End index of drifts
#
#     df = create_simple_dataframe(dataframe_size=DF_SIZE)
#
#     # Apply incremental drift
#     df_incremental = df.copy()
#     df_incremental = add_incremental_drift(
#         df_incremental,
#         "feature1",
#         start_index=start_drift,
#         end_index=end_drift,
#         change=0.1,
#         step=0.01,
#     )
#
#     # Apply sudden drift
#     df_sudden = df.copy()
#     df_sudden = add_abrupt_drift(
#         df_sudden,
#         "feature1",
#         start_index=start_drift,
#         end_index=end_drift,
#         change=5,
#     )
#
#     # Plot the original and drifted data
#     plt.figure(figsize=(12, 6))
#
#     # Original data
#     plt.subplot(3, 1, 1)
#     plt.plot(df["feature1"], label="Original")
#     plt.title("Original Data")
#     plt.xlabel("Index")
#     plt.ylabel("Feature 1")
#     plt.axvline(x=start_drift, color="black", linestyle="--", label="Drift Start")
#     plt.axvline(x=end_drift, color="gray", linestyle="--", label="Drift End")
#
#     # Sudden drift
#     plt.subplot(3, 1, 2)
#     plt.plot(df_sudden["feature1"], label="Abrupt Drift", color="orange")
#     plt.title("Abrupt Drift")
#     plt.xlabel("Index")
#     plt.ylabel("Feature 1")
#     plt.axvline(x=start_drift, color="black", linestyle="--", label="Drift Start")
#     plt.axvline(x=end_drift, color="black", linestyle="--", label="Drift End")
#
#     # Incremental drift
#     plt.subplot(3, 1, 3)
#     plt.plot(df_incremental["feature1"], label="Incremental Drift", color="red")
#     plt.title("Incremental Drift")
#     plt.xlabel("Index")
#     plt.ylabel("Feature 1")
#     plt.axvline(x=start_drift, color="black", linestyle="--", label="Drift Start")
#     plt.axvline(x=end_drift, color="black", linestyle="--", label="Drift End")
#
#     plt.tight_layout()
#     plt.show()


def plot_data_disturbance():
    """
    Plot data disturbance by applying different types of drifts to a DataFrame.

    This function creates a simple DataFrame and applies sudden and incremental drifts
    to one of its columns. It then plots the original and drifted data for comparison.

    Parameters:
    None

    Returns:
    None
    """
    DF_SIZE = 3000
    window_len = 750

    quarter = DF_SIZE // 6

    start_drift_1 = quarter
    end_drift_1 = quarter + window_len  # End index of drifts

    start_drift_2 = quarter * 4  # Start index of drifts
    end_drift_2 = quarter * 4 + window_len  # End index of drifts

    df = create_simple_dataframe(dataframe_size=DF_SIZE)

    # Apply incremental drift
    df_incremental = df.copy()

    df_incremental = add_incremental_drift(
        df_incremental,
        "feature1",
        start_index=start_drift_1,
        end_index=end_drift_1,
        change=0.1,
        step=0.0030,
    )

    df_incremental = add_incremental_drift(
        df_incremental,
        "feature1",
        start_index=start_drift_2,
        end_index=end_drift_2,
        change=0.1,
        step=0.0030,
    )

    # Apply sudden drift
    df_sudden = df.copy()
    df_sudden = add_abrupt_drift(
        df_sudden,
        "feature1",
        start_index=start_drift_1,
        end_index=end_drift_1,
        change=2,
    )

    df_sudden = add_abrupt_drift(
        df_sudden,
        "feature1",
        start_index=start_drift_2,
        end_index=end_drift_2,
        change=2,
    )

    # Create figure with equal-height subplots
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(5, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]},
    )

    # Original data
    axes[0].plot(df["feature1"], label="Original", color="blue")
    # axes[0].axvline(x=start_drift_1, color="black", linestyle="--")
    # axes[0].axvline(x=end_drift_1, color="black", linestyle="--")
    # axes[0].axvline(x=start_drift_2, color="black", linestyle="--")
    # axes[0].axvline(x=end_drift_2, color="black", linestyle="--")
    axes[0].set_ylabel("Feature 1")
    axes[0].set_ylim(-4, 5)  # Set Y-axis limits

    # Sudden drift
    axes[1].plot(df_sudden["feature1"], label="Abrupt Drift", color="orange")
    axes[1].axvline(x=start_drift_1, color="black", linestyle="--")
    axes[1].axvline(x=end_drift_1, color="black", linestyle="--", label="Data Drift")
    axes[1].axvline(x=start_drift_2, color="black", linestyle="--")
    axes[1].axvline(x=end_drift_2, color="black", linestyle="--")
    axes[1].set_ylabel("Feature 2")
    axes[1].set_ylim(-4, 5)  # Set Y-axis limits

    # Incremental drift
    axes[2].plot(df_incremental["feature1"], label="Incremental Drift", color="red")
    axes[2].axvline(x=start_drift_1, color="black", linestyle="--")
    axes[2].axvline(x=end_drift_1, color="black", linestyle="--")
    axes[2].axvline(x=start_drift_2, color="black", linestyle="--")
    axes[2].axvline(x=end_drift_2, color="black", linestyle="--")
    axes[2].set_ylabel("Feature 3")
    axes[2].set_ylim(-4, 5)  # Set Y-axis limits

    axes[-1].set_xlabel("Index")

    # Align y-labels across all subplots
    fig.align_ylabels(axes)

    # Create a single legend outside the subplots
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    handles = [h for h, l in zip(handles, labels) if l != "Data Drift"] + [
        handles[labels.index("Data Drift")]
    ]
    labels = [l for l in labels if l != "Data Drift"] + ["Data Drift"]
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02))

    # Adjust spacing to reduce blank space at the top
    plt.subplots_adjust(
        hspace=0.2, top=0.98, bottom=0.18
    )  # Less top margin, keep bottom for legend

    # Save figure with extra space for the legend
    plt.savefig("plot_of_disturbances.png", bbox_inches="tight", dpi=300)
    plt.show()


def there_is_space_for_all_drifts(num_drifts, drift_length, index_space_size):
    if num_drifts * drift_length > index_space_size:
        raise ValueError(
            "Total drift length exceeds the available space between the minimum index and the end of the stream."
        )
    return True


def determine_drift_points(
    dataframe_size: int,
    scenario: str,
    min_index: int,
    features_with_drifts: List[str] = None,
    total_drift_length: Optional[int] = None,
    drift_within_batch: float = 1.0,
    num_drifts: int = 2,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Determine the drift points for a synthetic dataset. The drift points occur only after the min_index.

    """
    if not features_with_drifts:
        print("No features with drifts specified.")
        return {}

    drift_points = {feature: [] for feature in features_with_drifts}

    if total_drift_length > dataframe_size or total_drift_length > (
        dataframe_size - min_index
    ):
        raise ValueError(
            "Total drift length exceeds dataframe size or minimum index constraint."
        )

    # Need space between the drifts - leave the drifts out of the first and last batch of the stream
    index_space_size = dataframe_size - 2 * min_index

    # Now match the drifts to the batch boundaries in a way that they are evenly split across the batches that are inside the index_space_size
    if "parallel" in scenario:
        drift_length = math.ceil(
            total_drift_length / (num_drifts * len(features_with_drifts))
        )
        spacing = (index_space_size - num_drifts * drift_length) // (num_drifts)

        if there_is_space_for_all_drifts(num_drifts, drift_length, index_space_size):
            for i in range(num_drifts):
                start_index = min_index + i * (drift_length + spacing)
                end_index = start_index + drift_length
                drift_start = start_index + int(
                    (end_index - start_index) * (1 - drift_within_batch)
                )
                drift_end = drift_start + drift_length
                for feature in features_with_drifts:
                    drift_points[feature].append((drift_start, drift_end))

    elif "switching" in scenario:
        # Drift length for each window per feature and per drift sequence
        drift_length = math.ceil(
            total_drift_length / (num_drifts * len(features_with_drifts))
        )

        # Calculate the available space for drift windows
        available_space = (
            index_space_size - num_drifts * len(features_with_drifts) * drift_length
        )
        # Ensure the available space is evenly distributed between drift windows
        spacing = available_space // (num_drifts * len(features_with_drifts))

        if there_is_space_for_all_drifts(
            num_drifts * len(features_with_drifts), drift_length, index_space_size
        ):
            for i in range(num_drifts):  # Loop over sequences of drifts
                for j, feature in enumerate(features_with_drifts):  # Loop over features
                    # Calculate the start and end index for each drift window
                    start_index = (
                        min_index
                        + i * (len(features_with_drifts) * (drift_length + spacing))
                        + j * (drift_length + spacing)
                    )
                    end_index = start_index + drift_length
                    drift_start = start_index + int(
                        (end_index - start_index) * (1 - drift_within_batch)
                    )
                    drift_end = drift_start + drift_length
                    drift_points[feature].append((drift_start, drift_end))

    return drift_points


def plot_accumulated_differences(
    accumulated_differences: pd.DataFrame,
    features_with_drifts: List[str],
    dataset_name: str,
    batch_size: int,
    detected_drifts: Optional[Dict[str, List[int]]] = None,
    drift_within_batch: float = 1.0,
):
    from codes.config import comparisons_output_dir as output_dir

    num_batches = len(accumulated_differences) // batch_size
    batch_indices = range(1, num_batches + 1)
    accumulated_differences["Batch"] = (accumulated_differences.index // batch_size) + 1

    # Calculate cumulative sum for each batch
    accumulated_differences_cumsum = accumulated_differences.groupby("Batch").cumsum()
    accumulated_differences_cumsum["Batch"] = accumulated_differences["Batch"]

    num_features = len(features_with_drifts)
    fig, axes = plt.subplots(
        num_features + 1, 1, figsize=(14, 7 * (num_features + 1)), sharex=True
    )

    # Plot the accumulated dataset difference
    axes[0].plot(
        batch_indices,
        accumulated_differences_cumsum.groupby("Batch").sum().sum(axis=1),
        label="Total Difference",
        color="black",
    )
    axes[0].set_ylabel("Total Accumulated Difference")
    axes[0].set_title("Total Accumulated Differences for Dataset")
    axes[0].legend()
    axes[0].grid(True)

    # Plot the accumulated differences for each feature
    for i, feature in enumerate(features_with_drifts, start=1):
        axes[i].plot(
            batch_indices,
            accumulated_differences_cumsum.groupby("Batch")[feature].sum(),
            label=feature,
        )
        axes[i].set_ylabel(f"Accumulated Difference")
        axes[i].set_title(f"Accumulated Differences for {feature}")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Batch Index")
    plt.tight_layout()

    # Save the first plot
    output_dir = os.path.join(output_dir, dataset_name, "feature_plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f"{dataset_name}_accumulated_differences.png",
    )
    plt.savefig(plot_path)
    plt.close()

    # Generate the second plot if detected_drifts is not None
    if detected_drifts is not None:
        # Create a new figure for the techniques' accumulated distances
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot the accumulated distances for each technique
        for technique, drift_batches in detected_drifts.items():
            # Reset cumulative sum when a drift is detected
            cumulative_sum = 0
            accumulated_distances = []
            for batch in batch_indices:
                if (
                    batch - 1 in drift_batches
                ):  # Reset cumulative sum if drift is detected
                    cumulative_sum = 0
                cumulative_sum += (
                    accumulated_differences_cumsum.groupby("Batch")
                    .sum()
                    .sum(axis=1)
                    .iloc[batch - 1]
                )
                accumulated_distances.append(cumulative_sum)
            ax.plot(batch_indices, accumulated_distances, label=technique)

        ax.set_xlabel("Batch Index")
        ax.set_ylabel("Accumulated Distance")
        ax.set_title("Accumulated Distances for Each Technique (Resetting at Drifts)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Save the second plot
        techniques_plot_path = os.path.join(
            output_dir,
            f"{dataset_name}_techniques_accumulated_differences_{batch_size}_{drift_within_batch}.png",
        )
        plt.savefig(techniques_plot_path)
        plt.close()

        # Generate the third plot: individual plots for each technique
        num_techniques = len(detected_drifts)
        fig, axes = plt.subplots(
            num_techniques, 1, figsize=(14, 7 * num_techniques), sharex=True
        )

        for i, (technique, drift_batches) in enumerate(detected_drifts.items()):
            # Reset cumulative sum when a drift is detected
            cumulative_sum = 0
            accumulated_distances = []
            for batch in batch_indices:
                if (
                    batch - 1 in drift_batches
                ):  # Reset cumulative sum if drift is detected
                    cumulative_sum = 0
                cumulative_sum += (
                    accumulated_differences_cumsum.groupby("Batch")
                    .sum()
                    .sum(axis=1)
                    .iloc[batch - 1]
                )
                accumulated_distances.append(cumulative_sum)
            axes[i].plot(batch_indices, accumulated_distances, label=technique)
            axes[i].set_ylabel("Accumulated Distance")
            axes[i].set_title(f"Accumulated Distances for {technique}")
            axes[i].legend()
            axes[i].grid(True)

        axes[-1].set_xlabel("Batch Index")
        plt.tight_layout()

        # Save the third plot
        individual_techniques_plot_path = os.path.join(
            output_dir,
            f"{dataset_name}_techniques_accumulated_differences_{batch_size}_{drift_within_batch}_split.png",
        )
        plt.savefig(individual_techniques_plot_path)
        plt.close()


# def create_synthetic_dataframe(
#     dataframe_size: int,
#     num_features: int = 5,
#     loc: float = 10,
#     scale: float = 1,
#     seed: int = 42,
# ) -> pd.DataFrame:
#     """Create a synthetic dataframe with a specified number of features and a class column."""
#     np.random.seed(seed)
#
#     # Generate features with the same loc and scale
#     features = {
#         f"feature{i+1}": np.random.normal(loc=loc, scale=scale, size=dataframe_size)
#         for i in range(num_features)
#     }
#
#     # Generate class labels based on the sum of all features
#     class_labels = np.where(sum(features.values()) > num_features * loc, 1, 0)
#
#     # Add class labels to the features dictionary
#     features["class"] = class_labels
#
#     return pd.DataFrame(features)
#
#
# def generate_synthetic_dataset(
#     dataframe_size: int,
#     num_features: int = 5,
#     loc: float = 10,
#     scale: float = 1,
#     seed: int = 42,
# ) -> pd.DataFrame:
#     """
#     Generate a synthetic dataframe.
#
#     Parameters:
#     dataframe_size (int): The size of the dataframe.
#     num_features (int): The number of features in the dataframe.
#     loc (float): The mean of the normal distribution for generating features.
#     scale (float): The standard deviation of the normal distribution for generating features.
#     seed (int): The random seed for reproducibility.
#
#     Returns:
#     pd.DataFrame: The generated synthetic dataframe.
#     """
#     np.random.seed(seed)
#     return create_synthetic_dataframe(dataframe_size, num_features, loc, scale, seed)

from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


def create_synthetic_dataframe(
    dataframe_size: int,
    num_features: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic dataframe using make_classification."""
    np.random.seed(seed)

    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=dataframe_size,
        n_features=num_features,
        n_informative=num_features - 1,  # Control number of informative features
        n_redundant=1,  # One redundant feature
        n_classes=2,  # Binary classification
        random_state=seed,
    )

    # Shift all values to be positive by adding a constant
    X_positive = X - np.min(X) + 1

    # Create feature names following the same convention
    feature_names = [f"feature{i + 1}" for i in range(num_features)]

    # Create dataframe
    df = pd.DataFrame(X_positive, columns=feature_names)
    df["class"] = y  # Add class label

    return df


def generate_synthetic_dataset(
    dataframe_size: int,
    num_features: int = 5,
    loc: float = 10,
    scale: float = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataframe.

    Parameters:
    dataframe_size (int): The size of the dataframe.
    num_features (int): The number of features in the dataframe.
    seed (int): The random seed for reproducibility.

    Returns:
    pd.DataFrame: The generated synthetic dataframe.
    """
    np.random.seed(seed)
    return create_synthetic_dataframe(dataframe_size, num_features, seed)


def determine_drift_points_wrapper(
    dataframe_size: int,
    scenario: str,
    features_with_drifts: List[str],
    drift_within_batch: float = 1.0,
    total_drift_length: int = 20000,
    min_index: int = 0,
    num_drifts: int = 2,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Determine drift points for the specified features.

    Parameters:
    dataframe_size (int): The size of the dataframe.
    scenario (str): The scenario type ('parallel' or 'switching').
    features_with_drifts (List[str]): The list of features to which drifts will be applied.
    drift_within_batch (float): The percentage of the drift that should be within a batch.
    total_drift_length (int): The total length of the drift.
    min_index (int): The minimum index to start the drift.

    Returns:
    Dict[str, List[Tuple[int, int]]]: A dictionary mapping features to their drift points.
    """
    return determine_drift_points(
        dataframe_size,
        scenario,
        drift_within_batch=drift_within_batch,
        total_drift_length=total_drift_length,
        min_index=min_index,
        features_with_drifts=features_with_drifts,
        num_drifts=num_drifts,
    )


def apply_drifts_to_dataframe(
    original_df: pd.DataFrame,
    features_with_drifts: List[str],
    drift_points: Dict[str, List[Tuple[int, int]]],
    scenario: str = None,
) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    """
    Apply abrupt drifts to the specified features in the dataframe.

    Parameters:
    original_df (pd.DataFrame): The original dataframe.
    features_with_drifts (List[str]): The list of features to which drifts will be applied.
    drift_points (Dict[str, List[Tuple[int, int]]]): A dictionary mapping features to their drift points.

    Returns:
    Tuple[pd.DataFrame, Dict[str, List[Any]]]: The drifted dataframe and drift information.
    """
    drifted_df = original_df.copy()
    drift_info = {}
    for feature in features_with_drifts:
        drift_info[feature] = []
        for start_index, end_index in drift_points[feature]:
            if "abrupt" in scenario:
                drifted_df = add_abrupt_drift(
                    drifted_df, feature, start_index, end_index, change=5
                )
                drift_info[feature].append(("abrupt", start_index, end_index))
            if "incremental" in scenario:
                drifted_df = add_incremental_drift(
                    drifted_df, feature, start_index, end_index, change=0.05, step=0.001
                )
                drift_info[feature].append(("incremental", start_index, end_index))
    return drifted_df, drift_info


def calculate_accumulated_differences(
    original_df: pd.DataFrame,
    drifted_df: pd.DataFrame,
    features_with_drifts: List[str],
    dataframe_size: int,
) -> pd.DataFrame:
    """
    Calculate the accumulated differences between the original and drifted dataframes.

    Parameters:
    original_df (pd.DataFrame): The original dataframe.
    drifted_df (pd.DataFrame): The drifted dataframe.
    features_with_drifts (List[str]): The list of features to which drifts were applied.
    dataframe_size (int): The size of the dataframe.

    Returns:
    pd.DataFrame: A dataframe containing the accumulated differences for each feature.
    """
    comparator = DataFrameComparator(original_df, drifted_df)
    accumulated_differences = pd.DataFrame()
    for feature in features_with_drifts:
        differences = comparator.measure_feature_difference(
            feature, 0, dataframe_size - 1
        )
        accumulated_differences[feature] = differences.cumsum()
    return accumulated_differences


def generate_synthetic_dataset_with_drifts(
    dataframe_size: int,
    features_with_drifts: List[str],
    batch_size: int,
    num_features: int = 5,
    loc: float = 10,
    scale: float = 1,
    seed: int = 42,
    scenario: str = "parallel",
    drift_within_batch: float = 1.0,
    num_drifts: int = 2,
) -> Tuple[
    pd.DataFrame,
    Dict[str, List[Tuple[int, int]]],
    Dict[str, List[Any]],
    pd.DataFrame,
    List[str],
]:
    """
    Generate a synthetic dataset with specified drifts.

    Parameters:
    dataframe_size (int): The size of the dataframe.
    features_with_drifts (List[str]): The list of features to which drifts will be applied.
    batch_size (int): The size of each batch.
    num_features (int): The number of features in the dataframe.
    loc (float): The mean of the normal distribution for generating features.
    scale (float): The standard deviation of the normal distribution for generating features.
    seed (int): The random seed for reproducibility.
    scenario (str): The scenario type ('parallel' or 'switching').
    drift_within_batch (float): The percentage of the drift that should be within a batch.

    Returns:
    Tuple[pd.DataFrame, Dict[str, List[Tuple[int, int]]], Dict[str, List[Any]], pd.DataFrame, List[str]]:
        The drifted dataframe, drift points, drift information, accumulated differences, and features with drifts.
    """
    # Generate the original synthetic dataframe
    original_df = generate_synthetic_dataset(
        dataframe_size, num_features, loc, scale, seed
    )

    # Determine drift points
    drift_points = determine_drift_points_wrapper(
        dataframe_size,
        scenario,
        features_with_drifts,
        drift_within_batch,
        min_index=2 * batch_size,  # start drifts at batch 3
        num_drifts=num_drifts,
    )

    # Apply drifts to the dataframe
    drifted_df, drift_info = apply_drifts_to_dataframe(
        original_df, features_with_drifts, drift_points, scenario=scenario
    )

    # Calculate accumulated differences
    accumulated_differences = calculate_accumulated_differences(
        original_df, drifted_df, features_with_drifts, dataframe_size
    )

    return (
        drifted_df,
        drift_points,
        drift_info,
        accumulated_differences,
        features_with_drifts,
    )


def save_synthetic_dataset(df: pd.DataFrame, dataset_name: str):
    """Save the synthetic dataset to a CSV file."""
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    csv_file_path = os.path.join(dataset_output_dir, f"{dataset_name}.csv")
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    plot_data_disturbance()
