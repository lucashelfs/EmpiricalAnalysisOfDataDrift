import os
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

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


def create_synthetic_dataframe(
    dataframe_size: int,
    num_features: int = 5,
    loc: float = 10,
    scale: float = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic dataframe with a specified number of features and a class column."""
    np.random.seed(seed)

    # Generate features with the same loc and scale
    features = {
        f"feature{i+1}": np.random.normal(loc=loc, scale=scale, size=dataframe_size)
        for i in range(num_features)
    }

    # Generate class labels based on the sum of all features
    class_labels = np.where(sum(features.values()) > num_features * loc, 1, 0)

    # Add class labels to the features dictionary
    features["class"] = class_labels

    return pd.DataFrame(features)


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


def plot_data_disturbance():
    """
    Plot data disturbance by applying different types of drifts to a DataFrame.

    This function creates a simple DataFrame and applies sudden, gradual, and incremental drifts
    to one of its columns. It then plots the original and drifted data for comparison.

    The function performs the following steps:
    1. Creates a simple DataFrame with random data.
    2. Applies a abrupt drift to the 'feature1' column.
    3. Applies a gradual drift to the 'feature1' column.
    4. Applies an incremental drift to the 'feature1' column.
    5. Plots the original data and the data with each type of drift in separate subplots.

    Parameters:
    None

    Returns:
    None
    """
    DF_SIZE = 3000
    df = create_simple_dataframe(dataframe_size=DF_SIZE)

    # Apply gradual drift
    df_gradual = df.copy()
    df_gradual = add_gradual_drift(
        df_gradual, "feature1", start_index=0, end_index=(DF_SIZE // 4), max_change=5
    )

    # Apply incremental drift
    df_incremental = df.copy()
    df_incremental = add_incremental_drift(
        df_incremental,
        "feature1",
        start_index=(DF_SIZE // 4),
        end_index=(DF_SIZE // 2),
        change=0.1,
        step=0.1,
    )

    # Apply sudden drift
    df_sudden = df.copy()
    df_sudden = add_abrupt_drift(
        df_sudden,
        "feature1",
        start_index=(DF_SIZE // 2),
        end_index=(DF_SIZE // 2 + DF_SIZE // 4),
        change=5,
    )

    # Plot the original and drifted data
    plt.figure(figsize=(12, 8))

    # Original data
    plt.subplot(4, 1, 1)
    plt.plot(df["feature1"], label="Original")
    plt.title("Original Data")
    plt.xlabel("Index")
    plt.ylabel("feature1")
    plt.legend()

    # Sudden drift
    plt.subplot(4, 1, 2)
    plt.plot(df_sudden["feature1"], label="Sudden Drift", color="orange")
    plt.title("Sudden Drift")
    plt.xlabel("Index")
    plt.ylabel("feature1")
    plt.legend()

    # Gradual drift
    plt.subplot(4, 1, 3)
    plt.plot(df_gradual["feature1"], label="Gradual Drift", color="green")
    plt.title("Gradual Drift")
    plt.xlabel("Index")
    plt.ylabel("feature1")
    plt.legend()

    # Incremental drift
    plt.subplot(4, 1, 4)
    plt.plot(df_incremental["feature1"], label="Incremental Drift", color="red")
    plt.title("Incremental Drift")
    plt.xlabel("Index")
    plt.ylabel("feature1")
    plt.legend()

    plt.tight_layout()
    plt.show()


def determine_drift_points(
    dataframe_size: int,
    num_features: int,
    scenario: str,
    min_index: int,
    num_drifts: int = 1,
    drift_length: Optional[int] = None,
    batch_size: int = 1000,
    drift_within_batch: float = 1.0,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Determine the drift points for a synthetic dataset. The drift points occur only after the min_index.

    Parameters:
    dataframe_size (int): The size of the dataframe.
    num_features (int): The number of features in the dataframe.
    scenario (str): The scenario type ('parallel' or 'switching').
    min_index (int): The minimum index where drifts can start.
    num_drifts (int): The number of parallel drifts to generate (default is 1).
    drift_length (Optional[int]): The length of each drift. If not provided, it will be calculated dynamically.
    batch_size (int): The size of each batch.
    drift_within_batch (float): The percentage of the drift that should be within a batch (default is 1.0).

    Returns:
    Dict[str, List[Tuple[int, int]]]: A dictionary where keys are feature names and values are lists of tuples
                                      representing the start and end indexes of the drifts.
    """
    drift_points = {}
    if drift_length is None:
        drift_length = (dataframe_size - min_index) // (4 * num_drifts)

    total_drift_length = num_drifts * drift_length
    if total_drift_length > dataframe_size or total_drift_length > (
        dataframe_size - min_index
    ):
        raise ValueError(
            "Total drift length exceeds dataframe size or minimum index constraint."
        )

    # Calculate batch boundaries
    batch_boundaries = [
        (i, min(i + batch_size, dataframe_size))
        for i in range(min_index, dataframe_size, batch_size)
    ]

    if scenario == "parallel":
        for i in range(num_drifts):
            start_index, end_index = batch_boundaries[i]
            drift_start = start_index + int(
                (end_index - start_index) * (1 - drift_within_batch)
            )
            drift_end = drift_start + drift_length
            for j in range(num_features):
                if f"feature{j + 1}" not in drift_points:
                    drift_points[f"feature{j + 1}"] = []
                drift_points[f"feature{j + 1}"].append((drift_start, drift_end))

    elif scenario == "switching":
        for i in range(num_features):
            start_index, end_index = batch_boundaries[i % len(batch_boundaries)]
            drift_start = start_index + int(
                (end_index - start_index) * (1 - drift_within_batch)
            )
            drift_end = drift_start + drift_length
            if drift_end > end_index:
                # If drift end exceeds the current batch, extend to the next batch
                next_batch_start, next_batch_end = batch_boundaries[
                    (i + 1) % len(batch_boundaries)
                ]
                drift_end = min(drift_end, next_batch_end)
            drift_points[f"feature{i + 1}"] = [(drift_start, drift_end)]

    return drift_points


def plot_accumulated_differences(
    accumulated_differences: pd.DataFrame,
    features_with_drifts: List[str],
    dataset_name: str,
    batch_size: int,
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

    # Save the plot
    output_dir = os.path.join(output_dir, dataset_name, "feature_plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{dataset_name}_accumulated_differences.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Accumulated differences plot saved to {plot_path}")


def generate_synthetic_dataset_with_drifts(
    dataframe_size: int,
    features_with_drifts: List[str],
    batch_size: int,
    num_features: int = 5,
    loc: float = 10,
    scale: float = 1,
    seed: int = 42,
    scenario: str = "parallel",
) -> Tuple[
    pd.DataFrame,
    List[int],
    Dict[str, List[Tuple[str, int, int]]],
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
    loc (float): The mean of the normal distribution for generating data.
    scale (float): The standard deviation of the normal distribution for generating data.
    seed (int): The random seed for reproducibility.
    scenario (str): The scenario type ('parallel' or 'switching').

    Returns:
    Tuple[pd.DataFrame, List[int], Dict[str, List[Tuple[str, int, int]]], pd.DataFrame, List[str]]: The synthetic dataframe with drifts,
                                                                                                    the list of drift points,
                                                                                                    the drift information,
                                                                                                    the accumulated differences,
                                                                                                    and the features with drifts.
    """
    np.random.seed(seed)

    # Generate the original synthetic dataframe
    original_df = create_synthetic_dataframe(
        dataframe_size, num_features, loc, scale, seed
    )

    # Determine drift points
    drift_points = determine_drift_points(
        dataframe_size,
        num_features,
        scenario,
        drift_within_batch=0.05,  # this is for sliding the drift
        batch_size=batch_size,
        drift_length=batch_size * 4,
        min_index=batch_size,  # this assures that the first batch is stable
        num_drifts=len(features_with_drifts),
    )

    # Apply abrupt drifts to the dataframe
    drifted_df = original_df.copy()
    drift_info = {}
    for feature in features_with_drifts:
        drift_info[feature] = []
        for start_index, end_index in drift_points[feature]:
            drifted_df = add_abrupt_drift(
                drifted_df, feature, start_index, end_index, change=5
            )
            drift_info[feature].append(("abrupt", start_index, end_index))

    # Use DataFrameComparator to measure differences
    comparator = DataFrameComparator(original_df, drifted_df)
    accumulated_differences = pd.DataFrame()
    for feature in features_with_drifts:
        differences = comparator.measure_feature_difference(
            feature, 0, dataframe_size - 1
        )
        accumulated_differences[feature] = differences.cumsum()

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
    print(f"Synthetic dataset saved to {csv_file_path}")


if __name__ == "__main__":
    plot_data_disturbance()
