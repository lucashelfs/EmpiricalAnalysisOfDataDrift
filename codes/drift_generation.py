import os
import random
from typing import Tuple, List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codes.config import comparisons_output_dir as output_dir

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

    if scenario == "parallel":
        space_between_drifts = (dataframe_size - total_drift_length) // (num_drifts + 1)
        current_start = min_index + space_between_drifts

        for drift_num in range(num_drifts):
            start_index = current_start
            end_index = min(start_index + drift_length, dataframe_size - 1)
            for i in range(num_features):
                if f"feature{i + 1}" not in drift_points:
                    drift_points[f"feature{i + 1}"] = []
                drift_points[f"feature{i + 1}"].append((start_index, end_index))
            current_start = end_index + space_between_drifts

    elif scenario == "switching":
        period = dataframe_size // num_features
        for i in range(num_features):
            start_index = max(i * period, min_index)
            end_index = min(start_index + drift_length, dataframe_size - 1)
            drift_points[f"feature{i + 1}"] = [(start_index, end_index)]

    return drift_points


def generate_synthetic_dataset_with_drifts(
    dataframe_size: int,
    features_with_drifts: List[str],
    batch_size: int,
    num_features: int = 5,
    loc: float = 10,
    scale: float = 1,
    seed: int = 42,
    scenario: str = "parallel",
) -> Tuple[pd.DataFrame, List[int], Dict[str, List[Tuple[str, int, int]]]]:
    """
    Generate a synthetic dataset and add disturbances to specified features.


    Parameters:
    dataframe_size (int): The size of the dataframe.
    features_with_drifts (List[str]): The list of features to which drifts will be added.
    batch_size (int): The batch size used to determine the minimum index for drift occurence.
    num_features (int): The number of features in the dataframe (default is 5).
    loc (float): The mean of the normal distribution used to generate the features (default is 10).
    scale (float): The standard deviation of the normal distribution used to generate the features (default is 1).
    seed (int): The random seed for reproducibility (default is 42).
    scenario (str): The scenario type ('parallel' or 'switching').

    Returns:
    Tuple[pd.DataFrame, List[int], Dict[str, List[Tuple[str, int, int]]]]: A tuple containing the dataframe with drifts,
                                                                          a list of all drift points, and a dictionary
                                                                          with drift information for each feature.
    """
    np.random.seed(seed)
    random.seed(seed)
    df = create_synthetic_dataframe(dataframe_size, num_features, loc, scale, seed)

    min_index = 2 * batch_size
    drift_points = determine_drift_points(
        dataframe_size, num_features, scenario, min_index, drift_length=5 * batch_size
    )
    all_drift_points = []
    drift_info = {feature: [] for feature in features_with_drifts}

    drift_types = ["abrupt", "gradual", "incremental"]

    for i, feature in enumerate(features_with_drifts):
        if feature in drift_points:
            for start_index, end_index in drift_points[feature]:
                std_dev = df[feature].std()
                change = std_dev * 0.5
                step = change / (dataframe_size // 4)

                drift_type = drift_types[i % len(drift_types)]

                if drift_type == "abrupt":
                    df = add_abrupt_drift(df, feature, start_index, end_index, change)
                elif drift_type == "gradual":
                    df = add_gradual_drift(df, feature, start_index, end_index, change)
                elif drift_type == "incremental":
                    df = add_incremental_drift(
                        df, feature, start_index, end_index, change, step
                    )

                all_drift_points.extend([start_index, end_index])
                drift_info[feature].append((drift_type, start_index, end_index))

    return df, all_drift_points, drift_info


def save_synthetic_dataset(df: pd.DataFrame, dataset_name: str):
    """Save the synthetic dataset to a CSV file."""
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    csv_file_path = os.path.join(dataset_output_dir, f"{dataset_name}.csv")
    df.to_csv(csv_file_path, index=False)
    print(f"Synthetic dataset saved to {csv_file_path}")


if __name__ == "__main__":
    plot_data_disturbance()
