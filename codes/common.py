import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from codes.config import comparisons_output_dir as output_dir

from typing import Tuple

from codes.config import insects_datasets, load_insect_dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

from codes.config import project_path_root
from codes.drift_config import drift_config
from codes.drift_generation import (
    apply_abrupt_drift,
    apply_gradual_drift,
    apply_incremental_drift,
    calculate_index,
)

from codes.river_datasets import (
    sea_concept_drift_dataset,
    stagger_concept_drift_dataset,
    generate_dataset_from_river_generator,
    multi_sea_dataset,
    multi_stagger_dataset,
)
from codes.river_config import seed, drift_central_position, drift_width, dataset_size

common_datasets_file_path_prefix = os.path.join(project_path_root, "data/")

common_datasets = {
    "electricity": {
        "filename": common_datasets_file_path_prefix + "electricity-normalized.csv",
        "class_column": "class",
        "change_point": [],
    },
    "magic": {
        "filename": common_datasets_file_path_prefix + "magic.csv",
        "class_column": "class",
        "change_point": [],
    },
    "SEA": {
        "change_point": [
            drift_central_position - drift_width // 2,
            drift_central_position + drift_width // 2,
        ]
    },
    "STAGGER": {
        "change_point": [
            drift_central_position - drift_width // 2,
            drift_central_position + drift_width // 2,
        ]
    },
    "MULTISEA": {"change_point": [(i + 1) * dataset_size // 4 for i in range(3)]},
    "MULTISTAGGER": {"change_point": [(i + 1) * dataset_size // 3 for i in range(2)]},
}

datasets_with_added_drifts = [
    f"{outer_key}_{inner_key}"
    for outer_key, inner_dict in drift_config.items()
    for inner_key in inner_dict.keys()
]


def load_magic_dataset_data(file_path=common_datasets_file_path_prefix + "magic.csv"):
    file_path = os.path.expanduser(file_path)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    magic_gamma_telescope = fetch_ucirepo(id=159)
    df = pd.DataFrame(
        magic_gamma_telescope.data.features,
        columns=magic_gamma_telescope.data.feature_names,
    )
    df["fConc1"] = df["fConc1"].sort_values().reset_index(drop=True)

    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    scaled_df.to_csv(file_path, index=False)
    return scaled_df


def load_magic_dataset_targets():
    file_path = os.path.expanduser(
        common_datasets_file_path_prefix + "magic_targets.csv"
    )
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    return fetch_ucirepo(id=159).data.targets


def load_synthetic_sea(seed, drift_central_position, drift_width, dataset_size):
    sea_generator = sea_concept_drift_dataset(
        seed, drift_central_position, drift_width, stream_variant=0, drift_variant=1
    )
    sea_df = generate_dataset_from_river_generator(sea_generator, dataset_size)
    return sea_df


def load_multi_sea(seed, dataset_size):
    msg = multi_sea_dataset(seed)
    multi_sea_df = generate_dataset_from_river_generator(msg, dataset_size)
    return multi_sea_df


def load_synthetic_stagger(seed, drift_central_position, drift_width, dataset_size):
    stagger_generator = stagger_concept_drift_dataset(
        seed, drift_central_position, drift_width, stream_variant=0, drift_variant=1
    )
    stagger_df = generate_dataset_from_river_generator(stagger_generator, dataset_size)
    return stagger_df


def load_multi_stagger(seed, dataset_size):
    msg = multi_stagger_dataset(seed)
    multi_stagger_df = generate_dataset_from_river_generator(msg, dataset_size)
    return multi_stagger_df


#  Tested methods below


def find_indexes(drifts_list: list) -> list:
    """Fetch the indexes where drifts occur."""
    return [index for index, value in enumerate(drifts_list) if value == "drift"]


def define_batches(X: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Define batches on dataframe based on batch size."""
    X["Batch"] = (X.index // batch_size) + 1
    return X


def load_and_prepare_dataset(
    dataset: str,
) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """Load the desired dataset."""

    if dataset in insects_datasets:
        dataset_filename_str = (
            dataset.lower()
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
        )
        df = load_insect_dataset(insects_datasets[dataset]["filename"])
        Y_og = df.pop("class")
        dataset_filename_str = dataset_filename_str

    if dataset == "electricity":
        df = pd.read_csv(common_datasets[dataset]["filename"])
        Y_og = df.pop("class")
        dataset_filename_str = "electricity"
        # return df, dataset

    if dataset == "magic":
        df = load_magic_dataset_data()
        Y_og = load_magic_dataset_targets().values.ravel()
        dataset_filename_str = "magic"

    if dataset == "SEA":
        df = load_synthetic_sea(seed, drift_central_position, drift_width, dataset_size)
        Y_og = df.pop("class")
        dataset_filename_str = "SEA"

    if dataset == "MULTISEA":
        df = load_multi_sea(seed, dataset_size)
        Y_og = df.pop("class")
        dataset_filename_str = "MULTISEA"

    if dataset == "STAGGER":
        df = load_synthetic_stagger(
            seed, drift_central_position, drift_width, dataset_size
        )
        Y_og = df.pop("class")
        dataset_filename_str = "STAGGER"

    if dataset == "MULTISTAGGER":
        df = load_multi_stagger(seed, dataset_size)
        Y_og = df.pop("class")
        dataset_filename_str = "MULTISTAGGER"

    if dataset in datasets_with_added_drifts:
        return load_and_prepare_dataset_with_drifts(dataset)

    le = LabelEncoder()
    Y = le.fit_transform(Y_og)

    return df, Y, dataset_filename_str


def extract_drift_info(dataset_id_with_scenario: str):
    """Extract drift information from the drift config."""
    dataset_id_with_scenario = dataset_id_with_scenario.split("_")
    dataset = dataset_id_with_scenario[0]
    scenario = "_".join(dataset_id_with_scenario[1:])

    if not drift_config.get(dataset, False):
        raise ValueError("Dataset not found in drift config file.")

    column = drift_config[dataset][scenario]["column"]
    drifts = drift_config[dataset][scenario]["drifts"]
    return dataset, column, drifts


def plot_data_streams_with_drifts(
    df_original, df_drifted, column, output_dir, dataset_name
):
    """Plot data streams before and after applying drifts."""
    # Plot the original data
    plt.figure(figsize=(14, 7))
    plt.plot(df_original.index, df_original[column], label="Original", color="blue")
    plt.plot(df_drifted.index, df_drifted[column], label="With Drifts", color="red")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.title(f"{dataset_name} - {column} Before and After Drifts")
    plt.legend()
    plt.grid(True)

    # Ensure the output directory exists
    dataset_output_dir = os.path.join(output_dir, dataset_name, "distributions")
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(dataset_output_dir, f"{dataset_name}_{column}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")


def apply_drifts(
    df: pd.DataFrame, column: str, drifts: dict, output_dir: str, dataset_name: str
) -> pd.DataFrame:
    """Apply drifts to the dataframe based on the drift information and save plots."""
    df_original = df.copy()
    for drift_type in drifts:
        if drift_type == "abrupt":
            for drift in drifts[drift_type]:
                start_index = calculate_index(df, drift[0])
                end_index = calculate_index(df, drift[1])
                change = df[column].mean() * 2
                df = apply_abrupt_drift(
                    df.copy(), column, start_index, end_index, change
                )
        elif drift_type == "gradual":
            for drift in drifts[drift_type]:
                start_index = calculate_index(df, drift[0])
                end_index = calculate_index(df, drift[1])
                change = df[column].mean() * 2
                df = apply_gradual_drift(
                    df.copy(), column, start_index, end_index, change
                )
        elif drift_type == "incremental":
            for drift in drifts[drift_type]:
                start_index = calculate_index(df, drift[0])
                end_index = calculate_index(df, drift[1])
                change = df[column].mean() * 2
                step = change // (drift[1] - drift[0])
                df = apply_incremental_drift(
                    df.copy(), column, start_index, end_index, change, step=step
                )
        else:
            raise ValueError("Invalid drift type.")

    plot_data_streams_with_drifts(df_original, df, column, output_dir, dataset_name)
    return df


def load_and_prepare_dataset_with_drifts(
    dataset_id_with_scenario: str,
) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """Load and prepare a dataset with drifts."""
    dataset, column, drifts = extract_drift_info(dataset_id_with_scenario)
    df, Y, dataset_filename_str = load_and_prepare_dataset(dataset)
    df = apply_drifts(df, column, drifts, output_dir, dataset_id_with_scenario)
    return df, Y, dataset_filename_str + "_" + dataset_id_with_scenario.split("_", 1)[1]
