import os
from typing import Tuple

import numpy as np
import pandas as pd

from codes.config import insects_datasets, load_insect_dataset
from codes.config import comparisons_output_dir as output_dir
from codes.utils import encode_labels, load_csv_dataset
from codes.common import (
    load_magic_dataset_data,
    load_magic_dataset_targets,
    load_synthetic_sea,
    load_multi_sea,
    load_synthetic_stagger,
    load_multi_stagger,
    datasets_with_added_drifts,
    common_datasets,
    load_and_prepare_dataset_with_drifts,
)
from codes.river_config import drift_central_position, drift_width, dataset_size, seed


def load_and_prepare_dataset(dataset: str) -> Tuple[pd.DataFrame, np.ndarray, str]:
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

    elif dataset == "electricity":
        df, Y_og = load_csv_dataset(common_datasets[dataset]["filename"], "class")
        dataset_filename_str = "electricity"

    elif dataset == "magic":
        df = load_magic_dataset_data()
        Y_og = load_magic_dataset_targets().values.ravel()
        dataset_filename_str = "magic"

    elif dataset == "SEA":
        df = load_synthetic_sea(seed, drift_central_position, drift_width, dataset_size)
        Y_og = df.pop("class")
        dataset_filename_str = "SEA"

    elif dataset == "MULTISEA":
        df = load_multi_sea(seed, dataset_size)
        Y_og = df.pop("class")
        dataset_filename_str = "MULTISEA"

    elif dataset == "STAGGER":
        df = load_synthetic_stagger(
            seed, drift_central_position, drift_width, dataset_size
        )
        Y_og = df.pop("class")
        dataset_filename_str = "STAGGER"

    elif dataset == "MULTISTAGGER":
        df = load_multi_stagger(seed, dataset_size)
        Y_og = df.pop("class")
        dataset_filename_str = "MULTISTAGGER"

    elif dataset in datasets_with_added_drifts:
        return load_and_prepare_dataset_with_drifts(dataset)

    elif dataset == "synthetic_dataset_no_drifts":
        df, Y_og = load_csv_dataset(
            os.path.join(
                output_dir,
                "synthetic_dataset_no_drifts",
                "synthetic_dataset_no_drifts.csv",
            ),
            "class",
        )
        dataset_filename_str = "synthetic_dataset_no_drifts"

    elif dataset == "synthetic_dataset_with_drifts":
        df, Y_og = load_csv_dataset(
            os.path.join(
                output_dir,
                "synthetic_dataset_with_drifts",
                "synthetic_dataset_with_drifts.csv",
            ),
            "class",
        )
        dataset_filename_str = "synthetic_dataset_with_drifts"

    elif dataset == "synthetic_dataset_with_parallel_drifts":
        df, Y_og = load_csv_dataset(
            os.path.join(
                output_dir,
                "synthetic_dataset_with_parallel_drifts",
                "synthetic_dataset_with_parallel_drifts.csv",
            ),
            "class",
        )
        dataset_filename_str = "synthetic_dataset_with_parallel_drifts"

    elif dataset == "synthetic_dataset_with_switching_drifts":
        df, Y_og = load_csv_dataset(
            os.path.join(
                output_dir,
                "synthetic_dataset_with_switching_drifts",
                "synthetic_dataset_with_switching_drifts.csv",
            ),
            "class",
        )
        dataset_filename_str = "synthetic_dataset_with_switching_drifts"

    else:
        raise ValueError(f"Dataset {dataset} not recognized.")

    Y = encode_labels(Y_og)
    return df, Y, dataset_filename_str
