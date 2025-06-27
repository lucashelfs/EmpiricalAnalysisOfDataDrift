import pandas as pd
from typing import Tuple

from codes.config import insects_datasets
from drift_info import extract_drift_info
from codes.common import (
    calculate_index,
    load_and_prepare_dataset,
    common_datasets,
    datasets_with_added_drifts,
)


def load_csv_dataset(
    file_path: str, class_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_path)
    Y_og = df.pop(class_column)
    return df, Y_og


def fetch_dataset_change_points(dataset_name: str, batch_size: int):
    """Fetch all change points from dataset."""
    change_points = []

    if dataset_name in insects_datasets.keys():
        change_points = insects_datasets[dataset_name]["change_point"]

    if dataset_name in common_datasets.keys():
        change_points = common_datasets[dataset_name]["change_point"]

    if dataset_name in datasets_with_added_drifts:
        df, _, _ = load_and_prepare_dataset(dataset_name)
        _, column, drifts = extract_drift_info(dataset_name)
        for drift_type in drifts:
            for drift in drifts[drift_type]:
                start_index = calculate_index(df, drift[0])
                end_index = calculate_index(df, drift[1])
                change_points.extend([start_index, end_index])

    batches_with_change_points = [cp // batch_size for cp in change_points]
    return batches_with_change_points
