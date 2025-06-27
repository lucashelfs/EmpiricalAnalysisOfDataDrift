# drift_info.py
from typing import Dict, List, Tuple
from codes.drift_config import drift_config


def extract_drift_info(
    dataset_id_with_scenario: str,
) -> Tuple[str, str, Dict[str, List[Tuple[str, int, int]]]]:
    """Extract drift information from the drift config."""
    dataset_id_with_scenario = dataset_id_with_scenario.split("_")
    dataset = dataset_id_with_scenario[0]
    scenario = "_".join(dataset_id_with_scenario[1:])

    if not drift_config.get(dataset, False):
        raise ValueError("Dataset not found in drift config file.")

    column = drift_config[dataset][scenario]["column"]
    drifts = drift_config[dataset][scenario]["drifts"]
    return dataset, column, drifts
