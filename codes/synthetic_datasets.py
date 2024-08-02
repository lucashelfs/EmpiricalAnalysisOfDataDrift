import numpy as np
import pandas as pd


def generate_feature_stream():
    random_state = np.random.RandomState(seed=42)
    dist_a = random_state.normal(0.8, 0.05, 1000)
    dist_b = random_state.normal(0.4, 0.02, 1000)
    dist_c = random_state.normal(0.6, 0.1, 1000)

    # Concatenate data to simulate a data stream with 2 drifts
    stream = np.concatenate((dist_a, dist_b, dist_c))
    return stream


def synthetic_dataframe_creator(parameters: dict = None) -> pd.DataFrame:
    """Create synthetic dataframe based on given parameters."""
    drift_type = parameters.get("drift_type", None)
    if not drift_type:
        raise ValueError("Parameter drift_type must be specified.")

    return pd.DataFrame()
