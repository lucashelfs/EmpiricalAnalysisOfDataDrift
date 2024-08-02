"""
Simple KSDDM implementation.
"""

import copy
import pandas as pd
from scipy import stats


class KSDDM:
    def __init__(self, mean_threshold=0.05) -> None:
        """
        Initialize the KSDDM class.

        Parameters:
        mean_threshold (float): The threshold for the mean p-value to detect drift.
        """
        self.drift_state = []
        self.all_p_values = []
        self.current_batch = None
        self.batches_since_reset = 0
        self.batches = 0
        self.p_value_diff = 0
        self.mean_threshold = mean_threshold
        self.reference = None
        self.p_values = pd.Series()
        self.p_values_diff = []
        self.last_p_value_series = None

    def set_reference(self, reference):
        """
        Set the reference batch for KSDDM.

        Parameters:
        reference (pd.DataFrame): The reference batch data.
        """
        self.reference = copy.deepcopy(reference)
        self.batches += 1

    def calculate_ks_distance(self):
        """
        Calculate the Kolmogorov-Smirnov distance for each feature between the reference and current batch.
        """
        feature_p_values = []

        for feature in self.reference.columns:
            test_stat = stats.ks_2samp(
                self.reference[feature], self.current_batch[feature]
            ).pvalue
            feature_p_values.append(test_stat)

        self.p_values = pd.Series(feature_p_values)
        self.all_p_values.append(self.p_values)

        if self.last_p_value_series is not None:
            self.p_value_diff = abs(self.p_values - self.last_p_value_series)
            self.p_values_diff.append(self.p_value_diff)
            self.last_p_value_series = self.p_values
        else:
            self.p_value_diff = abs(self.p_values)
            self.last_p_value_series = self.p_values

    def update(self, batch):
        """
        Update the KSDDM with a new batch of data.

        Parameters:
        batch (pd.DataFrame): The new batch of data.
        """
        self.batches += 1
        self.batches_since_reset += 1
        self.current_batch = batch

        if self.batches_since_reset >= 1:
            self.calculate_ks_distance()
            self.determine_drift()
        else:
            self.drift_state.append("")

    def determine_drift(self):
        """
        Determine if drift has occurred based on the p-values.
        """
        alpha = self.mean_threshold
        minimum = self.p_values.min()
        threshold = alpha / len(self.reference.columns)
        if minimum < threshold:
            self.reference = self.current_batch
            self.batches_since_reset = 0
            self.drift_state.append("drift")
        else:
            self.reference = pd.concat([self.reference, self.current_batch])
            self.drift_state.append(None)


def initialize_ksddm(reference, mean_threshold=0.05):
    """
    Initialize a KSDDM instance with a reference batch.

    Parameters:
    reference (pd.DataFrame): The reference batch data.
    mean_threshold (float): The threshold for the mean p-value to detect drift.

    Returns:
    KSDDM: An initialized KSDDM instance.
    """
    ksddm = KSDDM(mean_threshold=mean_threshold)
    ksddm.set_reference(reference)
    return ksddm


def process_batches(ksddm, X, reference_batch):
    """
    Process batches of data using KSDDM.

    Parameters:
    ksddm (KSDDM): The KSDDM instance.
    X (pd.DataFrame): The dataset containing batches.
    reference_batch (int): The batch number to be used as the reference.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Heatmap data, difference heatmap data, and detected drift states.
    """
    batches = X[X.Batch != reference_batch].Batch.unique()
    heatmap_data = pd.DataFrame(columns=batches)
    diff_heatmap_data = pd.DataFrame(columns=batches)
    detected_drift = []

    for batch, subset_data in X[X.Batch != reference_batch].groupby("Batch"):
        ksddm.update(subset_data.iloc[:, :-1])
        heatmap_data[batch] = ksddm.p_values
        diff_heatmap_data[batch] = ksddm.p_value_diff
        detected_drift.append(ksddm.drift_state)

    return heatmap_data, diff_heatmap_data, detected_drift
