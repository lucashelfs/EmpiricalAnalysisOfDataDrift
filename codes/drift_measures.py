import pandas as pd


class DataFrameComparator:
    def __init__(self, original_df: pd.DataFrame, drifted_df: pd.DataFrame):
        self.original_df = original_df
        self.drifted_df = drifted_df

    def measure_feature_difference(
        self, feature: str, start_index: int, end_index: int
    ) -> pd.Series:
        """
        Measure the difference for a given feature between the original and drifted dataframes over a specified index range.

        Parameters:
        feature (str): The name of the feature to compare.
        start_index (int): The start index of the range.
        end_index (int): The end index of the range.

        Returns:
        pd.Series: The difference between the original and drifted feature values over the specified range.
        """
        original_series = self.original_df.loc[start_index:end_index, feature]
        drifted_series = self.drifted_df.loc[start_index:end_index, feature]
        return drifted_series - original_series

    def measure_dataset_difference(
        self, start_index: int, end_index: int
    ) -> pd.DataFrame:
        """
        Measure the difference for all features between the original and drifted dataframes over a specified index range.

        Parameters:
        start_index (int): The start index of the range.
        end_index (int): The end index of the range.

        Returns:
        pd.DataFrame: The difference between the original and drifted feature values for all features over the specified range.
        """
        differences = {}
        for feature in self.original_df.columns:
            differences[feature] = self.measure_feature_difference(
                feature, start_index, end_index
            )
        return pd.DataFrame(differences)
