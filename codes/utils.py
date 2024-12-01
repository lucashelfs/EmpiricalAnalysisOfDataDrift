import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder


def load_csv_dataset(
    file_path: str, class_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(file_path)
    Y_og = df.pop(class_column)
    return df, Y_og


def encode_labels(Y_og: pd.Series) -> np.ndarray:
    le = LabelEncoder()
    return le.fit_transform(Y_og)
