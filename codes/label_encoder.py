# label_encoder.py
from sklearn.preprocessing import LabelEncoder
import numpy as np


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Encode class labels."""
    encoder = LabelEncoder()
    return encoder.fit_transform(labels)
