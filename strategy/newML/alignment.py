import numpy as np
from typing import Dict, Tuple

class RegimeLabelAligner:
    """
    Stabilizes unsupervised clustering labels by mapping them to a deterministic
    semantic order. E.g., Regime 0 is ALWAYS 'Low Volatility'.
    """

    def __init__(self, sort_feature_idx: int, reverse: bool = False):
        self.sort_feature_idx = sort_feature_idx
        self.reverse = reverse
        self.mapping: Dict[int, int] = {}
        self.is_fitted = False

    def fit(self, X: np.ndarray, raw_labels: np.ndarray):
        unique_labels = np.unique(raw_labels)
        cluster_scores = []
        
        for label in unique_labels:
            mask = (raw_labels == label)
            # Calculate centroid of the target feature
            score = np.mean(X[mask, self.sort_feature_idx])
            cluster_scores.append((label, score))

        # Sort clusters based on score (e.g., Volatility)
        cluster_scores.sort(key=lambda x: x[1], reverse=self.reverse)

        self.mapping = {}
        for new_idx, (raw_lbl, _) in enumerate(cluster_scores):
            self.mapping[raw_lbl] = new_idx

        self.is_fitted = True
        return self

    def transform(self, raw_labels: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Aligner not fitted")
        
        # Vectorized mapping
        max_lbl = max(self.mapping.keys()) if self.mapping else 0
        lookup = np.zeros(max_lbl + 1, dtype=int)
        for r, n in self.mapping.items():
            lookup[r] = n
            
        return lookup[raw_labels]