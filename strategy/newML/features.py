import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from numba import jit
from dataclasses import dataclass

@dataclass
class WindowConfig:
    window_size: int = 64
    scale_method: str = "robust"  # 'robust' (median/iqr) or 'log'

# --- Numba Optimized Calculations ---

@jit(nopython=True)
def _calc_log_returns(prices):
    """Calculates log returns: ln(p_t / p_{t-1})."""
    res = np.empty(len(prices) - 1)
    for i in range(1, len(prices)):
        res[i-1] = np.log(prices[i] / prices[i-1])
    return res

@jit(nopython=True)
def _calc_parkinson_vol(highs, lows):
    """Parkinson Volatility estimator (high/low based)."""
    n = len(highs)
    sum_sq = 0.0
    for i in range(n):
        hl_ratio = np.log(highs[i] / lows[i])
        sum_sq += hl_ratio * hl_ratio
    return np.sqrt(1.0 / (4.0 * np.log(2.0)) * sum_sq / n)

@jit(nopython=True)
def _calc_slope(y):
    """Simple linear regression slope on normalized data."""
    n = len(y)
    x_mean = (n - 1) / 2.0
    y_mean = 0.0
    for i in range(n):
        y_mean += y[i]
    y_mean /= n
    
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        x_diff = i - x_mean
        numerator += x_diff * (y[i] - y_mean)
        denominator += x_diff * x_diff
        
    if denominator == 0:
        return 0.0
    return numerator / denominator

# --- Feature Registry ---

class FeatureRegistry:
    def __init__(self):
        self._builders: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register("volatility_parkinson", self._feat_parkinson)
        self.register("trend_slope", self._feat_slope)
        self.register("momentum_rsi", self._feat_rsi_proxy)

    def register(self, name: str, func: Callable):
        self._builders[name] = func

    def build_window(self, df: pd.DataFrame, end_idx: int, config: WindowConfig) -> np.ndarray:
        """
        Constructs a feature vector for a specific point in time.
        CRITICAL: Strictly slices df[:end_idx] to prevent look-ahead bias.
        """
        start_idx = max(0, end_idx - config.window_size)
        # Slicing creates a copy/view up to end_idx (exclusive in Python, but we want inclusive for calculation)
        # If end_idx is the current bar, we include it.
        window = df.iloc[start_idx:end_idx+1]
        
        # Extract numpy arrays for Numba
        closes = window['close'].values
        highs = window['high'].values
        lows = window['low'].values
        
        features = []
        
        # 1. Volatility Feature
        features.append(self._builders["volatility_parkinson"](closes, highs, lows))
        
        # 2. Trend Slope (on normalized closes)
        # Normalize closes for slope calculation to be price-agnostic
        if config.scale_method == "robust":
            med = np.median(closes)
            iqr = np.subtract(*np.percentile(closes, [75, 25]))
            if iqr == 0: iqr = 1.0
            norm_closes = (closes - med) / iqr
        else:
            norm_closes = np.log(closes / closes[0])
            
        features.append(self._builders["trend_slope"](norm_closes))
        
        return np.array(features, dtype=np.float32)

    # --- Feature Implementations ---
    
    @staticmethod
    def _feat_parkinson(c, h, l):
        return _calc_parkinson_vol(h, l) * 100 # Scale up for numerical stability

    @staticmethod
    def _feat_slope(norm_c):
        return _calc_slope(norm_c) * 100

    @staticmethod
    def _feat_rsi_proxy(c, h, l):
        # Placeholder for RSI logic
        return 0.0