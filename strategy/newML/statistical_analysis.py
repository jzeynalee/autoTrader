# statistical_analysis.py
import numpy as np
import pandas as pd
import time
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import math
import itertools
from typing import Dict, List

# Optional dependencies handling
try:
    import statsmodels.api as sm
    STATSMODELS = True
except ImportError:
    STATSMODELS = False

from ...db_connector import DatabaseConnector

class RegimeStatisticalAnalyzer:
    """
    Advanced Statistical Engine.
    Replaces Pearson Correlation with Granger Causality, Mutual Information, and Distance Correlation.
    """

    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector
        # Exclude 1m as per your file's logic
        self.excluded_tfs = ['1m'] 

    def _results_to_dataframe(self, results, cols):
        if not results: return pd.DataFrame(columns=cols)
        return pd.DataFrame(results, columns=cols)

    # ---------------------- 1. DISTANCE CORRELATION --------------------------
    def distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Computes Distance Correlation (measures non-linear dependence)."""
        x = np.atleast_1d(x).astype(float)
        y = np.atleast_1d(y).astype(float)
        if x.size != y.size: raise ValueError("Size mismatch")
        n = x.size
        if n <= 1: return 0.0

        a = np.abs(x[:, None] - x[None, :])
        b = np.abs(y[:, None] - y[None, :])

        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum() / (n * n)
        dcov2_xx = (A * A).sum() / (n * n)
        dcov2_yy = (B * B).sum() / (n * n)

        if dcov2_xx * dcov2_yy <= 0: return 0.0
        return math.sqrt(max(dcov2_xy, 0.0) / math.sqrt(dcov2_xx * dcov2_yy))

    # ---------------------- 2. MUTUAL INFORMATION ------------------------
    def mutual_info_test(self, indicator_name: str, outcome='next_1d_return_pct') -> Dict:
        """Measures how much 'information' the indicator provides about the outcome."""
        query = f"""
        SELECT ri.{outcome},
               (SELECT COUNT(*) FROM regime_confirming_indicators rci
                WHERE rci.instance_id = ri.instance_id AND rci.indicator_name = ?) as has_indicator
        FROM regime_instances ri
        WHERE ri.{outcome} IS NOT NULL
        """
        results = self.db.execute(query, (indicator_name,), fetch=True)
        if not results or len(results) < 50:
            return {'mutual_info': 0.0, 'valid': False}

        df = self._results_to_dataframe(results, [outcome, 'has_indicator'])
        
        try:
            mi = mutual_info_regression(df[['has_indicator']].values, df[outcome].values, random_state=42)[0]
            return {'mutual_info': mi, 'valid': True}
        except:
            return {'mutual_info': 0.0, 'valid': False}

    # ---------------------- 3. GRANGER CAUSALITY ------------------------------
    def granger_causality_test(self, indicator_name: str, outcome='next_1d_return_pct', maxlag=3) -> Dict:
        """
        Determines if the indicator is a LEADING indicator (Causing Factor).
        """
        if not STATSMODELS:
            return {'p_value': 1.0, 'is_causal': False, 'note': 'Statsmodels missing'}

        # Get time-ordered series
        query = f"""
        SELECT ri.{outcome},
               (SELECT COUNT(*) FROM regime_confirming_indicators rci
                WHERE rci.instance_id = ri.instance_id AND rci.indicator_name = ?) as has_indicator
        FROM regime_instances ri
        WHERE ri.{outcome} IS NOT NULL
        ORDER BY ri.start_time ASC
        """
        results = self.db.execute(query, (indicator_name,), fetch=True)
        
        if not results or len(results) < 100:
            return {'p_value': 1.0, 'is_causal': False, 'note': 'Insufficient Data'}

        df = self._results_to_dataframe(results, [outcome, 'has_indicator'])
        
        # Check for stationarity or variance (Granger fails on constant data)
        if df['has_indicator'].var() == 0:
            return {'p_value': 1.0, 'is_causal': False, 'note': 'Constant Indicator'}

        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            # input format: [target, predictor]
            gc_res = grangercausalitytests(df[[outcome, 'has_indicator']], maxlag=maxlag, verbose=False)
            
            # Get min p-value across lags
            p_values = [gc_res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
            min_p = min(p_values)
            
            return {'p_value': min_p, 'is_causal': min_p < 0.05}
            
        except Exception as e:
            return {'p_value': 1.0, 'is_causal': False, 'note': str(e)}

    # ---------------------- 4. CORRELATION (Fallback/Direction) ----------------
    def get_correlation_direction(self, indicator_name: str, target_regime: int) -> str:
        """
        Still need basic correlation to know if HIGH indicator = Regime, or LOW indicator = Regime.
        """
        query = f"""
        SELECT ri.dominant_structure,
               (SELECT COUNT(*) FROM regime_confirming_indicators rci
                WHERE rci.instance_id = ri.instance_id AND rci.indicator_name = ?) as val
        FROM regime_instances ri
        """
        results = self.db.execute(query, (indicator_name,), fetch=True)
        if not results: return "UNKNOWN"
        
        df = self._results_to_dataframe(results, ['regime', 'val'])
        
        # Binary target: 1 if regime matches target, else 0
        df['target'] = (df['regime'] == target_regime).astype(int)
        corr = df['val'].corr(df['target'])
        
        return "HIGH" if corr > 0 else "LOW"