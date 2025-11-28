import os
# Fix for Windows KMeans memory leak warning with MKL
os.environ["OMP_NUM_THREADS"] = "1"

import unittest
import numpy as np
import pandas as pd
from ..features import FeatureRegistry, WindowConfig
from ..pipeline import RegimePipeline

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        # Create dummy OHLCV
        # freq='h' fixes the FutureWarning (was 'H')
        dates = pd.date_range(start='2023-01-01', periods=200, freq='h')
        self.df = pd.DataFrame({
            'open': np.random.rand(200) * 100,
            'high': np.random.rand(200) * 105,
            'low': np.random.rand(200) * 95,
            'close': np.random.rand(200) * 100,
            'volume': np.random.rand(200) * 1000
        }, index=dates)
        
        self.registry = FeatureRegistry()
        self.pipeline = RegimePipeline(n_components=2)

    def test_feature_no_lookahead(self):
        """
        Crucial: Ensure building a window at index T does not change 
        if we append data at T+1.
        """
        idx_t = 100
        config = WindowConfig(window_size=20)
        
        # Build features at T with limited data
        df_t = self.df.iloc[:idx_t+1].copy()
        feat_t = self.registry.build_window(df_t, idx_t, config)
        
        # Build features at T with FULL data
        feat_full = self.registry.build_window(self.df, idx_t, config)
        
        np.testing.assert_array_almost_equal(
            feat_t, feat_full, 
            err_msg="Look-ahead bias detected! Feature vector changed when future data was present."
        )

    def test_pipeline_flow(self):
        # 1. Generate Batch Data
        X = []
        for i in range(50, 150):
            feat = self.registry.build_window(self.df, i, WindowConfig(window_size=30))
            X.append(feat)
        X = np.array(X)
        
        # 2. Train
        self.pipeline.fit(X)
        self.assertTrue(self.pipeline.is_fitted)
        
        # 3. Predict Single
        result = self.pipeline.predict(X[0:1])
        self.assertIn("regime", result)
        self.assertIn("confidence", result)
        self.assertTrue(0 <= result['regime'] <= 1)

if __name__ == '__main__':
    unittest.main()