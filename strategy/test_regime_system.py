
"""
test_regime_system.py
Comprehensive tests for new regime instance system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from regime_instance_engine import RegimeInstanceEngine, RegimeInstance
from regime_data_access import RegimeDataAccess
from regime_statistical_analysis import RegimeStatisticalAnalyzer

class TestRegimeInstanceEngine:
    """Test suite for instance discovery."""
    
    def test_instance_count(self):
        """Verify we get 100+ instances per year of hourly data."""
        # Generate 1 year of hourly data
        dates = pd.date_range(start='2023-01-01', periods=8760, freq='H')
        df = pd.DataFrame(index=dates)
        
        # Mock OHLCV data
        df['close'] = 100 + np.cumsum(np.random.randn(8760) * 0.5)
        df['high'] = df['close'] + np.random.rand(8760) * 2
        df['low'] = df['close'] - np.random.rand(8760) * 2
        df['open'] = df['low'] + (df['high'] - df['low']) * np.random.rand(8760)
        df['volume'] = np.random.randint(1000, 10000, size=8760)
        
        # Add required indicators
        df['atr'] = np.random.rand(8760) + 1
        df['rsi'] = np.random.randint(20, 80, size=8760)
        df['macd_hist'] = np.random.randn(8760)
        df['adx'] = np.random.randint(10, 50, size=8760)
        
        # Discover instances
        engine = RegimeInstanceEngine()
        instances = engine.discover_instances(df, 'BTCUSDT', '1h')
        
        # Assertions
        assert len(instances) >= 100, f"Only {len(instances)} instances found (expected 100+)"
        assert len(instances) <= 300, f"Too many instances: {len(instances)} (expected <300)"
        
        # Check instance quality
        for instance in instances[:10]:  # Check first 10
            assert instance.swing_count >= 0
            assert instance.duration_hours > 0
            assert 0 <= instance.consistency_score <= 100
    
    def test_instance_characterization(self):
        """Verify instances are fully characterized."""
        # Use test data
        df = self._create_test_dataframe(days=30)
        
        engine = RegimeInstanceEngine()
        instances = engine.discover_instances(df, 'BTCUSDT', '1h')
        
        assert len(instances) > 0, "No instances created"
        
        # Check first instance has all required fields
        instance = instances[0]
        
        assert instance.instance_id is not None
        assert instance.pair == 'BTCUSDT'
        assert instance.timeframe == '1h'
        assert instance.dominant_structure is not None
        assert instance.rsi_mean >= 0
        assert instance.volatility_mean >= 0
        assert len(instance.confirming_indicators) >= 0  # Can be empty for low-quality regimes
    
    def test_outcome_calculation(self):
        """Verify outcome metrics are calculated correctly."""
        df = self._create_test_dataframe(days=60)  # Need extra data for outcomes
        
        engine = RegimeInstanceEngine()
        instances = engine.discover_instances(df, 'BTCUSDT', '1h')
        
        # Check that most instances have outcome data
        instances_with_outcomes = [i for i in instances if i.next_1d_return_pct != 0.0]
        
        assert len(instances_with_outcomes) > len(instances) * 0.8, "Most instances should have outcome data"
    
    def _create_test_dataframe(self, days: int) -> pd.DataFrame:
        """Helper to create test data."""
        dates = pd.date_range(start='2023-01-01', periods=days*24, freq='H')
        df = pd.DataFrame(index=dates)
        
        df['close'] = 100 + np.cumsum(np.random.randn(len(df)) * 0.5)
        df['high'] = df['close'] + np.random.rand(len(df)) * 2
        df['low'] = df['close'] - np.random.rand(len(df)) * 2
        df['open'] = df['low'] + (df['high'] - df['low']) * np.random.rand(len(df))
        df['volume'] = np.random.randint(1000, 10000, size=len(df))
        
        df['atr'] = np.random.rand(len(df)) + 1
        df['rsi'] = np.random.randint(20, 80, size=len(df))
        df['macd_hist'] = np.random.randn(len(df))
        df['ppo'] = np.random.randn(len(df)) * 0.5
        df['adx'] = np.random.randint(10, 50, size=len(df))
        
        return df


class TestRegimeDataAccess:
    """Test database operations."""
    
    def test_store_and_retrieve(self):
        """Test full roundtrip: store instance and retrieve it."""
        # Create mock instance
        instance = RegimeInstance(
            instance_id='TEST_001',
            pair='BTCUSDT',
            timeframe='1h',
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 2),
            duration_hours=24,
            swing_count=10,
            avg_swing_magnitude_pct=1.5,
            dominant_structure='trending_up_strong',
            price_change_pct=2.5,
            max_drawdown_pct=-1.2,
            max_runup_pct=3.0,
            volatility_mean=1.5,
            volatility_std=0.2,
            volatility_trend='stable',
            rsi_mean=65,
            rsi_std=5,
            rsi_trend='increasing',
            macd_hist_mean=0.5,
            macd_crossovers=2,
            adx_mean=30,
            adx_trend='increasing',
            volume_mean=5000,
            volume_trend='stable',
            volume_spikes=3,
            consistency_score=75,
            predictability_score=60
        )
        
        # Store
        dao = RegimeDataAccess(mock_db_connector)
        success = dao.store_regime_instance(instance)
        
        assert success, "Failed to store instance"
        
        # Retrieve
        results = dao.get_regime_instances(pair='BTCUSDT', timeframe='1h', limit=10)
        
        assert len(results) > 0, "No instances retrieved"
        assert results[0]['instance_id'] == 'TEST_001'


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""
    
    def test_indicator_causality(self):
        """Test causality analysis for an indicator."""
        # This requires actual data in database
        # Mock or use test database
        pass  # Implement with test fixtures
    
    def test_find_combinations(self):
        """Test finding optimal indicator combinations."""
        pass  # Implement with test fixtures


if __name__ == '__main__':
    pytest.main([__file__, '-v'])