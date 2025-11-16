
"""
regime_data_access.py
Database access layer for regime instance data.
"""

import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional
from regime_instance_engine import RegimeInstance

class RegimeDataAccess:
    """
    Handles all database operations for regime instances.
    Provides clean interface for storage and retrieval.
    """
    
    def __init__(self, db_connector):
        """
        Args:
            db_connector: Database connection object with execute() method
        """
        self.db = db_connector
    
    def store_regime_instance(self, instance: RegimeInstance) -> bool:
        """
        Store a complete regime instance with all related data.
        
        This is a transactional operation:
        1. Insert master record
        2. Insert confirming indicators
        3. Insert candlestick patterns
        4. Insert price action patterns
        5. Insert chart patterns
        
        Returns True on success, False on failure.
        """
        try:
            # 1. Master record
            self._insert_master_record(instance)
            
            # 2. Confirming indicators
            self._insert_confirming_indicators(instance)
            
            # 3. Candlestick patterns
            self._insert_candlestick_patterns(instance)
            
            # 4. Price action patterns
            self._insert_price_action_patterns(instance)
            
            # 5. Chart patterns
            self._insert_chart_patterns(instance)
            
            return True
            
        except Exception as e:
            print(f"Error storing regime instance {instance.instance_id}: {e}")
            return False
    
    def _insert_master_record(self, instance: RegimeInstance):
        """Insert the main regime_instances record."""
        query = """
        INSERT INTO regime_instances (
            instance_id, pair, timeframe,
            start_time, end_time, duration_hours,
            swing_count, avg_swing_magnitude_pct, dominant_structure,
            price_change_pct, max_drawdown_pct, max_runup_pct,
            volatility_mean, volatility_std, volatility_trend,
            rsi_mean, rsi_std, rsi_trend,
            macd_hist_mean, macd_crossovers,
            adx_mean, adx_trend,
            volume_mean, volume_trend, volume_spikes,
            higher_highs, higher_lows, lower_highs, lower_lows,
            structure_breaks_bullish, structure_breaks_bearish,
            pullback_count, avg_pullback_depth_pct, failed_pullbacks,
            next_1d_return_pct, next_3d_return_pct, next_7d_return_pct,
            max_favorable_excursion_1d, max_adverse_excursion_1d,
            consistency_score, predictability_score,
            bar_count
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            updated_at = CURRENT_TIMESTAMP
        """
        
        values = (
            instance.instance_id, instance.pair, instance.timeframe,
            instance.start_time, instance.end_time, instance.duration_hours,
            instance.swing_count, instance.avg_swing_magnitude_pct, instance.dominant_structure,
            instance.price_change_pct, instance.max_drawdown_pct, instance.max_runup_pct,
            instance.volatility_mean, instance.volatility_std, instance.volatility_trend,
            instance.rsi_mean, instance.rsi_std, instance.rsi_trend,
            instance.macd_hist_mean, instance.macd_crossovers,
            instance.adx_mean, instance.adx_trend,
            instance.volume_mean, instance.volume_trend, instance.volume_spikes,
            instance.higher_highs, instance.higher_lows, instance.lower_highs, instance.lower_lows,
            instance.structure_breaks_bullish, instance.structure_breaks_bearish,
            instance.pullback_count, instance.avg_pullback_depth_pct, instance.failed_pullbacks,
            instance.next_1d_return_pct, instance.next_3d_return_pct, instance.next_7d_return_pct,
            instance.max_favorable_excursion_1d, instance.max_adverse_excursion_1d,
            instance.consistency_score, instance.predictability_score,
            len(instance.bar_indices)
        )
        
        self.db.execute(query, values)
    
    def _insert_confirming_indicators(self, instance: RegimeInstance):
        """Insert confirming indicators for this instance."""
        if not instance.indicator_values:
            return
        
        query = """
        INSERT INTO regime_confirming_indicators (
            instance_id, indicator_name,
            mean_value, std_value, min_value, max_value, trend,
            confirmation_strength
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            mean_value = VALUES(mean_value),
            std_value = VALUES(std_value),
            min_value = VALUES(min_value),
            max_value = VALUES(max_value),
            trend = VALUES(trend),
            confirmation_strength = VALUES(confirmation_strength)
        """
        
        for indicator_name, stats in instance.indicator_values.items():
            # Calculate confirmation strength (0-100)
            # High strength = consistent, extreme values
            strength = self._calculate_confirmation_strength(indicator_name, stats)
            
            values = (
                instance.instance_id,
                indicator_name,
                stats.get('mean', 0),
                stats.get('std', 0),
                stats.get('min', 0),
                stats.get('max', 0),
                stats.get('trend', 'stable'),
                strength
            )
            
            self.db.execute(query, values)
    
    def _calculate_confirmation_strength(self, indicator_name: str, stats: Dict) -> float:
        """
        Calculate how strongly this indicator confirms the regime.
        
        Factors:
        - Low std/mean ratio = consistent = strong
        - Extreme values (RSI >70 or <30) = strong
        - Clear trend = strong
        """
        strength = 50.0  # Base strength
        
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)
        trend = stats.get('trend', 'stable')
        
        # Consistency bonus (low variance)
        if mean != 0:
            cv = std / abs(mean)  # Coefficient of variation
            if cv < 0.1:
                strength += 20
            elif cv < 0.2:
                strength += 10
        
        # Extreme value bonus (indicator-specific)
        if 'rsi' in indicator_name.lower():
            if mean > 70 or mean < 30:
                strength += 20
            elif mean > 60 or mean < 40:
                strength += 10
        
        if 'adx' in indicator_name.lower():
            if mean > 40:
                strength += 20
            elif mean > 25:
                strength += 10
        
        # Trend bonus
        if trend in ['increasing', 'decreasing']:
            strength += 10
        
        return min(strength, 100.0)
    
    def _insert_candlestick_patterns(self, instance: RegimeInstance):
        """Insert candlestick patterns for this instance."""
        query = """
        INSERT INTO regime_candlestick_patterns (
            instance_id, pattern_name, pattern_type, occurrence_count
        ) VALUES (%s, %s, %s, %s)
        """
        
        # Bullish patterns
        for pattern_name, count in instance.bullish_patterns.items():
            values = (instance.instance_id, pattern_name, 'bullish', count)
            self.db.execute(query, values)
        
        # Bearish patterns
        for pattern_name, count in instance.bearish_patterns.items():
            values = (instance.instance_id, pattern_name, 'bearish', count)
            self.db.execute(query, values)
        
        # Neutral patterns
        for pattern_name, count in instance.neutral_patterns.items():
            values = (instance.instance_id, pattern_name, 'neutral', count)
            self.db.execute(query, values)
    
    def _insert_price_action_patterns(self, instance: RegimeInstance):
        """Insert price action patterns for this instance."""
        if not instance.price_action_patterns:
            return
        
        query = """
        INSERT INTO regime_price_action_patterns (
            instance_id, pattern_name, occurrence_count
        ) VALUES (%s, %s, %s)
        """
        
        # Count occurrences of each pattern
        pattern_counts = {}
        for pattern in instance.price_action_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        for pattern_name, count in pattern_counts.items():
            values = (instance.instance_id, pattern_name, count)
            self.db.execute(query, values)
    
    def _insert_chart_patterns(self, instance: RegimeInstance):
        """Insert chart patterns for this instance."""
        if not instance.chart_patterns:
            return
        
        query = """
        INSERT INTO regime_chart_patterns (
            instance_id, pattern_name
        ) VALUES (%s, %s)
        """
        
        for pattern_name in instance.chart_patterns:
            values = (instance.instance_id, pattern_name)
            self.db.execute(query, values)
    
    def get_regime_instances(
        self,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        structure: Optional[str] = None,
        min_consistency: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Query regime instances with filters.
        
        Returns list of instance dictionaries with all data.
        """
        conditions = []
        params = []
        
        if pair:
            conditions.append("pair = %s")
            params.append(pair)
        
        if timeframe:
            conditions.append("timeframe = %s")
            params.append(timeframe)
        
        if structure:
            conditions.append("dominant_structure = %s")
            params.append(structure)
        
        if min_consistency:
            conditions.append("consistency_score >= %s")
            params.append(min_consistency)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT * FROM regime_instances
        WHERE {where_clause}
        ORDER BY start_time DESC
        LIMIT %s
        """
        
        params.append(limit)
        
        return self.db.execute(query, params, fetch=True)
    
    def get_indicator_statistics(
        self,
        indicator_name: str,
        min_strength: float = 50.0
    ) -> pd.DataFrame:
        """
        Get statistical distribution of an indicator across all regimes.
        
        Useful for understanding: "When RSI is >70, what typically happens?"
        """
        query = """
        SELECT 
            rci.instance_id,
            rci.mean_value,
            rci.std_value,
            rci.trend,
            rci.confirmation_strength,
            ri.dominant_structure,
            ri.next_1d_return_pct,
            ri.next_3d_return_pct,
            ri.consistency_score
        FROM regime_confirming_indicators rci
        JOIN regime_instances ri ON rci.instance_id = ri.instance_id
        WHERE rci.indicator_name = %s
          AND rci.confirmation_strength >= %s
        ORDER BY rci.confirmation_strength DESC
        """
        
        results = self.db.execute(query, (indicator_name, min_strength), fetch=True)
        return pd.DataFrame(results)
    
    def get_pattern_effectiveness(
        self,
        pattern_name: str,
        pattern_type: str = 'candlestick'
    ) -> pd.DataFrame:
        """
        Analyze how effective a pattern is for predicting outcomes.
        
        Returns DataFrame with:
        - Regimes where pattern occurred
        - Outcome metrics (1d, 3d, 7d returns)
        - Win rate, average return, etc.
        """
        if pattern_type == 'candlestick':
            query = """
            SELECT 
                rcp.instance_id,
                rcp.pattern_type,
                rcp.occurrence_count,
                ri.dominant_structure,
                ri.price_change_pct,
                ri.next_1d_return_pct,
                ri.next_3d_return_pct,
                ri.next_7d_return_pct,
                ri.predictability_score
            FROM regime_candlestick_patterns rcp
            JOIN regime_instances ri ON rcp.instance_id = ri.instance_id
            WHERE rcp.pattern_name = %s
            ORDER BY ri.start_time DESC
            """
        else:  # price_action
            query = """
            SELECT 
                rpap.instance_id,
                rpap.occurrence_count,
                ri.dominant_structure,
                ri.price_change_pct,
                ri.next_1d_return_pct,
                ri.next_3d_return_pct,
                ri.next_7d_return_pct,
                ri.predictability_score
            FROM regime_price_action_patterns rpap
            JOIN regime_instances ri ON rpap.instance_id = ri.instance_id
            WHERE rpap.pattern_name = %s
            ORDER BY ri.start_time DESC
            """
        
        results = self.db.execute(query, (pattern_name,), fetch=True)
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            # Calculate effectiveness metrics
            df['outcome_positive'] = df['next_1d_return_pct'] > 0
            
            summary = {
                'sample_size': len(df),
                'win_rate': df['outcome_positive'].mean() * 100,
                'avg_1d_return': df['next_1d_return_pct'].mean(),
                'avg_3d_return': df['next_3d_return_pct'].mean(),
                'avg_7d_return': df['next_7d_return_pct'].mean(),
                'avg_predictability': df['predictability_score'].mean()
            }
            
            df.attrs['summary'] = summary
        
        return df