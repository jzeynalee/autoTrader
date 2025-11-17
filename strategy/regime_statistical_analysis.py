
"""
regime_statistical_analysis.py
Statistical experiments to determine indicator/pattern causality.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from .regime_data_access import RegimeDataAccess

class RegimeStatisticalAnalyzer:
    """
    Runs statistical experiments on regime instance data to find:
    1. Which indicators/patterns CAUSE positive outcomes
    2. Which combinations are most predictive
    3. Optimal thresholds for each indicator
    """
    
    def __init__(self, regime_dao: RegimeDataAccess):
        self.dao = regime_dao
        self.experiments = []
    
    def analyze_indicator_causality(
        self,
        indicator_name: str,
        outcome_metric: str = 'next_1d_return_pct',
        min_sample_size: int = 30
    ) -> Dict:
        """
        Determine if an indicator has causal relationship with outcomes.
        
        Uses:
        - Correlation analysis
        - T-test for mean differences
        - Logistic regression for predictive power
        
        Returns:
            {
                'correlation': float,
                'p_value': float,
                'mean_when_high': float,
                'mean_when_low': float,
                'predictive_power': float,  # 0-100
                'optimal_threshold': float,
                'sample_size': int,
                'recommendation': str
            }
        """
        # Get indicator data with outcomes
        df = self.dao.get_indicator_statistics(indicator_name, min_strength=0)
        
        if len(df) < min_sample_size:
            return {
                'error': f'Insufficient data: {len(df)} samples (need {min_sample_size})'
            }
        
        # 1. Correlation analysis
        correlation = df['mean_value'].corr(df[outcome_metric])
        
        # 2. Split by indicator value (median split)
        median_value = df['mean_value'].median()
        high_group = df[df['mean_value'] > median_value][outcome_metric]
        low_group = df[df['mean_value'] <= median_value][outcome_metric]
        
        # 3. T-test for mean difference
        t_stat, p_value = stats.ttest_ind(high_group, low_group)
        
        # 4. Calculate win rates
        high_win_rate = (high_group > 0).mean() * 100
        low_win_rate = (low_group > 0).mean() * 100
        
        # 5. Find optimal threshold (maximize win rate)
        thresholds = np.percentile(df['mean_value'], np.arange(10, 91, 5))
        best_threshold = median_value
        best_win_rate = 0
        
        for threshold in thresholds:
            above_thresh = df[df['mean_value'] > threshold][outcome_metric]
            if len(above_thresh) >= 10:  # Minimum sample
                win_rate = (above_thresh > 0).mean()
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_threshold = threshold
        
        # 6. Predictive power score (0-100)
        # Based on: correlation strength, p-value, win rate difference
        predictive_power = 0
        if abs(correlation) > 0.3:
            predictive_power += 40
        elif abs(correlation) > 0.2:
            predictive_power += 20
        
        if p_value < 0.01:
            predictive_power += 30
        elif p_value < 0.05:
            predictive_power += 15
        
        win_rate_diff = abs(high_win_rate - low_win_rate)
        if win_rate_diff > 20:
            predictive_power += 30
        elif win_rate_diff > 10:
            predictive_power += 15
        
        # 7. Recommendation
        if predictive_power >= 70:
            recommendation = "STRONG SIGNAL - Use in strategy"
        elif predictive_power >= 50:
            recommendation = "MODERATE SIGNAL - Combine with others"
        elif predictive_power >= 30:
            recommendation = "WEAK SIGNAL - Use as confirmation only"
        else:
            recommendation = "NO SIGNAL - Do not use"
        
        return {
            'indicator_name': indicator_name,
            'correlation': float(correlation),
            'p_value': float(p_value),
            'mean_when_high': float(high_group.mean()),
            'mean_when_low': float(low_group.mean()),
            'win_rate_when_high': float(high_win_rate),
            'win_rate_when_low': float(low_win_rate),
            'predictive_power': float(predictive_power),
            'optimal_threshold': float(best_threshold),
            'sample_size': len(df),
            'recommendation': recommendation
        }
    
    def analyze_pattern_effectiveness(
        self,
        pattern_name: str,
        pattern_type: str = 'candlestick'
    ) -> Dict:
        """
        Analyze how effective a pattern is for prediction.
        
        Returns statistical significance and effect size.
        """
        df = self.dao.get_pattern_effectiveness(pattern_name, pattern_type)
        
        if len(df) < 10:
            return {'error': f'Insufficient data: {len(df)} occurrences'}
        
        # Compare outcomes with vs. without pattern
        # (Need to query all instances for comparison)
        all_instances_query = """
        SELECT 
            instance_id,
            next_1d_return_pct,
            next_3d_return_pct,
            next_7d_return_pct
        FROM regime_instances
        WHERE next_1d_return_pct IS NOT NULL
        """
        
        all_df = pd.DataFrame(self.dao.db.execute(all_instances_query, fetch=True))
        
        # Instances WITH pattern
        with_pattern = df['next_1d_return_pct']
        
        # Instances WITHOUT pattern
        pattern_instance_ids = set(df['instance_id'])
        without_pattern = all_df[~all_df['instance_id'].isin(pattern_instance_ids)]['next_1d_return_pct']
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(with_pattern, without_pattern)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((with_pattern.std()**2 + without_pattern.std()**2) / 2)
        cohens_d = (with_pattern.mean() - without_pattern.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Win rates
        win_rate_with = (with_pattern > 0).mean() * 100
        win_rate_without = (without_pattern > 0).mean() * 100
        
        return {
            'pattern_name': pattern_name,
            'pattern_type': pattern_type,
            'occurrences': len(df),
            'avg_return_with_pattern': float(with_pattern.mean()),
            'avg_return_without_pattern': float(without_pattern.mean()),
            'win_rate_with_pattern': float(win_rate_with),
            'win_rate_without_pattern': float(win_rate_without),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
            'statistically_significant': p_value < 0.05
        }
    
    def find_optimal_indicator_combinations(
        self,
        target_metric: str = 'next_1d_return_pct',
        max_indicators: int = 5
    ) -> List[Dict]:
        """
        Use machine learning to find best indicator combinations.
        
        Returns ranked list of indicator combinations with predictive power.
        """
        # Get all instances with indicator data
        query = """
        SELECT 
            ri.instance_id,
            ri.next_1d_return_pct,
            ri.next_3d_return_pct,
            ri.consistency_score,
            ri.dominant_structure,
            GROUP_CONCAT(DISTINCT rci.indicator_name) as indicators,
            GROUP_CONCAT(DISTINCT rci.mean_value) as indicator_values
        FROM regime_instances ri
        LEFT JOIN regime_confirming_indicators rci ON ri.instance_id = rci.instance_id
        WHERE ri.next_1d_return_pct IS NOT NULL
        GROUP BY ri.instance_id
        HAVING COUNT(DISTINCT rci.indicator_name) >= 3
        """
        
        df = pd.DataFrame(self.dao.db.execute(query, fetch=True))
        
        # Build feature matrix (one-hot encode indicators)
        all_indicators = set()
        for indicators_str in df['indicators']:
            if pd.notna(indicators_str):
                all_indicators.update(indicators_str.split(','))
        
        # Create binary features for each indicator
        for indicator in all_indicators:
            df[f'has_{indicator}'] = df['indicators'].str.contains(indicator, na=False).astype(int)
        
        # Prepare for Random Forest
        feature_cols = [col for col in df.columns if col.startswith('has_')]
        X = df[feature_cols].values
        y = (df[target_metric] > 0).astype(int).values  # Binary classification
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'indicator': [col.replace('has_', '') for col in feature_cols],
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Find top combinations
        top_indicators = importances.head(max_indicators)['indicator'].tolist()
        
        combinations = []
        
        # Test different combinations
        for size in range(2, min(max_indicators + 1, len(top_indicators) + 1)):
            from itertools import combinations as comb
            
            for combo in comb(top_indicators, size):
                # Filter instances that have ALL indicators in combo
                mask = pd.Series([True] * len(df))
                for indicator in combo:
                    mask &= df[f'has_{indicator}'] == 1
                
                subset = df[mask]
                
                if len(subset) >= 20:  # Minimum sample size
                    win_rate = (subset[target_metric] > 0).mean() * 100
                    avg_return = subset[target_metric].mean()
                    
                    combinations.append({
                        'indicators': list(combo),
                        'sample_size': len(subset),
                        'win_rate': float(win_rate),
                        'avg_return': float(avg_return),
                        'score': float(win_rate * avg_return)  # Combined score
                    })
        
        # Sort by score
        combinations.sort(key=lambda x: x['score'], reverse=True)
        
        return combinations[:20]  # Top 20 combinations
    
    def generate_strategy_from_instances(
        self,
        instance_ids: List[str],
        strategy_name: str
    ) -> Dict:
        """
        Generate a concrete trading strategy from a group of similar instances.
        
        Analyzes common characteristics and creates entry/exit rules.
        """
        # Get all data for these instances
        placeholders = ','.join(['%s'] * len(instance_ids))
        query = f"""
        SELECT 
            ri.*,
            GROUP_CONCAT(DISTINCT rci.indicator_name) as all_indicators,
            GROUP_CONCAT(DISTINCT rcp.pattern_name) as all_patterns
        FROM regime_instances ri
        LEFT JOIN regime_confirming_indicators rci ON ri.instance_id = rci.instance_id
        LEFT JOIN regime_candlestick_patterns rcp ON ri.instance_id = rcp.instance_id
        WHERE ri.instance_id IN ({placeholders})
        GROUP BY ri.instance_id
        """
        
        df = pd.DataFrame(self.dao.db.execute(query, tuple(instance_ids), fetch=True))
        
        if len(df) == 0:
            return {'error': 'No instances found'}
        
        # Extract common characteristics
        common_indicators = self._find_common_elements(df['all_indicators'])
        common_patterns = self._find_common_elements(df['all_patterns'])
        
        # Calculate average metrics
        avg_rsi = df['rsi_mean'].mean()
        avg_adx = df['adx_mean'].mean()
        avg_volatility = df['volatility_mean'].mean()
        
        # Determine entry conditions
        entry_conditions = []
        
        if avg_rsi > 55:
            entry_conditions.append(f"RSI > {avg_rsi - 5:.1f}")
        elif avg_rsi < 45:
            entry_conditions.append(f"RSI < {avg_rsi + 5:.1f}")
        
        if avg_adx > 25:
            entry_conditions.append(f"ADX > {avg_adx - 5:.1f}")
        
        for indicator in common_indicators[:3]:  # Top 3
            entry_conditions.append(f"{indicator} confirming")
        
        for pattern in common_patterns[:2]:  # Top 2
            entry_conditions.append(f"{pattern} present")
        
        # Determine risk management
        avg_drawdown = abs(df['max_drawdown_pct'].mean())
        avg_runup = abs(df['max_runup_pct'].mean())
        
        stop_loss = avg_drawdown * 1.2  # 20% buffer
        take_profit = avg_runup * 0.8   # Conservative target
        
        # Calculate historical performance
        win_rate = (df['next_1d_return_pct'] > 0).mean() * 100
        avg_return = df['next_1d_return_pct'].mean()
        
        strategy = {
            'strategy_name': strategy_name,
            'based_on_instances': len(df),
            'entry_conditions': entry_conditions,
            'stop_loss_pct': float(stop_loss),
            'take_profit_pct': float(take_profit),
            'position_sizing': 'normal' if avg_volatility < 2.0 else 'reduced',
            'historical_win_rate': float(win_rate),
            'historical_avg_return': float(avg_return),
            'common_indicators': common_indicators,
            'common_patterns': common_patterns,
            'status': 'candidate'
        }
        
        return strategy
    
    def _find_common_elements(self, series: pd.Series, min_frequency: float = 0.5) -> List[str]:
        """Find elements that appear in at least min_frequency of rows."""
        all_elements = []
        
        for row in series:
            if pd.notna(row):
                elements = row.split(',')
                all_elements.extend([e.strip() for e in elements])
        
        if not all_elements:
            return []
        
        # Count frequencies
        from collections import Counter
        counter = Counter(all_elements)
        
        total = len(series)
        threshold = total * min_frequency
        
        common = [elem for elem, count in counter.items() if count >= threshold]
        
        # Sort by frequency
        common.sort(key=lambda x: counter[x], reverse=True)
        
        return common