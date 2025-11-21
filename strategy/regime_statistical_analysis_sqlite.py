"""
regime_statistical_analysis_sqlite.py
Statistical experiments to determine indicator/pattern causality - SQLite version.

CHANGES FROM MySQL VERSION:
- Uses ? instead of %s for parameters
- Uses GROUP_CONCAT without DISTINCT keyword (SQLite doesn't need it)
- Compatible with SQLite3 syntax
"""

import pandas as pd
import numpy as np
from scipy import stats
try:
    import shap
    import xgboost as xgb
    from sklearn.feature_selection import mutual_info_classif
except ImportError:
    print("⚠️ Advanced stats libs (shap, xgboost, sklearn) missing.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from .regime_data_access_sqlite import RegimeDataAccess

class RegimeStatisticalAnalyzer:
    """
    Runs statistical experiments on regime instance data to find:
    1. Which indicators/patterns CAUSE positive outcomes
    2. Which combinations are most predictive
    3. Optimal thresholds for each indicator
    
    SQLite-compatible version.
    """
    
    def __init__(self, regime_dao: RegimeDataAccess):
        self.dao = regime_dao
        self.experiments = []
    
    #def analyze_indicator_causality(
    def analyze_numeric_factor(
        self,
        indicator_name: str,
        #outcome_metric: str = 'next_1d_return_pct',
        regime_col: str = 'dominant_structure',
        min_sample_size: int = 30
    ) -> Dict:
        """
        Phase 1 (Screening): Tests if a numeric indicator statistically distinguishes regimes.
        Uses Kruskal-Wallis (Non-parametric ANOVA) and Mutual Information.
        """
        # Get raw data: value vs regime
        query = """
        SELECT 
            rci.mean_value as value,
            ri.dominant_structure as regime
        FROM regime_confirming_indicators rci
        JOIN regime_instances ri ON rci.instance_id = ri.instance_id
        WHERE rci.indicator_name = ?
        """
        results = self.dao.db.execute(query, (indicator_name,), fetch=True)
        df = pd.DataFrame(results)
        
        
        if len(df) < min_sample_size:
            return {
                'error': f'Insufficient data: {len(df)} samples (need {min_sample_size})'
            }
        
        # 1. Kruskal-Wallis H Test (Robust to non-normal financial data)
        groups = [group['value'].values for name, group in df.groupby('regime')]

        if len(groups) < 2:
            return {'error': 'Not enough unique regimes for comparison'}
            
        stat, p_value = stats.kruskal(*groups)
        
        # 2. Mutual Information (Captures non-linear dependencies)
        # Encodes regime labels to integers for MI calculation
        df['regime_code'] = df['regime'].astype('category').cat.codes
        mi_score = mutual_info_classif(
            df[['value']], 
            df['regime_code'], 
            discrete_features=False, 
            random_state=42
        )[0]
        
        # 3. Interpretation
        is_significant = p_value < 0.05
        strength = "High" if mi_score > 0.1 else "Low"
        
        return {
            'test': 'Kruskal-Wallis + Mutual Info',
            'statistic': float(stat),
            'p_value': float(p_value),
            'mutual_info': float(mi_score),
            'significant': is_significant,
            'recommendation': 'KEEP' if is_significant else 'DISCARD',
            'summary': f"{indicator_name} has {strength} influence (p={p_value:.4f}, MI={mi_score:.3f})"
        }

    def analyze_categorical_factor(
        self,
        pattern_name: str,
        regime_col: str = 'dominant_structure'
    ) -> Dict:
        """
        Phase 1 (Screening): Tests if a categorical pattern appears differently across regimes.
        Uses G-Test (Log-Likelihood) instead of Chi-Square for better handling of rare patterns.
        """
        # Get occurrence data
        query = """
        SELECT 
            ri.dominant_structure as regime,
            CASE WHEN rcp.pattern_name IS NOT NULL THEN 1 ELSE 0 END as present
        FROM regime_instances ri
        LEFT JOIN regime_candlestick_patterns rcp 
            ON ri.instance_id = rcp.instance_id AND rcp.pattern_name = ?
        """
        results = self.dao.db.execute(query, (pattern_name,), fetch=True)
        df = pd.DataFrame(results)
        
        contingency_table = pd.crosstab(df['present'], df['regime'])
        
        # G-Test (Log-Likelihood Ratio)
        # lambda_=0 gives G-test, lambda_=1 gives Chi-square
        g_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table, lambda_="log-likelihood")
        
        # Cramer's V for effect size
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(g_stat / (n * min_dim)) if min_dim > 0 else 0


        return {
            'test': 'G-Test (Log-Likelihood)',
            'statistic': float(g_stat),
            'p_value': float(p_value),
            'cramers_v': float(cramers_v),
            'significant': p_value < 0.05,
            'summary': f"Pattern '{pattern_name}' is {'significantly' if p_value < 0.05 else 'not'} regime-dependent."
        }

    def perform_shap_analysis(self, target_metric='next_1d_return_pct', top_n=10):
        """
        Phase 2 (Explanation): Uses XGBoost + SHAP to find TRUE drivers of profit.
        Handles non-linearity and interaction effects.
        """
        # 1. Prepare Data (One-hot encoded indicators + Outcomes)
        query = """
        SELECT 
            ri.next_1d_return_pct,
            ri.volatility_mean,
            ri.rsi_mean,
            ri.adx_mean,
            GROUP_CONCAT(rci.indicator_name) as indicators
        FROM regime_instances ri
        LEFT JOIN regime_confirming_indicators rci ON ri.instance_id = rci.instance_id
        WHERE ri.next_1d_return_pct IS NOT NULL
        GROUP BY ri.instance_id
        """
        df = pd.DataFrame(self.dao.db.execute(query, fetch=True))
        if len(df) < 50: return {'error': 'Insufficient data for SHAP'}

        # Feature Engineering
        # Convert indicator list to binary features
        all_inds = set()
        for i in df['indicators'].dropna(): all_inds.update(i.split(','))
        
        for ind in list(all_inds)[:50]: # Limit to top 50 to prevent explosion
            df[ind] = df['indicators'].apply(lambda x: 1 if isinstance(x, str) and ind in x else 0)
            
        X = df.drop(columns=['next_1d_return_pct', 'indicators']).fillna(0)
        y = (df[target_metric] > 0).astype(int) # Binary classification: Profit or Loss
        
        # 2. Train XGBoost
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        
        # 3. Calculate SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # 4. Summarize Importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False).head(top_n)

        return {
            'test': 'SHAP Feature Importance',
            'top_drivers': importance_df.to_dict('records')
        }
   
    def verify_robustness(self, feature_col, target_col='next_1d_return_pct', n_permutations=1000):
        """
        Phase 3 (Validation): Permutation Test (White's Reality Check).
        Shuffles the target variable to see if the relationship holds by chance.
        """
        query = f"SELECT {feature_col}, {target_col} FROM regime_instances WHERE {target_col} IS NOT NULL"
        try:
            df = pd.DataFrame(self.dao.db.execute(query, fetch=True))
        except:
            return {'error': f'Column {feature_col} not found in instances'}
            
        if len(df) < 50: return {'error': 'Insufficient data'}
        
        # Calculate real correlation (Spearman for non-linear rank)
        real_corr, _ = stats.spearmanr(df[feature_col], df[target_col])
        
        # Permutation loop
        fake_corrs = []
        y_values = df[target_col].values.copy()
        
        for _ in range(n_permutations):
            np.random.shuffle(y_values)
            corr, _ = stats.spearmanr(df[feature_col], y_values)
            fake_corrs.append(corr)
            
        # Calculate p-value: portion of fake correlations stronger than real
        fake_corrs = np.array(fake_corrs)
        p_value = (np.abs(fake_corrs) >= np.abs(real_corr)).mean()
        
        return {
            'test': 'Permutation Robustness Check',
            'real_correlation': float(real_corr),
            'permutation_p_value': float(p_value),
            'is_robust': p_value < 0.05,
            'conclusion': "Robust Feature" if p_value < 0.05 else "Likely Noise/Overfitting"
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
        # Note: SQLite GROUP_CONCAT doesn't need DISTINCT, it's automatic
        query = """
        SELECT 
            ri.instance_id,
            ri.next_1d_return_pct,
            ri.next_3d_return_pct,
            ri.consistency_score,
            ri.dominant_structure,
            GROUP_CONCAT(rci.indicator_name) as indicators,
            GROUP_CONCAT(rci.mean_value) as indicator_values
        FROM regime_instances ri
        LEFT JOIN regime_confirming_indicators rci ON ri.instance_id = rci.instance_id
        WHERE ri.next_1d_return_pct IS NOT NULL
        GROUP BY ri.instance_id
        HAVING COUNT(rci.indicator_name) >= 3
        """
        
        results = self.dao.db.execute(query, fetch=True)
        if not results:
            return []
        
        df = pd.DataFrame(results)
        
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
        # SQLite uses ? for parameters
        placeholders = ','.join(['?'] * len(instance_ids))
        query = f"""
        SELECT 
            ri.*,
            GROUP_CONCAT(rci.indicator_name) as all_indicators,
            GROUP_CONCAT(rcp.pattern_name) as all_patterns
        FROM regime_instances ri
        LEFT JOIN regime_confirming_indicators rci ON ri.instance_id = rci.instance_id
        LEFT JOIN regime_candlestick_patterns rcp ON ri.instance_id = rcp.instance_id
        WHERE ri.instance_id IN ({placeholders})
        GROUP BY ri.instance_id
        """
        
        results = self.dao.db.execute(query, tuple(instance_ids), fetch=True)
        if not results:
            return {'error': 'No instances found'}
        
        df = pd.DataFrame(results)
        
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
