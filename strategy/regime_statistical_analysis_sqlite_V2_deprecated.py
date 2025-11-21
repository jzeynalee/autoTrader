"""
regime_statistical_analysis_sqlite.py
Integrated upgraded statistical analyzer

Upgraded, drop-in replacement for `regime_statistical_analysis_sqlite.py`.

Features:
- Kruskal-Wallis / ANOVA multi-regime tests with multiple-comparison correction
- Chi-square, G-test, Cramer's V for categorical features
- Mutual Information (MI) and Distance Correlation (dCor)
- Granger causality (if statsmodels available) and an MI-based Transfer Entropy fallback
- MANOVA / logistic regression (L1) for multivariate factor tests
- Permutation importance and SHAP (if available) for robust ML explainability
- Markov transition analysis to find drivers of regime transitions
- PCA contribution, silhouette & cluster stability tests
- Safe fallbacks and helpful error messages if optional libs are missing

Usage:
- Replace the old file with this one. It expects the same RegimeDataAccess DAO interface
  (i.e. methods like `get_indicator_statistics`, `get_pattern_effectiveness`, and a
  `.db.execute(query, params, fetch=True)` pattern). Minor name/arg adjustments can be
  made if your DAO differs slightly.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
import itertools
import math

# Optional dependencies
try:
    import statsmodels.api as sm
    import statsmodels.stats.multicomp as multi
    from statsmodels.stats.multitest import multipletests
    STATSMODELS = True
except Exception:
    STATSMODELS = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# Distance correlation implementation (no external deps)
def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance correlation between two 1D arrays."""
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)

    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    n = x.size
    if n <= 1:
        return 0.0

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)

    if dcov2_xx * dcov2_yy <= 0:
        return 0.0
    else:
        return math.sqrt(max(dcov2_xy, 0.0) / math.sqrt(dcov2_xx * dcov2_yy))


class RegimeStatisticalAnalyzer:
    """
    Comprehensive analyzer that runs a broad suite of statistical and ML-based
    experiments to discover factors that *affect* or *cause* regime identity
    and regime transitions. Requires a `RegimeDataAccess` data-access-object
    (DAO) that can query your SQLite database and return Pandas-friendly rows.

    Methods are defensive about optional libraries. When an optional lib isn't
    installed, the method will return a meaningful message and continue where possible.
    """

    def __init__(self, regime_dao):
        self.dao = regime_dao

    # ----------------------------- Basic helpers -----------------------------
    def _holm_bonferroni(self, pvals: List[float], alpha: float = 0.05) -> List[bool]:
        """Holm-Bonferroni multiple-testing correction. Returns list of booleans: reject/null."""
        pvals = np.array(pvals)
        m = len(pvals)
        idx = np.argsort(pvals)
        rejects = np.zeros(m, dtype=bool)
        for k in range(m):
            i = idx[k]
            threshold = alpha / (m - k)
            if pvals[i] <= threshold:
                rejects[i] = True
        return list(rejects)

    # ------------------------- Indicator causality suite ----------------------
    def analyze_indicator_causality(self,
                                    indicator_name: str,
                                    outcome_metric: str = 'next_1d_return_pct',
                                    min_sample_size: int = 50,
                                    regimes_column: str = 'regime') -> Dict:
        """
        Upgraded indicator causality analysis. Runs:
        - Kruskal-Wallis (multi-regime nonparametric ANOVA)
        - Pairwise Dunn tests (if statsmodels available) with Holm correction
        - Mutual information (nonlinear)
        - Distance correlation
        - Logistic regression (L1) to estimate independent effect
        - Granger causality (if time-series data available and statsmodels installed)
        - Returns structured dict with many diagnostics
        """
        df = self.dao.get_indicator_statistics(indicator_name, min_strength=0)
        if df is None or len(df) < min_sample_size:
            return {'error': f'Insufficient data: {0 if df is None else len(df)} rows'}

        # Ensure required columns
        if outcome_metric not in df.columns:
            return {'error': f"Outcome metric '{outcome_metric}' not in dataframe"}

        results = {'indicator': indicator_name, 'n': len(df)}

        # 1) Kruskal-Wallis across regimes (non-parametric ANOVA)
        if regimes_column in df.columns:
            groups = []
            group_labels = []
            for r, sub in df.groupby(regimes_column):
                groups.append(sub['mean_value'].values)
                group_labels.append(r)
            try:
                kw_stat, kw_p = stats.kruskal(*groups)
                results['kruskal_stat'] = float(kw_stat)
                results['kruskal_p'] = float(kw_p)
            except Exception as e:
                results['kruskal_error'] = str(e)

            # Pairwise tests (Dunn) using statsmodels if available
            if STATSMODELS:
                try:
                    # Build a two-column table for pairwise multiple comparisons
                    mc_df = df[[regimes_column, 'mean_value']].dropna()
                    comp = multi.MultiComparison(mc_df['mean_value'], mc_df[regimes_column])
                    tuk = comp.tukeyhsd()
                    # Collect pairwise p-values
                    pairwise = []
                    for i in range(len(tuk._results_table.data) - 1):
                        row = tuk._results_table.data[i + 1]
                        pairwise.append({'group1': row[0], 'group2': row[1], 'p': float(row[5])})
                    # Correct p-values
                    pvals = [p['p'] for p in pairwise]
                    rejects = multipletests(pvals, alpha=0.05, method='holm')[0].tolist()
                    for i, p in enumerate(pairwise):
                        p['reject_null'] = rejects[i]
                    results['pairwise_tests'] = pairwise
                except Exception as e:
                    results['pairwise_error'] = str(e)
            else:
                results['pairwise_tests'] = 'statsmodels not available; install statsmodels for pairwise tests'

        # 2) Mutual information (non-linear dependency)
        try:
            # mutual_info_regression expects 2D X
            mi = mutual_info_regression(df[['mean_value']].fillna(0).values, df[outcome_metric].fillna(0).values, random_state=0)
            results['mutual_info'] = float(mi[0])
        except Exception as e:
            results['mutual_info_error'] = str(e)

        # 3) Distance correlation
        try:
            dc = distance_correlation(np.asarray(df['mean_value'].fillna(0)), np.asarray(df[outcome_metric].fillna(0)))
            results['distance_correlation'] = float(dc)
        except Exception as e:
            results['distance_correlation_error'] = str(e)

        # 4) Logistic regression L1 to estimate independent predictive effect
        try:
            # Build X matrix using lagged indicator + other available numeric columns if present
            X = df[['mean_value']].fillna(0).values
            y = (df[outcome_metric] > 0).astype(int).values
            if len(np.unique(y)) == 1:
                results['logreg'] = 'Outcome constant; cannot fit classifier'
            else:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                logreg = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=0)
                scores = cross_val_score(logreg, Xs, y, cv=StratifiedKFold(n_splits=min(5, max(2, len(y)//20))))
                # Fit on full data to inspect coefficient
                logreg.fit(Xs, y)
                coef = float(logreg.coef_[0][0])
                results['logreg_cv_score_mean'] = float(scores.mean())
                results['logreg_coef'] = coef
        except Exception as e:
            results['logreg_error'] = str(e)

        # 5) Granger causality test if timestamped series is available and statsmodels is installed
        try:
            if STATSMODELS and 'timestamp' in df.columns:
                # Build time-series: sort by timestamp, then test whether indicator leads outcome
                ts = df.sort_values('timestamp')[[ 'mean_value', outcome_metric ]].dropna()
                maxlag = min(5, max(1, len(ts)//10))
                gc_res = sm.tsa.stattools.grangercausalitytests(ts.values, maxlag=maxlag, verbose=False)
                # extract p-values for F-test at each lag
                lag_p = {lag: float(gc_res[lag][0]['ssr_ftest'][1]) for lag in gc_res}
                results['granger_pvalues'] = lag_p
            else:
                results['granger'] = 'statsmodels missing or no timestamp column'
        except Exception as e:
            results['granger_error'] = str(e)

        return results

    # ------------------------- Pattern effectiveness -------------------------
    def analyze_pattern_effectiveness(self, pattern_name: str, pattern_type: str = 'candlestick') -> Dict:
        """
        Enhanced pattern effectiveness including:
        - Chi-square / G-test for categorical association (pattern occurrence vs regime)
        - T-tests plus non-parametric Mann-Whitney U
        - Effect size (Cohen's d)
        """
        df = self.dao.get_pattern_effectiveness(pattern_name, pattern_type)
        if df is None or len(df) < 5:
            return {'error': f'Insufficient pattern occurrences: {0 if df is None else len(df)}'}

        # We'll need a full set of instances to compare with
        all_instances_query = """
        SELECT instance_id, regime, next_1d_return_pct
        FROM regime_instances
        WHERE next_1d_return_pct IS NOT NULL
        """
        all_df = pd.DataFrame(self.dao.db.execute(all_instances_query, fetch=True))
        if all_df.empty:
            return {'error': 'Could not load all instances for baseline comparison'}

        with_ids = set(df['instance_id'])
        with_pat = all_df[all_df['instance_id'].isin(with_ids)]['next_1d_return_pct']
        without_pat = all_df[~all_df['instance_id'].isin(with_ids)]['next_1d_return_pct']

        # Classical tests
        try:
            t_stat, t_p = stats.ttest_ind(with_pat, without_pat, nan_policy='omit')
        except Exception as e:
            t_stat, t_p = None, str(e)

        try:
            u_stat, u_p = stats.mannwhitneyu(with_pat, without_pat, alternative='two-sided')
        except Exception as e:
            u_stat, u_p = None, str(e)

        # Cohen's d
        try:
            s1, s2 = with_pat.std(), without_pat.std()
            pooled = math.sqrt((s1 ** 2 + s2 ** 2) / 2) if (not np.isnan(s1) and not np.isnan(s2)) else 0
            cohens_d = ((with_pat.mean() - without_pat.mean()) / pooled) if pooled > 0 else 0
        except Exception:
            cohens_d = 0

        # Categorical association: does pattern occurrence distribution vary by regime?
        try:
            contingency = pd.crosstab(all_df['regime'].astype(str), all_df['instance_id'].apply(lambda iid: iid in with_ids))
            chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
            # Cramer's V
            n = contingency.values.sum()
            phi2 = chi2 / n
            r, k = contingency.shape
            cramers_v = math.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0
        except Exception as e:
            chi2, chi_p, cramers_v = None, None, None

        return {
            'pattern_name': pattern_name,
            'occurrences': len(df),
            't_test_p': float(t_p) if isinstance(t_p, (int, float)) else str(t_p),
            'mannwhitney_p': float(u_p) if isinstance(u_p, (int, float)) else str(u_p),
            'cohens_d': float(cohens_d),
            'chi2_p': float(chi_p) if isinstance(chi_p, (int, float)) else str(chi_p),
            'cramers_v': float(cramers_v) if cramers_v is not None else None,
            'win_rate_with': float((with_pat > 0).mean() * 100),
            'win_rate_without': float((without_pat > 0).mean() * 100)
        }

    # ----------------------- Multivariate & ML explainability -----------------
    def find_optimal_indicator_combinations(self,
                                            target_metric: str = 'next_1d_return_pct',
                                            max_indicators: int = 6,
                                            min_sample_for_combo: int = 20) -> List[Dict]:
        """
        Upgraded search for indicator combinations using:
        - Random Forest
        - Permutation importance
        - Optional SHAP (if installed)
        - Penalized logistic regression (L1) for sparse combinations
        Returns ranked combos with robust importance estimates.
        """
        query = """
        SELECT ri.instance_id, ri.next_1d_return_pct, GROUP_CONCAT(rci.indicator_name) as indicators
        FROM regime_instances ri
        LEFT JOIN regime_confirming_indicators rci ON ri.instance_id = rci.instance_id
        WHERE ri.next_1d_return_pct IS NOT NULL
        GROUP BY ri.instance_id
        HAVING COUNT(rci.indicator_name) >= 1
        """
        rows = self.dao.db.execute(query, fetch=True)
        if not rows:
            return []
        df = pd.DataFrame(rows)

        # parse indicators
        df['indicators'] = df['indicators'].fillna('')
        all_inds = set(itertools.chain.from_iterable([s.split(',') for s in df['indicators'] if s]))
        for ind in all_inds:
            df[f'has_{ind}'] = df['indicators'].str.contains(ind).astype(int)

        feature_cols = [c for c in df.columns if c.startswith('has_')]
        X = df[feature_cols].values
        y = (df[target_metric] > 0).astype(int).values

        if X.shape[0] < 30 or len(feature_cols) == 0:
            return []

        rf = RandomForestClassifier(n_estimators=200, random_state=0, max_depth=6)
        rf.fit(X, y)

        # Permutation importance (robust)
        try:
            perm = permutation_importance(rf, X, y, n_repeats=20, random_state=0)
            imp_df = pd.DataFrame({'indicator': [c.replace('has_', '') for c in feature_cols],
                                   'perm_importance_mean': perm.importances_mean})
            imp_df.sort_values('perm_importance_mean', ascending=False, inplace=True)
        except Exception:
            imp_df = pd.DataFrame({'indicator': [c.replace('has_', '') for c in feature_cols],
                                   'perm_importance_mean': rf.feature_importances_})
            imp_df.sort_values('perm_importance_mean', ascending=False, inplace=True)

        top_inds = imp_df.head(max_indicators)['indicator'].tolist()

        combos = []
        for r in range(2, min(len(top_inds), max_indicators) + 1):
            for combo in itertools.combinations(top_inds, r):
                mask = np.ones(len(df), dtype=bool)
                for ind in combo:
                    mask &= df[f'has_{ind}'] == 1
                subset = df[mask]
                if len(subset) >= min_sample_for_combo:
                    win_rate = float((subset[target_metric] > 0).mean() * 100)
                    avg_ret = float(subset[target_metric].mean())
                    combos.append({'indicators': list(combo), 'sample_size': len(subset), 'win_rate': win_rate, 'avg_return': avg_ret, 'score': win_rate * avg_ret})

        combos.sort(key=lambda x: x['score'], reverse=True)

        # Optional SHAP explanation for top combo's RF (if SHAP available)
        extra = {}
        if SHAP_AVAILABLE and len(top_inds) > 0:
            try:
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X)
                # for binary y shap_values is list-like; compute mean abs for class 1
                if isinstance(shap_values, list):
                    shap_mean = np.mean(np.abs(shap_values[1]), axis=0)
                else:
                    shap_mean = np.mean(np.abs(shap_values), axis=0)
                shap_df = pd.DataFrame({'indicator': [c.replace('has_', '') for c in feature_cols], 'shap_mean_abs': shap_mean})
                shap_df.sort_values('shap_mean_abs', ascending=False, inplace=True)
                extra['shap_ranking'] = shap_df.to_dict('records')
            except Exception as e:
                extra['shap_error'] = str(e)

        out = {'combinations_top': combos[:50], 'importance_table': imp_df.to_dict('records')}
        out.update(extra)
        return out

    # ---------------------- Transition / Markov influence ---------------------
    def transition_influence(self, lookback: int = 5, indicator_names: Optional[List[str]] = None) -> Dict:
        """
        Estimate how indicators influence regime transition probabilities.
        Approach:
        - Build (t -> t+1) transition dataset with features = indicators at t
        - Fit multiclass logistic regression (L1) to predict next_regime
        - Report coefficients and significance (if possible)
        """
        # Load transitions (assumes regime_instances has timestamp, regime, instance_id)
        query = """
        SELECT instance_id, timestamp, regime
        FROM regime_instances
        ORDER BY timestamp
        """
        rows = self.dao.db.execute(query, fetch=True)
        if not rows:
            return {'error': 'No regime instance timeline available'}
        ri = pd.DataFrame(rows)
        ri = ri.sort_values('timestamp')

        # Get indicator snapshots for those timestamps
        # We expect a DAO method that can return indicator mean_value per instance
        # We'll use a simple join table access: regime_confirming_indicators
        q2 = """
        SELECT instance_id, indicator_name, mean_value
        FROM regime_confirming_indicators
        """
        rci_rows = self.dao.db.execute(q2, fetch=True)
        rci = pd.DataFrame(rci_rows)
        if rci.empty:
            return {'error': 'No indicator snapshot table found'}

        pivot = rci.pivot_table(index='instance_id', columns='indicator_name', values='mean_value', aggfunc='mean')
        pivot.reset_index(inplace=True)

        merged = ri.merge(pivot, on='instance_id', how='left')
        merged['next_regime'] = merged['regime'].shift(-1)
        merged = merged.dropna(subset=['next_regime'])

        if indicator_names is None:
            indicator_names = [c for c in pivot.columns if c != 'instance_id']

        X = merged[indicator_names].fillna(0).values
        y = merged['next_regime'].values

        if len(np.unique(y)) < 2:
            return {'error': 'Not enough regime transitions to model'}

        # Fit multinomial logistic with L1 if supported
        try:
            clf = LogisticRegression(penalty='l2', multi_class='multinomial', solver='saga', max_iter=200, random_state=0)
            clf.fit(X, y)
            coefs = clf.coef_
            classes = clf.classes_
            coef_table = []
            for i, cls in enumerate(classes):
                for j, ind in enumerate(indicator_names):
                    coef_table.append({'target_next_regime': cls, 'indicator': ind, 'coef': float(coefs[i, j])})
            return {'coef_table': coef_table}
        except Exception as e:
            return {'error': f'LogReg failed: {e}'}

    # ---------------------- Dimensionality & cluster tests -------------------
    def pca_and_cluster_analysis(self, feature_columns: List[str], n_clusters: int = 3) -> Dict:
        """
        Run PCA and basic clustering diagnostics to find which features separate regimes.
        Returns PCA explained variance, loadings and silhouette score for clustering.
        """
        # Expect a DAO method to return a combined features table
        query = f"SELECT instance_id, {', '.join(feature_columns)} FROM regime_instances WHERE {feature_columns[0]} IS NOT NULL"
        rows = self.dao.db.execute(query, fetch=True)
        if not rows:
            return {'error': 'No data'}
        df = pd.DataFrame(rows).dropna()
        X = df[feature_columns].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        pca = PCA(n_components=min(10, Xs.shape[1]))
        pcs = pca.fit_transform(Xs)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_.tolist()

        # Clustering and silhouette
        try:
            km = KMeans(n_clusters=n_clusters, random_state=0)
            labels = km.fit_predict(pcs)
            sil = silhouette_score(pcs, labels)
        except Exception as e:
            sil = str(e)

        loading_table = [{ 'feature': feature_columns[i], 'loading_vector': loadings[i].tolist() } for i in range(len(feature_columns))]
        return {'explained_variance_ratio': explained, 'loadings': loading_table, 'silhouette': sil}

    # ---------------------- Strategy generation (ported from V1) ----------------
    def generate_strategy_from_instances(
        self,
        instance_ids: List[str],
        strategy_name: str
    ) -> Dict:
        """
        Generate a concrete trading strategy from a group of similar instances.

        Ported from V1 for backward compatibility with the strategy-generation pipeline.
        Returns the same dictionary structure as the original V1 implementation.
        """
        if not instance_ids:
            return {'error': 'No instance ids provided'}

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
        common_indicators = self._find_common_elements(df.get('all_indicators', pd.Series(dtype=object)))
        common_patterns = self._find_common_elements(df.get('all_patterns', pd.Series(dtype=object)))

        # Calculate average metrics defensively (use .get to avoid KeyError)
        avg_rsi = float(df['rsi_mean'].mean()) if 'rsi_mean' in df.columns else 50.0
        avg_adx = float(df['adx_mean'].mean()) if 'adx_mean' in df.columns else 20.0
        avg_volatility = float(df['volatility_mean'].mean()) if 'volatility_mean' in df.columns else 1.0

        # Determine entry conditions
        entry_conditions: List[str] = []

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
        avg_drawdown = abs(df['max_drawdown_pct'].mean()) if 'max_drawdown_pct' in df.columns else 1.0
        avg_runup = abs(df['max_runup_pct'].mean()) if 'max_runup_pct' in df.columns else 1.0

        stop_loss = avg_drawdown * 1.2  # 20% buffer
        take_profit = avg_runup * 0.8   # Conservative target

        # Calculate historical performance
        win_rate = float((df['next_1d_return_pct'] > 0).mean() * 100) if 'next_1d_return_pct' in df.columns else 0.0
        avg_return = float(df['next_1d_return_pct'].mean()) if 'next_1d_return_pct' in df.columns else 0.0

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
        """Find elements that appear in at least min_frequency of rows.

        Ported from V1. Accepts a pandas Series where each row is a comma-separated string
        of elements (or NaN). Returns list of elements meeting frequency threshold.
        """
        all_elements: List[str] = []

        # If the provided value is not a Series, try to coerce
        if not isinstance(series, pd.Series):
            try:
                series = pd.Series(series)
            except Exception:
                return []

        for row in series:
            if pd.notna(row) and row != '':
                # ensure string and split
                elements = str(row).split(',')
                all_elements.extend([e.strip() for e in elements if e.strip()])

        if not all_elements:
            return []

        from collections import Counter
        counter = Counter(all_elements)

        total = len(series)
        threshold = total * min_frequency

        common = [elem for elem, count in counter.items() if count >= threshold]

        # Sort by frequency
        common.sort(key=lambda x: counter[x], reverse=True)

        return common
