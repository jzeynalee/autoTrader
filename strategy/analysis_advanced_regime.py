"""
analysis_advanced_regime.py - REFACTORED
Hybrid ZigZag-Price Action Regime Detection Framework

This module implements a comprehensive regime detection system that:
1. Extracts ZigZag swing points for structural consistency
2. Builds a Hybrid Swing Registry with rich price action metadata
3. Uses GaussianHMM for unsupervised regime classification
4. Uses XGBoost for supervised regime prediction
5. Provides adaptive strategy parameters per regime

Architecture:
    OHLCV Data → ZigZag Swing Extraction → Hybrid Swing Registry
    → Feature Engineering Layer (price action, volume, structure)
    → GaussianHMM (Unsupervised Regime Detection)
    → XGBoost (Supervised Regime Prediction)
    → Strategy Adaptation Framework
"""

import pandas as pd
import numpy as np
import uuid
import warnings
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ML Libraries
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. Install with: pip install hmmlearn")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not available. Install with: pip install xgboost")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# SECTION 1: DATA STRUCTURES
# ============================================================================

@dataclass
class SwingPoint:
    """
    Canonical representation of a swing point in the Hybrid Swing Registry.
    
    Each swing encapsulates:
    - Core ZigZag fields: type, magnitude, duration, ATR context
    - Trend features: slope, return gradient, directional persistence
    - Volatility features: ATR ratios, BB/KC width, normalized variance
    - Momentum features: RSI, ROC, PPO, MACD
    - Volume features: volume ROC, OBV, volume z-score
    - Structure features: HH/HL/LH/LL, structure breaks, pullback depth
    - Candle context: recent patterns (engulfing, doji, etc.)
    - Price action anomalies: gaps, extended bars, volume spikes
    """
    # Core identification
    timestamp: pd.Timestamp
    index: int
    swing_type: str  # 'high' or 'low'
    price: float
    
    # ZigZag structural fields
    magnitude_pct: float = 0.0
    duration: int = 0
    atr_context: float = 0.0
    
    # Trend features
    local_slope: float = 0.0
    return_gradient: float = 0.0
    directional_persistence: float = 0.0
    
    # Volatility features
    atr_ratio: float = 1.0
    bb_width: float = 0.0
    kc_width: float = 0.0
    normalized_variance: float = 0.0
    
    # Momentum features
    rsi: float = 50.0
    roc: float = 0.0
    ppo: float = 0.0
    macd_hist: float = 0.0
    adx: float = 0.0
    
    # Volume features
    volume_roc: float = 0.0
    obv_change: float = 0.0
    volume_zscore: float = 0.0
    
    # Structure features
    structure_type: str = 'unknown'  # HH, HL, LH, LL
    structure_break: bool = False
    pullback_depth: float = 0.0
    
    # Candle context (recent patterns)
    recent_bullish_patterns: int = 0
    recent_bearish_patterns: int = 0
    recent_doji: int = 0
    
    # Price action anomalies
    gap_intensity: float = 0.0
    extended_bar_ratio: float = 0.0
    volume_spike: bool = False
    
    # Regime labels (filled by HMM/XGBoost)
    hmm_regime: Optional[int] = None
    hmm_probability: Optional[float] = None
    predicted_regime: Optional[int] = None
    prediction_confidence: Optional[float] = None


@dataclass
class RegimeState:
    """
    Represents a detected market regime with its characteristics.
    """
    regime_id: int
    name: str
    trend_direction: str  # 'bull', 'bear', 'neutral'
    volatility_level: str  # 'low', 'high'
    
    # Statistical characteristics (from HMM means)
    mean_atr_ratio: float = 1.0
    mean_rsi: float = 50.0
    mean_volume_zscore: float = 0.0
    mean_slope: float = 0.0
    
    # Transition probabilities
    persistence_probability: float = 0.0
    transition_probabilities: Dict[int, float] = field(default_factory=dict)
    
    # Strategy parameters
    position_sizing: str = 'normal'
    risk_multiplier: float = 1.0
    stop_loss_type: str = 'standard'
    take_profit_ratio: float = 2.0
    stop_width_atr: float = 2.0


# ============================================================================
# SECTION 2: ZIGZAG SWING EXTRACTION
# ============================================================================

class ZigZagSwingExtractor:
    """
    Extracts structural swing points using ZigZag algorithm with ATR-based threshold.
    Ensures cross-timeframe consistency and structural uniformity.
    """
    
    def __init__(self, atr_multiplier: float = 2.0, min_bars: int = 5):
        """
        Args:
            atr_multiplier: Minimum swing size as multiple of ATR
            min_bars: Minimum bars between swings
        """
        self.atr_multiplier = atr_multiplier
        self.min_bars = min_bars

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr

    
    def extract_swings(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract ZigZag swing points from OHLC data.
        
        Returns:
            List of swing dictionaries with basic structural info
        """
        if len(df) < 20:
            return []
        
        # Calculate ATR for threshold
        atr = self._calculate_atr(df, period=14)
        threshold = atr.iloc[-1] * self.atr_multiplier if len(atr) > 0 else df['close'].std() * 0.02
        
        swings = []
        last_swing_idx = 0
        last_swing_price = df['close'].iloc[0]
        last_swing_type = None
        
        for i in range(self.min_bars, len(df)):
            current_price = df['close'].iloc[i]
            price_change = abs(current_price - last_swing_price)
            
            # Check for significant move
            if price_change >= threshold and (i - last_swing_idx) >= self.min_bars:
                # Determine if this is a high or low swing
                if current_price > last_swing_price and last_swing_type != 'high':
                    # New swing high
                    swings.append({
                        'timestamp': df.index[i],
                        'index': i,
                        'swing_type': 'high',
                        'price': df['high'].iloc[i],
                        'magnitude_pct': ((current_price - last_swing_price) / last_swing_price) * 100,
                        'duration': i - last_swing_idx,
                        'atr_context': atr.iloc[i] if i < len(atr) else threshold
                    })
                    last_swing_type = 'high'
                    last_swing_price = current_price
                    last_swing_idx = i
                    
                elif current_price < last_swing_price and last_swing_type != 'low':
                    # New swing low
                    swings.append({
                        'timestamp': df.index[i],
                        'index': i,
                        'swing_type': 'low',
                        'price': df['low'].iloc[i],
                        'magnitude_pct': ((last_swing_price - current_price) / last_swing_price) * 100,
                        'duration': i - last_swing_idx,
                        'atr_context': atr.iloc[i] if i < len(atr) else threshold
                    })
                    last_swing_type = 'low'
                    last_swing_price = current_price
                    last_swing_idx = i
        
        return swings


# ============================================================================
# SECTION 3: HYBRID SWING REGISTRY
# ============================================================================

class HybridSwingRegistry:
    """
    Consolidates ZigZag-derived swing points and augments them with rich
    price action and market microstructure metadata.
    
    This is the canonical data layer for regime detection.
    """
    
    def __init__(self):
        self.swings: List[SwingPoint] = []
        self.feature_cache = {}
    
    def build_from_dataframe(self, df: pd.DataFrame, zigzag_swings: List[Dict]) -> None:
        """
        Build registry by enriching ZigZag swings with price action features.
        
        Args:
            df: Full OHLCV dataframe with indicators
            zigzag_swings: Output from ZigZagSwingExtractor
        """
        self.swings = []
        
        for swing_dict in zigzag_swings:
            idx = swing_dict['index']
            
            # Create base swing point
            swing = SwingPoint(
                timestamp=swing_dict['timestamp'],
                index=idx,
                swing_type=swing_dict['swing_type'],
                price=swing_dict['price'],
                magnitude_pct=swing_dict['magnitude_pct'],
                duration=swing_dict['duration'],
                atr_context=swing_dict['atr_context']
            )
            
            # Enrich with features from dataframe
            self._enrich_swing_with_features(swing, df, idx)
            
            self.swings.append(swing)
    
    '''def _enrich_swing_with_features(self, swing: SwingPoint, df: pd.DataFrame, idx: int) -> None:
        """
        Augment swing with all price action, volume, structure, and momentum features.
        """
        if idx >= len(df):
            return
        
        # === TREND FEATURES ===
        swing.local_slope = self._calculate_local_slope(df, idx, window=10)
        swing.return_gradient = self._calculate_return_gradient(df, idx, window=5)
        swing.directional_persistence = self._calculate_directional_persistence(df, idx, window=10)
        
        # === VOLATILITY FEATURES ===
        swing.atr_ratio = self._get_safe_value(df, idx, 'atr', default=1.0) / df['close'].iloc[idx]
        swing.bb_width = self._get_safe_value(df, idx, 'bb_width', default=0.0)
        swing.kc_width = self._get_safe_value(df, idx, 'kc_width', default=0.0)
        swing.normalized_variance = self._calculate_normalized_variance(df, idx, window=20)
        
        # === MOMENTUM FEATURES ===
        swing.rsi = self._get_safe_value(df, idx, 'rsi', default=50.0)
        swing.roc = self._get_safe_value(df, idx, 'roc', default=0.0)
        swing.ppo = self._get_safe_value(df, idx, 'ppo', default=0.0)
        swing.macd_hist = self._get_safe_value(df, idx, 'macd_hist', default=0.0)
        swing.adx = self._get_safe_value(df, idx, 'adx', default=0.0)
        
        # === VOLUME FEATURES ===
        swing.volume_roc = self._calculate_volume_roc(df, idx, window=5)
        swing.obv_change = self._calculate_obv_change(df, idx)
        swing.volume_zscore = self._calculate_volume_zscore(df, idx, window=20)
        
        # === STRUCTURE FEATURES ===
        swing.structure_type = self._classify_swing_structure(swing, df, idx)
        swing.structure_break = self._detect_structure_break(df, idx)
        swing.pullback_depth = self._get_safe_value(df, idx, 'pullback_from_high_pct', default=0.0)
        
        # === CANDLE CONTEXT ===
        swing.recent_bullish_patterns = self._count_recent_patterns(df, idx, pattern_type='bullish')
        swing.recent_bearish_patterns = self._count_recent_patterns(df, idx, pattern_type='bearish')
        swing.recent_doji = self._count_recent_patterns(df, idx, pattern_type='doji')
        
        # === PRICE ACTION ANOMALIES ===
        swing.gap_intensity = self._calculate_gap_intensity(df, idx)
        swing.extended_bar_ratio = self._calculate_extended_bar_ratio(df, idx)
        swing.volume_spike = self._detect_volume_spike(df, idx)'''
    
    def _enrich_swing_with_features(self, swing: SwingPoint, df: pd.DataFrame, idx: int) -> None:
        """
        Augment swing with all price action, volume, structure, and momentum features.
        This now READS pre-calculated bar-by-bar features from the dataframe
        and calculates ONLY swing-specific features.
        """
        if idx >= len(df):
            return
        
        # === TREND FEATURES (Read from df) ===
        swing.local_slope = self._get_safe_value(df, idx, 'local_slope', default=0.0)
        swing.return_gradient = self._get_safe_value(df, idx, 'return_gradient', default=0.0)
        swing.directional_persistence = self._get_safe_value(df, idx, 'directional_persistence', default=0.0)
        
        # === VOLATILITY FEATURES (Read from df) ===
        swing.atr_ratio = self._get_safe_value(df, idx, 'atr_ratio', default=1.0)
        swing.bb_width = self._get_safe_value(df, idx, 'bb_width', default=0.0)
        swing.kc_width = self._get_safe_value(df, idx, 'kc_width', default=0.0)
        swing.normalized_variance = self._get_safe_value(df, idx, 'normalized_variance', default=0.0)
        
        # === MOMENTUM FEATURES (Read from df) ===
        swing.rsi = self._get_safe_value(df, idx, 'rsi', default=50.0)
        swing.roc = self._get_safe_value(df, idx, 'roc', default=0.0)
        swing.ppo = self._get_safe_value(df, idx, 'ppo', default=0.0)
        swing.macd_hist = self._get_safe_value(df, idx, 'macd_hist', default=0.0)
        swing.adx = self._get_safe_value(df, idx, 'adx', default=0.0)
        
        # === VOLUME FEATURES (Read from df) ===
        swing.volume_roc = self._get_safe_value(df, idx, 'volume_roc', default=0.0)
        swing.obv_change = self._get_safe_value(df, idx, 'obv_change', default=0.0)
        swing.volume_zscore = self._get_safe_value(df, idx, 'volume_zscore', default=0.0)
        
        # === STRUCTURE FEATURES (Calculated here) ===
        swing.structure_type = self._classify_swing_structure(swing, df, idx)
        swing.structure_break = self._detect_structure_break(df, idx)
        swing.pullback_depth = self._get_safe_value(df, idx, 'pullback_from_high_pct', default=0.0)
        
        # === CANDLE CONTEXT (Read from df) ===
        swing.recent_bullish_patterns = self._get_safe_value(df, idx, 'recent_bullish_patterns', default=0)
        swing.recent_bearish_patterns = self._get_safe_value(df, idx, 'recent_bearish_patterns', default=0)
        swing.recent_doji = self._get_safe_value(df, idx, 'recent_doji', default=0)
        
        # === PRICE ACTION ANOMALIES (Read from df) ===
        swing.gap_intensity = self._get_safe_value(df, idx, 'gap_intensity', default=0.0)
        swing.extended_bar_ratio = self._get_safe_value(df, idx, 'extended_bar_ratio', default=1.0)
        swing.volume_spike = bool(self._get_safe_value(df, idx, 'volume_spike', default=0))
        
    
    # ========== Helper Methods ==========
    
    def _get_safe_value(self, df: pd.DataFrame, idx: int, column: str, default: float) -> float:
        """Safely extract value from dataframe with fallback."""
        if column not in df.columns or idx >= len(df):
            return default
        val = df[column].iloc[idx]
        return float(val) if pd.notna(val) else default
    
    '''def _calculate_local_slope(self, df: pd.DataFrame, idx: int, window: int) -> float:
        """Calculate local price slope using linear regression."""
        if idx < window:
            return 0.0
        
        prices = df['close'].iloc[idx-window:idx+1].values
        x = np.arange(len(prices))
        
        if len(prices) < 2:
            return 0.0
        
        # Simple slope calculation
        slope = (prices[-1] - prices[0]) / len(prices)
        return slope / df['close'].iloc[idx]  # Normalize by price
    
    def _calculate_return_gradient(self, df: pd.DataFrame, idx: int, window: int) -> float:
        """Calculate rate of change of returns."""
        if idx < window + 1:
            return 0.0
        
        returns = df['close'].pct_change().iloc[idx-window:idx+1]
        if len(returns) < 2:
            return 0.0
        
        return returns.diff().mean()
    
    def _calculate_directional_persistence(self, df: pd.DataFrame, idx: int, window: int) -> float:
        """Measure how consistently price moves in one direction."""
        if idx < window:
            return 0.0
        
        returns = df['close'].pct_change().iloc[idx-window:idx+1]
        positive_days = (returns > 0).sum()
        
        return (positive_days / window) - 0.5  # -0.5 to 0.5 range, 0 = neutral
    
    def _calculate_normalized_variance(self, df: pd.DataFrame, idx: int, window: int) -> float:
        """Calculate normalized price variance."""
        if idx < window:
            return 0.0
        
        returns = df['close'].pct_change().iloc[idx-window:idx+1]
        return returns.std() if len(returns) > 1 else 0.0
    
    def _calculate_volume_roc(self, df: pd.DataFrame, idx: int, window: int) -> float:
        """Volume rate of change."""
        if idx < window or 'volume' not in df.columns:
            return 0.0
        
        current_vol = df['volume'].iloc[idx]
        past_vol = df['volume'].iloc[idx-window]
        
        if past_vol == 0:
            return 0.0
        
        return (current_vol - past_vol) / past_vol
    
    def _calculate_obv_change(self, df: pd.DataFrame, idx: int) -> float:
        """On-Balance Volume change."""
        if 'obv' not in df.columns or idx < 1:
            return 0.0
        
        return df['obv'].iloc[idx] - df['obv'].iloc[idx-1]
    
    def _calculate_volume_zscore(self, df: pd.DataFrame, idx: int, window: int) -> float:
        """Calculate volume z-score."""
        if idx < window or 'volume' not in df.columns:
            return 0.0
        
        vol_window = df['volume'].iloc[idx-window:idx+1]
        mean_vol = vol_window.mean()
        std_vol = vol_window.std()
        
        if std_vol == 0:
            return 0.0
        
        return (df['volume'].iloc[idx] - mean_vol) / std_vol'''
    
    def _classify_swing_structure(self, swing: SwingPoint, df: pd.DataFrame, idx: int) -> str:
        """Classify swing as HH, HL, LH, or LL."""
        if len(self.swings) < 2:
            return 'unknown'
        
        prev_swing = self.swings[-1]
        
        if swing.swing_type == 'high':
            if swing.price > prev_swing.price:
                return 'HH'  # Higher High
            else:
                return 'LH'  # Lower High
        else:  # swing_type == 'low'
            if swing.price > prev_swing.price:
                return 'HL'  # Higher Low
            else:
                return 'LL'  # Lower Low
    
    def _detect_structure_break(self, df: pd.DataFrame, idx: int) -> bool:
        """Detect if current bar breaks recent structure."""
        if idx < 20:
            return False
        
        # Check for structure break indicators from features
        bullish_break = self._get_safe_value(df, idx, 'structure_break_bullish', default=0.0) == 1
        bearish_break = self._get_safe_value(df, idx, 'structure_break_bearish', default=0.0) == 1
        
        return bullish_break or bearish_break
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert registry to pandas DataFrame for ML."""
        records = []
        
        for swing in self.swings:
            record = {
                'timestamp': swing.timestamp,
                'index': swing.index,
                'swing_type': swing.swing_type,
                'price': swing.price,
                'magnitude_pct': swing.magnitude_pct,
                'duration': swing.duration,
                'atr_context': swing.atr_context,
                'local_slope': swing.local_slope,
                'return_gradient': swing.return_gradient,
                'directional_persistence': swing.directional_persistence,
                'atr_ratio': swing.atr_ratio,
                'bb_width': swing.bb_width,
                'kc_width': swing.kc_width,
                'normalized_variance': swing.normalized_variance,
                'rsi': swing.rsi,
                'roc': swing.roc,
                'ppo': swing.ppo,
                'macd_hist': swing.macd_hist,
                'adx': swing.adx,
                'volume_roc': swing.volume_roc,
                'obv_change': swing.obv_change,
                'volume_zscore': swing.volume_zscore,
                'structure_type': swing.structure_type,
                'structure_break': int(swing.structure_break),
                'pullback_depth': swing.pullback_depth,
                'recent_bullish_patterns': swing.recent_bullish_patterns,
                'recent_bearish_patterns': swing.recent_bearish_patterns,
                'recent_doji': swing.recent_doji,
                'gap_intensity': swing.gap_intensity,
                'extended_bar_ratio': swing.extended_bar_ratio,
                'volume_spike': int(swing.volume_spike),
                'hmm_regime': swing.hmm_regime,
                'hmm_probability': swing.hmm_probability,
                'predicted_regime': swing.predicted_regime,
                'prediction_confidence': swing.prediction_confidence,
                # NEW: Regime instance metadata
                'regime_instance_id': getattr(swing, 'regime_instance_id', None),
                'regime_instance_index': getattr(swing, 'regime_instance_index', None)
            }
            records.append(record)
        
        return pd.DataFrame(records)



# ============================================================================
# SECTION 4: GAUSSIAN HMM REGIME CLASSIFIER
# ============================================================================

class GaussianHMMRegimeClassifier:
    """
    Unsupervised regime detection using Gaussian Hidden Markov Model.
    
    Identifies 6 latent market states:
    0: Low-Volatility Bull Trend
    1: High-Volatility Bull Trend
    2: Low-Volatility Bear Trend
    3: High-Volatility Bear Trend
    4: Low-Volatility Range (Neutral)
    5: High-Volatility Chop (Turbulent)
    """
    
    def __init__(self, n_regimes: int = 6, random_state: int = 42):
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not available. Install with: pip install hmmlearn")
        
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_columns = []
        self.regime_states: Dict[int, RegimeState] = {}
    
    def fit(self, swing_df: pd.DataFrame) -> None:
        """
        Fit HMM model on swing registry features.
        
        Args:
            swing_df: DataFrame from HybridSwingRegistry.to_dataframe()
        """
        # Select continuous features for HMM
        self.feature_columns = [
            'magnitude_pct', 'duration', 'local_slope', 'return_gradient',
            'directional_persistence', 'atr_ratio', 'bb_width', 'kc_width',
            'normalized_variance', 'rsi', 'roc', 'ppo', 'macd_hist', 'adx',
            'volume_roc', 'obv_change', 'volume_zscore'
        ]
        
        # Extract and prepare features
        X = swing_df[self.feature_columns].fillna(0).values
        
        # Standardize features
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Fit HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled)
        
        # Build regime state objects
        self._build_regime_states()
        
        print(f"✅ HMM trained with {self.n_regimes} regimes")
        print(f"   Converged: {self.model.monitor_.converged}")
    
    def predict(self, swing_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime labels and probabilities for swings.
        
        Returns:
            regime_labels: Array of regime IDs (0-5)
            probabilities: Array of confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Extract and scale features
        X = swing_df[self.feature_columns].fillna(0).values
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict regimes
        regime_labels = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled).max(axis=1)
        
        return regime_labels, probabilities
    
    def _build_regime_states(self) -> None:
        """Build RegimeState objects from HMM parameters."""
        if self.model is None:
            return
        
        # Extract HMM parameters
        means = self.model.means_
        transmat = self.model.transmat_
        
        # Map feature indices
        feat_map = {name: idx for idx, name in enumerate(self.feature_columns)}
        
        for regime_id in range(self.n_regimes):
            regime_means = means[regime_id]
            
            # Extract key statistics
            slope_idx = feat_map.get('local_slope', 0)
            atr_idx = feat_map.get('atr_ratio', 0)
            rsi_idx = feat_map.get('rsi', 0)
            vol_z_idx = feat_map.get('volume_zscore', 0)
            
            mean_slope = regime_means[slope_idx]
            mean_atr = regime_means[atr_idx]
            mean_rsi = regime_means[rsi_idx]
            mean_vol_z = regime_means[vol_z_idx]
            
            # Classify regime
            is_bullish = mean_slope > 0.1 or mean_rsi > 55
            is_bearish = mean_slope < -0.1 or mean_rsi < 45
            is_high_vol = mean_atr > 0.02  # 2% ATR threshold
            
            if is_bullish and not is_high_vol:
                name = "Low-Vol Bull Trend"
                name = f"{name} (R{regime_id})" 
                trend = "bull"
                volatility = "low"
            elif is_bullish and is_high_vol:
                name = "High-Vol Bull Trend"
                name = f"{name} (R{regime_id})" 
                trend = "bull"
                volatility = "high"
            elif is_bearish and not is_high_vol:
                name = "Low-Vol Bear Trend"
                name = f"{name} (R{regime_id})" 
                trend = "bear"
                volatility = "low"
            elif is_bearish and is_high_vol:
                name = "High-Vol Bear Trend"
                name = f"{name} (R{regime_id})" 
                trend = "bear"
                volatility = "high"
            elif is_high_vol:
                name = "High-Vol Chop"
                name = f"{name} (R{regime_id})" 
                trend = "neutral"
                volatility = "high"
            else:
                name = "Low-Vol Range"
                name = f"{name} (R{regime_id})" 
                trend = "neutral"
                volatility = "low"
            
            # Build regime state
            regime = RegimeState(
                regime_id=regime_id,
                name=name,
                trend_direction=trend,
                volatility_level=volatility,
                mean_atr_ratio=mean_atr,
                mean_rsi=mean_rsi,
                mean_volume_zscore=mean_vol_z,
                mean_slope=mean_slope,
                persistence_probability=transmat[regime_id, regime_id]
            )
            
            # Store transition probabilities
            for next_regime in range(self.n_regimes):
                regime.transition_probabilities[next_regime] = transmat[regime_id, next_regime]
            
            # Assign strategy parameters
            self._assign_strategy_parameters(regime)
            
            self.regime_states[regime_id] = regime
    
    def _assign_strategy_parameters(self, regime: RegimeState) -> None:
        """Assign adaptive strategy parameters based on regime characteristics."""
        trend = regime.trend_direction
        vol = regime.volatility_level
        
        if trend == 'bull' and vol == 'low':
            regime.position_sizing = 'normal'
            regime.risk_multiplier = 1.0
            regime.stop_loss_type = 'tight'
            regime.take_profit_ratio = 2.5
            regime.stop_width_atr = 1.5
            
        elif trend == 'bull' and vol == 'high':
            regime.position_sizing = 'reduced'
            regime.risk_multiplier = 0.7
            regime.stop_loss_type = 'wide'
            regime.take_profit_ratio = 3.0
            regime.stop_width_atr = 3.0
            
        elif trend == 'bear' and vol == 'low':
            regime.position_sizing = 'normal'
            regime.risk_multiplier = 1.0
            regime.stop_loss_type = 'tight'
            regime.take_profit_ratio = 2.5
            regime.stop_width_atr = 1.5
            
        elif trend == 'bear' and vol == 'high':
            regime.position_sizing = 'minimal'
            regime.risk_multiplier = 0.5
            regime.stop_loss_type = 'wide'
            regime.take_profit_ratio = 3.0
            regime.stop_width_atr = 3.0
            
        elif trend == 'neutral' and vol == 'low':
            regime.position_sizing = 'normal'
            regime.risk_multiplier = 1.0
            regime.stop_loss_type = 'range_based'
            regime.take_profit_ratio = 1.5
            regime.stop_width_atr = 1.0
            
        else:  # High-vol chop
            regime.position_sizing = 'off'
            regime.risk_multiplier = 0.0
            regime.stop_loss_type = 'none'
            regime.take_profit_ratio = 0.0
            regime.stop_width_atr = 0.0
    
    def get_regime_report(self) -> str:
        """Generate human-readable regime report."""
        if not self.regime_states:
            return "No regimes detected. Train model first."
        
        report = "\n" + "="*80 + "\n"
        report += "HMM REGIME CLASSIFICATION REPORT\n"
        report += "="*80 + "\n\n"
        
        for regime_id, regime in self.regime_states.items():
            report += f"Regime {regime_id}: {regime.name}\n"
            report += f"  Trend: {regime.trend_direction.upper()}, "
            report += f"Volatility: {regime.volatility_level.upper()}\n"
            report += f"  Mean ATR Ratio: {regime.mean_atr_ratio:.4f}\n"
            report += f"  Mean RSI: {regime.mean_rsi:.2f}\n"
            report += f"  Mean Slope: {regime.mean_slope:.4f}\n"
            report += f"  Persistence: {regime.persistence_probability:.2%}\n"
            report += f"  Strategy:\n"
            report += f"    - Position Sizing: {regime.position_sizing}\n"
            report += f"    - Risk Multiplier: {regime.risk_multiplier}x\n"
            report += f"    - Stop Type: {regime.stop_loss_type}\n"
            report += f"    - TP Ratio: {regime.take_profit_ratio}:1\n"
            report += f"    - Stop Width: {regime.stop_width_atr}×ATR\n"
            report += "\n"
        
        return report


# ============================================================================
# SECTION 5: XGBOOST REGIME PREDICTOR
# ============================================================================

class XGBoostRegimePredictor:
    """
    Supervised regime prediction using XGBoost.
    
    Predicts next swing's regime given current structural and behavioral context.
    """
    
    def __init__(self, n_regimes: int = 6, random_state: int = 42):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost not available. Install with: pip install xgboost")
        
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.feature_columns = []
        self.feature_importance = {}
    
    def fit(self, swing_df: pd.DataFrame, hmm_labels: np.ndarray) -> None:
        """
        Train XGBoost to predict next regime.
        
        Args:
            swing_df: DataFrame from HybridSwingRegistry with features
            hmm_labels: Current regime labels from HMM
        """
        # Add HMM regime as feature
        swing_df = swing_df.copy()
        swing_df['current_regime'] = hmm_labels
        
        # Select all features (continuous + discrete)
        self.feature_columns = [
            'current_regime', 'magnitude_pct', 'duration', 'local_slope',
            'return_gradient', 'directional_persistence', 'atr_ratio',
            'bb_width', 'kc_width', 'normalized_variance', 'rsi', 'roc',
            'ppo', 'macd_hist', 'adx', 'volume_roc', 'obv_change',
            'volume_zscore', 'structure_break', 'pullback_depth',
            'recent_bullish_patterns', 'recent_bearish_patterns',
            'recent_doji', 'gap_intensity', 'extended_bar_ratio', 'volume_spike'
        ]
        
        # Encode structure_type as numeric
        structure_map = {'HH': 1, 'HL': 2, 'LH': 3, 'LL': 4, 'unknown': 0}
        swing_df['structure_numeric'] = swing_df['structure_type'].map(structure_map).fillna(0)
        self.feature_columns.append('structure_numeric')
        
        # Prepare features and target
        X = swing_df[self.feature_columns].fillna(0).values
        
        # Target is next regime (shift labels forward)
        y = np.roll(hmm_labels, -1)
        
        # Remove last row (no future regime)
        X = X[:-1]
        y = y[:-1]
        
        # Split chronologically (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=self.n_regimes,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Store feature importance
        importance = self.model.feature_importances_
        self.feature_importance = {
            self.feature_columns[i]: importance[i]
            for i in range(len(self.feature_columns))
        }
        
        # Evaluate
        from sklearn.metrics import accuracy_score, classification_report
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"✅ XGBoost trained")
        print(f"   Train Accuracy: {train_acc:.2%}")
        print(f"   Test Accuracy: {test_acc:.2%}")
        
        # Print classification report
        print("\n" + classification_report(y_test, y_pred_test, zero_division=0))
    
    def predict(self, swing_df: pd.DataFrame, current_regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next regime for each swing.
        
        Returns:
            predicted_regimes: Array of predicted regime IDs
            confidences: Array of prediction confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Add current regime as feature
        swing_df = swing_df.copy()
        swing_df['current_regime'] = current_regimes
        
        # Encode structure
        structure_map = {'HH': 1, 'HL': 2, 'LH': 3, 'LL': 4, 'unknown': 0}
        swing_df['structure_numeric'] = swing_df['structure_type'].map(structure_map).fillna(0)
        
        # Extract features
        X = swing_df[self.feature_columns].fillna(0).values
        
        # Predict
        predicted_regimes = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = probabilities.max(axis=1)
        
        return predicted_regimes, confidences
    
    def get_feature_importance_report(self) -> str:
        """Generate feature importance report."""
        if not self.feature_importance:
            return "No feature importance data. Train model first."
        
        report = "\n" + "="*80 + "\n"
        report += "XGBOOST FEATURE IMPORTANCE\n"
        report += "="*80 + "\n\n"
        
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feature, importance) in enumerate(sorted_features[:20], 1):
            report += f"{i:2d}. {feature:30s} {importance:.4f}\n"
        
        return report


# ============================================================================
# SECTION 6: MAIN REGIME DETECTION SYSTEM
# ============================================================================

class AdvancedRegimeDetectionSystem:
    """
    Main entry point for Hybrid ZigZag-Price Action Regime Detection.
    
    Integrates:
    - ZigZag swing extraction
    - Hybrid Swing Registry construction
    - GaussianHMM unsupervised classification
    - XGBoost supervised prediction
    - Strategy adaptation framework
    """
    
    def __init__(
        self,
        atr_multiplier: float = 2.0,
        min_bars: int = 5,
        n_regimes: int = 6,
        random_state: int = 42
    ):
        self.zigzag = ZigZagSwingExtractor(atr_multiplier, min_bars)
        self.registry = HybridSwingRegistry()
        self.hmm_classifier = None
        self.xgb_predictor = None
        self.n_regimes = n_regimes
        self.random_state = random_state
        
        # Legacy compatibility cache
        self.regime_cache = {}
    
    def detect_advanced_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to add historical regime data to a DataFrame.
        
        This is the entry point called by existing code for backward compatibility.
        
        Returns:
            pd.DataFrame: Original dataframe with regime columns added:
                - 'regime_volatility' (low, normal, high)
                - 'regime_trend' (ranging, trending, strong_trend)
                - 'historical_regime' (synthesized regime label)
                - 'hmm_regime' (unsupervised HMM regime ID)
                - 'predicted_regime' (supervised XGBoost prediction)
        """
        if len(df) < 50:
            df['historical_regime'] = 'unknown'
            df['regime_volatility'] = 'normal'
            df['regime_trend'] = 'ranging'
            return df
        
        # Cache check
        cache_key = f"{len(df)}_{df.index[-1]}"
        if cache_key in self.regime_cache:
            return self.regime_cache[cache_key]
        
        df_analysis = df.copy()
        
        # === PHASE 1: Extract ZigZag Swings ===
        print("  🔍 Extracting ZigZag swings...")
        zigzag_swings = self.zigzag.extract_swings(df_analysis)
        
        if len(zigzag_swings) < 10:
            print("  ⚠️ Insufficient swings for regime detection")
            df_analysis['historical_regime'] = 'unknown'
            df_analysis['regime_volatility'] = 'normal'
            df_analysis['regime_trend'] = 'ranging'
            return df_analysis
        
        print(f"     Found {len(zigzag_swings)} swing points")
        
        # === PHASE 2: Build Hybrid Swing Registry ===
        print("  📊 Building Hybrid Swing Registry...")
        self.registry.build_from_dataframe(df_analysis, zigzag_swings)
        swing_df = self.registry.to_dataframe()
        
        print(f"     Registry built with {len(self.registry.swings)} enriched swings")
        
        # === PHASE 3: HMM Classification ===
        if HMM_AVAILABLE and len(swing_df) >= 20:
            print("  🤖 Running HMM regime classification...")
            self.hmm_classifier = GaussianHMMRegimeClassifier(
                n_regimes=self.n_regimes,
                random_state=self.random_state
            )
            
            try:
                self.hmm_classifier.fit(swing_df)
                hmm_labels, hmm_probs = self.hmm_classifier.predict(swing_df)
                
                # Update registry with HMM results
                for i, swing in enumerate(self.registry.swings):
                    swing.hmm_regime = int(hmm_labels[i])
                    swing.hmm_probability = float(hmm_probs[i])
                
                print(f"     HMM classification complete")
                
            except Exception as e:
                print(f"  ⚠️ HMM classification failed: {e}")
                hmm_labels = np.zeros(len(swing_df), dtype=int)
        else:
            print("  ⚠️ Skipping HMM (insufficient data or library unavailable)")
            hmm_labels = np.zeros(len(swing_df), dtype=int)

        # === NEW PHASE: compute regime instances (adaptive clustering) ===
        try:
            swing_df = self.registry.to_dataframe()
            swing_df = self._compute_regime_instances(swing_df)
            # apply back to registry.swings so objects carry instance ids
            id_map = dict()
            for idx, row in swing_df.iterrows():
                key = (int(row['index']),)  # unique by swing.index
                id_map[int(row['index'])] = row.get('regime_instance_id')
            for swing in self.registry.swings:
                swing.regime_instance_id = id_map.get(swing.index, None)
        except Exception as e:
            print(f"  ⚠️ Regime instance computation failed: {e}")
            # continue without instances
        
        # === PHASE 4: XGBoost Prediction ===
        if XGB_AVAILABLE and len(swing_df) >= 50:
            print("  🎯 Training XGBoost regime predictor...")
            self.xgb_predictor = XGBoostRegimePredictor(
                n_regimes=self.n_regimes,
                random_state=self.random_state
            )
            
            try:
                swing_df = self.registry.to_dataframe()  # Refresh with HMM labels
                self.xgb_predictor.fit(swing_df, hmm_labels)
                
                pred_labels, pred_confidences = self.xgb_predictor.predict(swing_df, hmm_labels)
                
                # Update registry with predictions
                for i, swing in enumerate(self.registry.swings):
                    swing.predicted_regime = int(pred_labels[i])
                    swing.prediction_confidence = float(pred_confidences[i])
                
                print(f"     XGBoost prediction complete")
                
            except Exception as e:
                print(f"  ⚠️ XGBoost prediction failed: {e}")
        else:
            print("  ⚠️ Skipping XGBoost (insufficient data or library unavailable)")
        
        # === PHASE 5: Map Regimes to DataFrame ===
        print("  📈 Mapping regimes + instances to dataframe...")
        df_analysis = self._map_regimes_to_dataframe(df_analysis, swing_df, hmm_labels)
        
        # === PHASE 6: Generate Legacy Columns for Compatibility ===
        df_analysis = self._generate_legacy_columns(df_analysis)
        
        # Cache result
        self.regime_cache[cache_key] = df_analysis
        
        print("  ✅ Regime detection complete")
        
        return df_analysis
    
    def _map_regimes_to_dataframe(
        self,
        df: pd.DataFrame,
        swing_df: pd.DataFrame,
        hmm_labels: np.ndarray
    ) -> pd.DataFrame:
        """Map swing-level regimes to bar-level dataframe."""
        # Initialize regime columns
        df['hmm_regime'] = -1
        df['hmm_probability'] = 0.0
        df['predicted_regime'] = -1
        df['prediction_confidence'] = 0.0
        
        # Forward-fill regime from each swing point
        for i, swing in enumerate(self.registry.swings):
            idx = swing.index
            
            if idx < len(df):
                df.loc[df.index[idx], 'hmm_regime'] = swing.hmm_regime if swing.hmm_regime is not None else -1
                df.loc[df.index[idx], 'hmm_probability'] = swing.hmm_probability if swing.hmm_probability is not None else 0.0
                df.loc[df.index[idx], 'predicted_regime'] = swing.predicted_regime if swing.predicted_regime is not None else -1
                df.loc[df.index[idx], 'prediction_confidence'] = swing.prediction_confidence if swing.prediction_confidence is not None else 0.0
        
        # Forward-fill to propagate regimes
        df['hmm_regime'] = df['hmm_regime'].replace(-1, np.nan).ffill().fillna(-1).astype(int)
        df['hmm_probability'] = df['hmm_probability'].replace(0.0, np.nan).ffill().fillna(0.0)
        df['predicted_regime'] = df['predicted_regime'].replace(-1, np.nan).ffill().fillna(-1).astype(int)
        df['prediction_confidence'] = df['prediction_confidence'].replace(0.0, np.nan).ffill().fillna(0.0)
        
        return df
    
    def _generate_legacy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate legacy regime columns for backward compatibility.
        
        Maps HMM regimes to expected string labels.
        """
        # Map HMM regime IDs to string labels
        if self.hmm_classifier and self.hmm_classifier.regime_states:
            regime_map = {}
            for regime_id, regime_state in self.hmm_classifier.regime_states.items():
                # Convert to legacy format
                if regime_state.trend_direction == 'bull' and regime_state.volatility_level == 'low':
                    regime_map[regime_id] = 'strong_trend_normal_vol'
                elif regime_state.trend_direction == 'bull' and regime_state.volatility_level == 'high':
                    regime_map[regime_id] = 'strong_trend_high_vol'
                elif regime_state.trend_direction == 'bear' and regime_state.volatility_level == 'low':
                    regime_map[regime_id] = 'strong_trend_normal_vol'  # Bearish trends also use this
                elif regime_state.trend_direction == 'bear' and regime_state.volatility_level == 'high':
                    regime_map[regime_id] = 'strong_trend_high_vol'
                elif regime_state.volatility_level == 'low':
                    regime_map[regime_id] = 'ranging_low_vol'
                else:
                    regime_map[regime_id] = 'ranging_high_vol'
            
            df['historical_regime'] = df['hmm_regime'].map(regime_map).fillna('transition_normal_vol')
        else:
            # Fallback to simple volatility-based classification
            if 'atr' in df.columns:
                atr_pct = (df['atr'] / df['close']) * 100
                atr_ma = atr_pct.rolling(50).mean()
                atr_std = atr_pct.rolling(50).std()
                
                df['regime_volatility'] = 'normal'
                df.loc[atr_pct > (atr_ma + atr_std), 'regime_volatility'] = 'high'
                df.loc[atr_pct < (atr_ma - atr_std), 'regime_volatility'] = 'low'
            else:
                df['regime_volatility'] = 'normal'
            
            # Trend classification
            if 'adx' in df.columns:
                df['regime_trend'] = 'ranging'
                df.loc[df['adx'] > 20, 'regime_trend'] = 'trending'
                df.loc[df['adx'] > 30, 'regime_trend'] = 'strong_trend'
            else:
                df['regime_trend'] = 'ranging'
            
            # Synthesize historical_regime
            df['historical_regime'] = 'transition_normal_vol'
            
            # Ranging regimes
            mask = (df['regime_trend'] == 'ranging') & (df['regime_volatility'] == 'low')
            df.loc[mask, 'historical_regime'] = 'ranging_low_vol'
            
            mask = (df['regime_trend'] == 'ranging') & (df['regime_volatility'] == 'normal')
            df.loc[mask, 'historical_regime'] = 'ranging_normal_vol'
            
            mask = (df['regime_trend'] == 'ranging') & (df['regime_volatility'] == 'high')
            df.loc[mask, 'historical_regime'] = 'ranging_high_vol'
            
            # Trending regimes
            mask = df['regime_trend'].isin(['trending', 'strong_trend']) & (df['regime_volatility'] == 'normal')
            df.loc[mask, 'historical_regime'] = 'strong_trend_normal_vol'
            
            mask = df['regime_trend'].isin(['trending', 'strong_trend']) & (df['regime_volatility'] == 'high')
            df.loc[mask, 'historical_regime'] = 'strong_trend_high_vol'
        
        return df
    
    def get_regime_summary(self) -> str:
        """Generate comprehensive regime detection summary."""
        report = "\n" + "="*80 + "\n"
        report += "HYBRID ZIGZAG-PRICE ACTION REGIME DETECTION SUMMARY\n"
        report += "="*80 + "\n\n"
        
        report += f"Registry Size: {len(self.registry.swings)} swings\n\n"
        
        if self.hmm_classifier:
            report += self.hmm_classifier.get_regime_report()
        
        if self.xgb_predictor:
            report += self.xgb_predictor.get_feature_importance_report()
        
        return report
    
    def export_swing_registry(self, filename: str = 'swing_registry.csv') -> None:
        """Export Hybrid Swing Registry to CSV."""
        swing_df = self.registry.to_dataframe()
        swing_df.to_csv(filename, index=False)
        print(f"✅ Swing registry exported to {filename}")

    def _compute_regime_instances(self, swing_df: pd.DataFrame, min_instance_length: int = 3, z_threshold: float = 1.0) -> pd.DataFrame:
        """
        Adaptive segmentation within each HMM regime segment.
        Strategy (Option 3):
         - For each contiguous block of the same hmm_regime, inspect the sequence of
           atr_ratio and local_slope (or other metrics).
         - Compute z-scores for atr_ratio and local_slope inside the block.
         - If z-scores cross the threshold (abs > z_threshold) persistently for > min_instance_length,
           mark a cut and start a new instance.
         - Output: swing_df with a new column 'regime_instance_id' (string) and 'regime_instance_index' (int)
        """
        if 'hmm_regime' not in swing_df.columns:
            swing_df['regime_instance_id'] = None
            swing_df['regime_instance_index'] = -1
            return swing_df

        swing_df = swing_df.reset_index(drop=True)
        swing_df['regime_instance_id'] = None
        swing_df['regime_instance_index'] = -1

        # helper to create instance id
        def _new_instance_name(regime_label, counter):
            return f"R{int(regime_label)}_I{counter}"

        i = 0
        while i < len(swing_df):
            regime_label = swing_df.at[i, 'hmm_regime']
            # gather contiguous block
            j = i
            while j < len(swing_df) and swing_df.at[j, 'hmm_regime'] == regime_label:
                j += 1
            block = swing_df.iloc[i:j].copy()
            if len(block) == 0:
                i = j
                continue

            # compute zscores safely (fallback to simple normalization if nan)
            atr = block.get('atr_ratio', pd.Series(np.zeros(len(block))))
            slope = block.get('local_slope', pd.Series(np.zeros(len(block))))
            try:
                atr_z = stats.zscore(atr.fillna(0).astype(float))
                slope_z = stats.zscore(slope.fillna(0).astype(float))
            except Exception:
                # fallback: manual zscore
                def _z(a):
                    a = np.asarray(a, dtype=float)
                    m = np.nanmean(a); s = np.nanstd(a)
                    return (a - m) / (s if s != 0 else 1.0)
                atr_z = _z(atr)
                slope_z = _z(slope)

            # combine signals: mark change points where either absolute zscore exceeds threshold
            change_points = np.zeros(len(block), dtype=bool)
            for k in range(len(block)):
                if (abs(atr_z[k]) > z_threshold) or (abs(slope_z[k]) > z_threshold):
                    change_points[k] = True

            # convert change_points into instance segments: require persistence for min_instance_length
            # algorithm: walk block; start new instance at block start; if a run of 'change_points' of length >= min_instance_length,
            # cut before that run.
            instance_counter = 0
            seg_start = 0
            k = 0
            while k < len(block):
                # look ahead for persistent change
                if change_points[k]:
                    # find length of this run
                    run_start = k
                    while k < len(block) and change_points[k]:
                        k += 1
                    run_len = k - run_start
                    if run_len >= min_instance_length and run_start != 0:
                        # create instance for [seg_start:run_start)
                        inst_name = _new_instance_name(regime_label, instance_counter)
                        idxs = block.index[seg_start:run_start].tolist()
                        swing_df.loc[idxs, 'regime_instance_id'] = inst_name
                        swing_df.loc[idxs, 'regime_instance_index'] = instance_counter
                        instance_counter += 1
                        seg_start = run_start
                    # continue scanning
                else:
                    k += 1

            # final segment
            if seg_start < len(block):
                inst_name = _new_instance_name(regime_label, instance_counter)
                idxs = block.index[seg_start:len(block)].tolist()
                swing_df.loc[idxs, 'regime_instance_id'] = inst_name
                swing_df.loc[idxs, 'regime_instance_index'] = instance_counter
                instance_counter += 1

            # move to next block
            i = j

        # If any swings still None (edge cases), give them single-instance names
        for idx in swing_df[swing_df['regime_instance_id'].isnull()].index:
            rid = swing_df.at[idx, 'hmm_regime']
            swing_df.at[idx, 'regime_instance_id'] = _new_instance_name(rid, 0)
            swing_df.at[idx, 'regime_instance_index'] = 0

        return swing_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Hybrid ZigZag-Price Action Regime Detection System")
    print("="*80)
    print("\nThis module requires integration with the main trading system.")
    print("Import and use AdvancedRegimeDetectionSystem in your pipeline.")
    print("\nExample:")
    print("  from analysis_advanced_regime import AdvancedRegimeDetectionSystem")
    print("  detector = AdvancedRegimeDetectionSystem()")
    print("  df_with_regimes = detector.detect_advanced_market_regimes(df)")
    print("\nRequired dependencies:")
    print("  - hmmlearn (pip install hmmlearn)")
    print("  - xgboost (pip install xgboost)")
    print("  - scikit-learn (pip install scikit-learn)")