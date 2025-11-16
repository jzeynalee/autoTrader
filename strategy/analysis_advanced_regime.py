"""
analysis_advanced_regime.py - REFACTORED
Hybrid Price Action Regime Detection Framework

This module implements a comprehensive regime detection system that:
1. Extracts Price Action swing points for structural consistency
2. Builds a Hybrid Swing Registry with rich price action metadata
3. Uses GaussianHMM for unsupervised regime classification
4. Uses XGBoost for supervised regime prediction
5. Provides adaptive strategy parameters per regime

Architecture:
    OHLCV Data → Price Action Swing Extraction → Hybrid Swing Registry
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
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Precision loss occurred.*')

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
    
    # structural fields
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
    
    def build_from_dataframe(self, df: pd.DataFrame, swing_points : List[Dict]) -> None:
        """
        Build registry by enriching ZigZag swings with price action features.
        
        Args:
            df: Full OHLCV dataframe with indicators
            swing_points : Output from ZigZagSwingExtractor
        """
        self.swings = []
        
        for swing_dict in swing_points :
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
    
    def _sanitize_label(self, label: str) -> str:
        """
        Make a safe single-word label from an arbitrary regime description.
        Example: "Low-Vol Bull Trend" -> "Low-Vol_Bull_Trend"
        """
        if label is None:
            return "Unknown_Regime"
        s = str(label)
        # Replace whitespace and slashes with underscore, remove problematic chars
        s = s.strip()
        s = s.replace('/', '_').replace('\\', '_')
        s = "_".join(s.split())               # spaces -> underscore
        # remove characters that are not alnum, dash, or underscore
        import re
        s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
        if s == "":
            s = "Regime"
        return s
    
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
        self.use_gpu: bool = True
    
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

    def _sanitize_label(self, label: str) -> str:
        """
        Make a safe single-word label from an arbitrary regime description.
        Example: "Low-Vol Bull Trend" -> "Low-Vol_Bull_Trend"
        """
        if label is None:
            return "Unknown_Regime"
        
        s = str(label)
        # Replace whitespace and slashes with underscore
        s = s.strip()
        s = s.replace('/', '_').replace('\\', '_')
        s = "_".join(s.split())  # spaces -> underscore
        
        # Remove characters that are not alphanumeric, dash, or underscore
        import re
        s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
        
        if s == "":
            s = "Regime"
        
        return s

    def _compute_local_slope(self, df, i, window=5):
        """Compute the slope of close prices around index i."""
        start = max(0, i - window)
        end = min(len(df), i + window)
        segment = df['close'].iloc[start:end].values

        if len(segment) < 2:
            return 0.0

        x = np.arange(len(segment))
        slope = np.polyfit(x, segment, 1)[0]
        return float(slope)
    
    def _compute_return_gradient(self, df, i, window=5):
        start = max(0, i - window)
        end = min(len(df), i + window)
        segment = df['close'].iloc[start:end].values

        if len(segment) < 2:
            return 0.0

        return float((segment[-1] - segment[0]) / segment[0] * 100)


    def _extract_feature_engineered_swings(self, df, lookback=2):
        """
        Extract swing points from feature-engineered columns (swing_high, swing_low)
        and compute the same metadata fields that the ZigZag extractor produced.
        This ensures full compatibility with HybridSwingRegistry.build_from_dataframe().
        """

        swings = []

        if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
            print("⚠️ swing_high/swing_low not found. Run feature engineering first.")
            return swings

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        atr = df['atr'].values if 'atr' in df.columns else np.zeros(len(df))
        timestamps = df.index

        for i in range(len(df)):

            # Swing HIGH
            if df['swing_high'].iloc[i] == 1:
                swing = {
                    'timestamp': timestamps[i],
                    'index': i,
                    'price': highs[i],
                    'swing_type': 'high',

                    # ===== Required fields for registry =====
                    'magnitude_pct': 0.0,    # placeholder, filled later
                    'duration': lookback,    # simple default
                    'atr_context': atr[i] if i < len(atr) else 0,
                    'local_slope': self._compute_local_slope(df, i),
                    'return_gradient': self._compute_return_gradient(df, i),
                }
                swings.append(swing)

            # Swing LOW
            if df['swing_low'].iloc[i] == 1:
                swing = {
                    'timestamp': timestamps[i],
                    'index': i,
                    'price': lows[i],
                    'swing_type': 'low',

                    # ===== Required fields for registry =====
                    'magnitude_pct': 0.0,
                    'duration': lookback,
                    'atr_context': atr[i] if i < len(atr) else 0,
                    'local_slope': self._compute_local_slope(df, i),
                    'return_gradient': self._compute_return_gradient(df, i),
                }
                swings.append(swing)

        # Ensure swing order is correct
        swings = sorted(swings, key=lambda x: x["index"])

        # Compute magnitude_pct after ordering
        for k in range(1, len(swings)):
            prev_price = swings[k-1]['price']
            curr_price = swings[k]['price']
            swings[k]['magnitude_pct'] = (curr_price - prev_price) / prev_price * 100

        return swings
    
    def _compute_regime_centroids(self, swing_df):
        """
        Compute mean feature vector for each HMM regime.
        Returns a dict: regime_id → feature vector dict.
        """
        centroids = {}

        if 'hmm_regime' not in swing_df.columns:
            return {}

        # Needed feature columns (fallbacks added)
        fcols = [
            'atr_ratio', 'local_slope', 'rsi', 'macd_hist', 'ppo', 'adx',
            'price_change', 'structure_type'
        ]

        df = swing_df.copy()

        # Fallback missing cols
        for c in fcols:
            if c not in df.columns:
                if c == 'structure_type':
                    df[c] = 'unknown'
                else:
                    df[c] = 0.0

        # One-hot encode structure types
        struct_dummies = pd.get_dummies(df['structure_type'], prefix='struct')
        df = pd.concat([df, struct_dummies], axis=1)

        # Compute regime centroids
        for r in sorted(df['hmm_regime'].dropna().unique()):
            sub = df[df['hmm_regime'] == r]

            if len(sub) == 0:
                continue

            # numeric mean features
            num_means = {
                'atr_ratio': float(sub['atr_ratio'].mean()),
                'local_slope': float(sub['local_slope'].mean()),
                'rsi': float(sub['rsi'].mean()),
                'macd_hist': float(sub['macd_hist'].mean()),
                'ppo': float(sub['ppo'].mean()),
                'adx': float(sub['adx'].mean()),
                'price_change': float(sub['price_change'].mean()),
            }

            # structure frequencies
            struct_means = {}
            for col in struct_dummies.columns:
                struct_means[col] = float(sub[col].mean())

            centroids[int(r)] = {**num_means, **struct_means}

        return centroids
    

    # ---------------------------
    # XGBoost GPU-first trainer
    # ---------------------------
    def _train_xgboost_gpu_fallback(self, X_train, y_train, X_test, y_test, num_class=None,
                                     num_boost_round=200, early_stopping_rounds=15,
                                     use_gpu=True, verbose_eval=True):
        """
        Train an XGBoost multiclass model preferring GPU (gpu_hist) and falling back to CPU.

        Returns:
            model (xgb.Booster), preds_test_prob (np.ndarray), preds_test (np.ndarray), history (dict)
        """
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("xgboost is not installed or cannot be imported.") from e

        # prepare data matrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        evals = [(dtrain, 'train'), (dtest, 'eval')]

        if num_class is None:
            # infer num_class from labels
            num_class = int(np.unique(y_train).max()) + 1

        base_params = {
            'objective': 'multi:softprob',
            'num_class': num_class,
            'eval_metric': 'mlogloss',
            'verbosity': 1,
            'seed': 42
        }

        # GPU attempt
        tried_gpu = False
        model = None
        history = None
        preds = None
        preds_label = None

        if use_gpu:
            tried_gpu = True
            gpu_params = dict(base_params)
            gpu_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                # 'gpu_id': 0  # optional; uses 0 by default
            })
            try:
                if verbose_eval:
                    print("  🎯 Training XGBoost regime predictor on GPU (gpu_hist)...")
                model = xgb.train(gpu_params, dtrain,
                                  num_boost_round=num_boost_round,
                                  evals=evals,
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=verbose_eval)
                history = model
                preds = model.predict(dtest)
                preds_label = preds.argmax(axis=1)
            except Exception as gpu_err:
                # fallback to CPU
                print(f"  ⚠️ GPU training failed ({type(gpu_err).__name__}: {gpu_err}). Falling back to CPU 'hist'.")
                model = None

        # CPU fallback
        if model is None:
            cpu_params = dict(base_params)
            cpu_params.update({
                'tree_method': 'hist',
                'predictor': 'cpu_predictor'
            })
            try:
                if verbose_eval:
                    print("  🎯 Training XGBoost regime predictor on CPU (hist)...")
                model = xgb.train(cpu_params, dtrain,
                                  num_boost_round=num_boost_round,
                                  evals=evals,
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=verbose_eval)
                history = model
                preds = model.predict(dtest)
                preds_label = preds.argmax(axis=1)
            except Exception as cpu_err:
                raise RuntimeError("Both GPU and CPU XGBoost training attempts failed.") from cpu_err

        return model, preds, preds_label, history

    def _map_regime_centroid_to_label(self, vec, vol_thresholds):
        """
        Assign human-readable regime label from centroid vector.
        vol_thresholds = (low_thr, high_thr) from ATR percentiles.
        """

        atr = vec['atr_ratio']
        slope = vec['local_slope']
        rsi = vec['rsi']
        ppo = vec['ppo']
        adx = vec['adx']
        macd = vec['macd_hist']
        pc = vec['price_change']

        low_thr, high_thr = vol_thresholds

        # ---------------------------
        # 1) Volatility bucket
        # ---------------------------
        if atr < low_thr:
            vol_label = "Low-Vol"
        elif atr > high_thr:
            vol_label = "High-Vol"
        else:
            vol_label = "Medium-Vol"

        # ---------------------------
        # 2) Trend bucket (slope + ADX)
        # ---------------------------
        if adx < 18:
            trend_label = "Sideways"
        else:
            if slope > 0:
                trend_label = "Bull Trend"
            elif slope < 0:
                trend_label = "Bear Trend"
            else:
                trend_label = "Sideways"

        # ---------------------------
        # 3) Momentum refinement
        # ---------------------------
        if "Trend" not in trend_label:
            # sideways subtypes
            if rsi > 60:
                trend_label = "Bullish Range"
            elif rsi < 40:
                trend_label = "Bearish Range"
            else:
                trend_label = "Choppy Range"
        else:
            # trending — refine with momentum
            if "Bull" in trend_label:
                if ppo < 0 or macd < 0:
                    trend_label = "Weak Bull Trend"
            if "Bear" in trend_label:
                if ppo > 0 or macd > 0:
                    trend_label = "Weak Bear Trend"

        return f"{vol_label} {trend_label}"

    def map_hmm_regimes_to_descriptions(self, swing_df):
        """
        Option 3 Hybrid Mapping:
        - compute regime centroids
        - compute ATR thresholds
        - assign human-readable labels
        Returns dict: regime_id → text_label
        """

        # -----------------------
        # A) compute regime centroids
        # -----------------------
        centroids = self._compute_regime_centroids(swing_df)
        if not centroids:
            return {}

        # -----------------------
        # B) compute volatility percentiles
        # -----------------------
        atr_values = [v['atr_ratio'] for v in centroids.values()]
        low_thr = np.percentile(atr_values, 35)
        high_thr = np.percentile(atr_values, 70)

        vol_thresholds = (low_thr, high_thr)

        # -----------------------
        # C) produce labels
        # -----------------------
        labels = {}
        for regime_id, vec in centroids.items():
            labels[regime_id] = self._map_regime_centroid_to_label(vec, vol_thresholds)

        return labels
    def _smooth_hmm_states(self, states, window=5):
        """
        Smooth HMM output using median filter to remove micro-flips.
        """
        if len(states) < window:
            return states
        padded = np.pad(states, (window//2, window//2), mode='edge')
        smoothed = []
        for i in range(len(states)):
            window_vals = padded[i:i+window]
            smoothed.append(int(np.median(window_vals)))
        return np.array(smoothed, dtype=int)

    def _apply_confidence_smoothing(self, states, prob, threshold=0.45, min_len=4):
        """
        Replace low-confidence short segments with the surrounding state.
        """
        states = states.copy()
        n = len(states)

        i = 0
        while i < n:
            if prob[i] < threshold:
                # find low-confidence run
                j = i + 1
                while j < n and prob[j] < threshold:
                    j += 1

                length = j - i
                if length <= min_len:
                    # replace with nearest stable state
                    left = states[i - 1] if i > 0 else states[j]
                    for k in range(i, j):
                        states[k] = left
                i = j
            else:
                i += 1

        return states

    def detect_advanced_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main function to add historical regime data to a DataFrame."""
        
        if len(df) < 50:
            df['historical_regime'] = 'unknown'
            df['regime_volatility'] = 'normal'
            df['regime_trend'] = 'ranging'
            return df
        
        cache_key = f"{len(df)}_{df.index[-1]}"
        if cache_key in self.regime_cache:
            return self.regime_cache[cache_key]
        
        df_analysis = df.copy()
        
        # ========================================================================
        # PHASE 1: Extract Price Action Swings
        # ========================================================================
        swing_points = self._extract_feature_engineered_swings(df_analysis)
        
        if len(swing_points) < 10:
            print(f"  ⚠️  Insufficient swings ({len(swing_points)}) for regime detection")
            df_analysis['historical_regime'] = 'unknown'
            df_analysis['regime_volatility'] = 'normal'
            df_analysis['regime_trend'] = 'ranging'
            return df_analysis
        
        # ========================================================================
        # PHASE 2: Build Hybrid Swing Registry
        # ========================================================================
        self.registry.build_from_dataframe(df_analysis, swing_points)
        swing_df = self.registry.to_dataframe()
        
        # ========================================================================
        # PHASE 3: HMM Classification
        # ========================================================================
        if HMM_AVAILABLE and len(swing_df) >= 20:
            self.hmm_classifier = GaussianHMMRegimeClassifier(
                n_regimes=self.n_regimes,
                random_state=self.random_state
            )
            
            try:
                self.hmm_classifier.fit(swing_df)
                hmm_labels, hmm_probs = self.hmm_classifier.predict(swing_df)
                
                for i, swing in enumerate(self.registry.swings):
                    swing.hmm_regime = int(hmm_labels[i])
                    swing.hmm_probability = float(hmm_probs[i])
                    
            except Exception as e:
                print(f"  ⚠️  HMM failed: {e}")
                hmm_labels = np.zeros(len(swing_df), dtype=int)
        else:
            hmm_labels = np.zeros(len(swing_df), dtype=int)
        
        # ========================================================================
        # PHASE 3.5: Build Regime Type Map
        # ========================================================================
        swing_df = self.registry.to_dataframe()
        self.regime_type_map = self.map_hmm_regimes_to_descriptions(swing_df)
        
        for swing in self.registry.swings:
            regime_id = swing.hmm_regime
            if regime_id is not None and not pd.isna(regime_id):
                swing.regime_type = self.regime_type_map.get(int(regime_id), "Unknown")
            else:
                swing.regime_type = "Unknown"
        
        # ========================================================================
        # PHASE 3.6: Segment Regime Instances
        # ========================================================================
        try:
            swing_df = self.registry.to_dataframe()
            
            segmented = self._segment_regime_instances(
                swing_df,
                min_instance_swings=3,
                max_instance_swings=40,
                vol_jump_pct=0.15,
                slope_sign_change=True,
                require_structure_rotation=False,
                low_confidence_threshold=0.50
            )
            
            id_map = {}
            idx_map = {}
            
            for _, row in segmented.iterrows():
                bar_idx = int(row['index'])
                inst_id = row['regime_instance_id']
                inst_idx = row['regime_instance_index']
                
                id_map[bar_idx] = inst_id
                idx_map[bar_idx] = inst_idx
            
            for swing in self.registry.swings:
                inst_id = id_map.get(swing.index, None)
                inst_idx = idx_map.get(swing.index, -1)
                
                swing.regime_instance_id = inst_id
                swing.regime_instance_index = inst_idx
            
        except Exception as e:
            print(f"  ⚠️  Instance segmentation failed: {e}")
        
        # ========================================================================
        # PHASE 4: XGBoost Prediction
        # ========================================================================
        if XGB_AVAILABLE and len(swing_df) >= 50:
            self.xgb_predictor = XGBoostRegimePredictor(
                n_regimes=self.n_regimes,
                random_state=self.random_state
            )
            
            try:
                swing_df = self.registry.to_dataframe()
                self.xgb_predictor.fit(swing_df, hmm_labels)
                
                pred_labels, pred_confidences = self.xgb_predictor.predict(swing_df, hmm_labels)
                
                for i, swing in enumerate(self.registry.swings):
                    swing.predicted_regime = int(pred_labels[i])
                    swing.prediction_confidence = float(pred_confidences[i])
                    
            except Exception as e:
                print(f"  ⚠️  XGBoost failed: {e}")
        
        # ========================================================================
        # PHASE 5: Map Regimes to DataFrame
        # ========================================================================
        swing_df = self.registry.to_dataframe()
        df_analysis = self._map_regimes_to_dataframe(df_analysis, swing_df, hmm_labels)
        
        # ========================================================================
        # PHASE 6: Generate Legacy Columns
        # ========================================================================
        df_analysis = self._generate_legacy_columns(df_analysis)
        
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

    def _segment_regime_instances(
        self,
        swing_df: pd.DataFrame,
        min_instance_swings: int = 6,
        max_instance_swings: int = 300,
        vol_jump_pct: float = 0.4,
        slope_sign_change: bool = True,
        require_structure_rotation: bool = True,
        low_confidence_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Robust segmentation of regime *instances* within each HMM regime type.

        Logic (for each contiguous block of the same `hmm_regime`):
        - Start instance at block start.
        - Split to a new instance when one or more of these occurs:
            * sustained volatility jump (abs(delta_atr)/prev_atr >= vol_jump_pct)
            * slope sign change (if slope_sign_change True)
            * price-structure rotation (structure_type changes direction if require_structure_rotation True)
            * regime probability drops below low_confidence_threshold (optional)
        - Enforce a minimum instance length (min_instance_swings) so brief fluctuations don't fragment.
        - Enforce a maximum instance length to force periodic splits (max_instance_swings).

        Adds / returns columns:
        - 'regime_instance_id'  (str): like "R{regime}_I{n}"
        - 'regime_instance_index' (int): 0-based index within regime type
        """

        df = swing_df.reset_index(drop=True).copy()

        # =========================================================================
        # ADAPTIVE THRESHOLD CALCULATION (OPTIONAL ENHANCEMENT)
        # =========================================================================
        
        '''# Calculate data-driven thresholds if possible
        if 'atr_ratio' in df.columns:
            atr_values = df['atr_ratio'].dropna()
            if len(atr_values) > 10:
                # Use percentile-based threshold instead of fixed value
                atr_std = atr_values.std()
                adaptive_vol_jump = min(0.3, max(0.10, atr_std * 2))  # 10-30% range
                
                # Only override if adaptive threshold is more reasonable
                if adaptive_vol_jump < vol_jump_pct:
                    print(f"     Adaptive vol_jump_pct: {adaptive_vol_jump:.2%} (was {vol_jump_pct:.2%})")
                    vol_jump_pct = adaptive_vol_jump'''
        
        # Calculate adaptive max_instance_swings based on total swings
        total_swings = len(df)
        if total_swings < 100:
            adaptive_max = 20
        elif total_swings < 500:
            adaptive_max = 40
        else:
            adaptive_max = 60
        
        if adaptive_max < max_instance_swings:
            print(f"     Adaptive max_instance_swings: {adaptive_max} (was {max_instance_swings})")
            max_instance_swings = adaptive_max

        # =========================================================================
        # DATA VALIDATION & COLUMN CHECKS
        # =========================================================================
        
        # ensure required columns exist; create safe defaults if missing
        if 'hmm_regime' not in df.columns:
            df['hmm_regime'] = -1
        
        if 'atr_ratio' not in df.columns:
            if 'atr' in df.columns and 'price' in df.columns:
                # try derive relative atr
                df['atr_ratio'] = df['atr'] / df['price'].replace(0, np.nan)
                df['atr_ratio'] = df['atr_ratio'].fillna(0.0)
            else:
                df['atr_ratio'] = 0.0

        if 'local_slope' not in df.columns:
            df['local_slope'] = 0.0
        if 'hmm_probability' not in df.columns:
            df['hmm_probability'] = 1.0
        if 'structure_type' not in df.columns:
            df['structure_type'] = 'unknown'

        # prepare result columns
        df['regime_instance_id'] = None
        df['regime_instance_index'] = -1

        # =========================================================================
        # HELPER FUNCTION: CREATE INSTANCE LABEL
        # =========================================================================
        
        def _new_instance_label(regime_label, counter):
            """Create human-readable instance ID."""
            # If the mapping of regime id -> text label exists, prefer that.
            try:
                # self.regime_type_map may be defined elsewhere in the detector pipeline
                mapped = getattr(self, 'regime_type_map', None)
                if mapped and regime_label in mapped:
                    base = mapped.get(regime_label)
                elif mapped and int(regime_label) in mapped:
                    base = mapped.get(int(regime_label))
                else:
                    base = f"R{int(regime_label)}"
            except Exception:
                base = f"R{int(regime_label)}"

            safe_base = self._sanitize_label(base)
            return f"{safe_base}_I{counter}"

        # =========================================================================
        # MAIN SEGMENTATION LOGIC
        # =========================================================================
        
        # iterate contiguous blocks of same regime label
        n = len(df)
        i = 0
        
        while i < n:
            regime_label = df.at[i, 'hmm_regime']
            
            # find block end
            j = i
            while j < n and df.at[j, 'hmm_regime'] == regime_label:
                j += 1
            
            block = df.iloc[i:j].copy().reset_index(drop=True)
            L = len(block)
            
            if L == 0:
                i = j
                continue

            # compute deltas for block (safe ops)
            atr = block['atr_ratio'].astype(float).fillna(0.0).values
            slope = block['local_slope'].astype(float).fillna(0.0).values
            prob = block['hmm_probability'].astype(float).fillna(1.0).values
            struct = block['structure_type'].astype(str).fillna('unknown').values

            # precompute volatility jump flags (relative diff vs previous)
            vol_change = np.zeros(L, dtype=bool)
            for k in range(1, L):
                prev_atr = atr[k-1] if atr[k-1] != 0 else 1e-9
                if prev_atr != 0:
                    if abs((atr[k] - atr[k-1]) / prev_atr) >= vol_jump_pct:
                        vol_change[k] = True

            # slope sign change flags
            slope_change = np.zeros(L, dtype=bool)
            if slope_sign_change:
                for k in range(1, L):
                    if slope[k] * slope[k-1] < 0:  # sign change
                        slope_change[k] = True

            # structure rotation flags (detect change between rising->falling categories)
            struct_change = np.zeros(L, dtype=bool)
            if require_structure_rotation:
                for k in range(1, L):
                    if struct[k] != struct[k-1]:
                        # only consider meaningful rotations (exclude unknown)
                        if struct[k] != 'unknown' and struct[k-1] != 'unknown':
                            struct_change[k] = True

            # low confidence flags
            low_conf = (prob < low_confidence_threshold)

            # now walk the block and assign instance ids using persistence rules
            instance_counter = 0
            seg_start = 0
            k = 0
            
            while k < L:
                # check for split triggers starting at k
                triggered = False
                
                # combine triggers at this position
                if vol_change[k] or slope_change[k] or struct_change[k] or low_conf[k]:
                    triggered = True

                # if triggered, check persistence and minimum length constraints
                if triggered:
                    # only allow a split if the segment [seg_start : k) has at least min_instance_swings
                    if (k - seg_start) >= min_instance_swings:
                        # assign current segment
                        instance_label = _new_instance_label(regime_label, instance_counter)
                        indices = block.index[seg_start:k].tolist()
                        df.loc[df.index[i + seg_start:i + k], 'regime_instance_id'] = instance_label
                        df.loc[df.index[i + seg_start:i + k], 'regime_instance_index'] = instance_counter
                        instance_counter += 1
                        seg_start = k

                # enforce a hard max instance length: if current segment grows too large, force a split
                if (k - seg_start + 1) >= max_instance_swings:
                    instance_label = _new_instance_label(regime_label, instance_counter)
                    df.loc[df.index[i + seg_start:i + k + 1], 'regime_instance_id'] = instance_label
                    df.loc[df.index[i + seg_start:i + k + 1], 'regime_instance_index'] = instance_counter
                    instance_counter += 1
                    seg_start = k + 1

                k += 1

            # finalize last segment in the block
            if seg_start < L:
                instance_label = _new_instance_label(regime_label, instance_counter)
                df.loc[df.index[i + seg_start:i + L], 'regime_instance_id'] = instance_label
                df.loc[df.index[i + seg_start:i + L], 'regime_instance_index'] = instance_counter
                instance_counter += 1

            # move to next block
            i = j

        # fallback: any None instance ids -> assign based on regime only
        for idx in df[df['regime_instance_id'].isnull()].index:
            rid = df.at[idx, 'hmm_regime']
            try:
                mapped = getattr(self, 'regime_type_map', None)
                if mapped and rid in mapped:
                    base = mapped.get(rid)
                elif mapped and pd.notna(rid) and int(rid) in mapped:
                    base = mapped.get(int(rid))
                else:
                    base = f"R{int(rid) if pd.notna(rid) else 'NA'}"
            except Exception:
                base = f"R{int(rid) if pd.notna(rid) else 'NA'}"

            safe_base = self._sanitize_label(base)
            df.at[idx, 'regime_instance_id'] = f"{safe_base}_I0"
            df.at[idx, 'regime_instance_index'] = 0

        # return with original index semantics
        return df
    
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