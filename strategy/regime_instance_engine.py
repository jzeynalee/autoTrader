"""
regime_instance_engine.py
Direct Regime Instance Discovery Engine

Discovers regime instances WITHOUT intermediate HMM classification.
Each instance is a self-contained period with measurable characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class RegimeInstance:
    """
    A regime instance is a complete data record for statistical analysis.
    Contains ALL information needed for future causality experiments.
    """
    # Identity
    instance_id: str
    pair: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    duration_hours: float
    
    # Market Structure
    swing_count: int
    avg_swing_magnitude_pct: float
    dominant_structure: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    
    # Price Action Metrics (distributions, not just means)
    price_change_pct: float
    max_drawdown_pct: float
    max_runup_pct: float
    
    volatility_mean: float
    volatility_std: float
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    
    # Momentum Metrics
    rsi_mean: float
    rsi_std: float
    rsi_trend: str
    
    macd_hist_mean: float
    macd_crossovers: int
    
    adx_mean: float
    adx_trend: str
    
    # Volume Metrics
    volume_mean: float
    volume_trend: str
    volume_spikes: int
    
    # Structure Characteristics
    higher_highs: int
    higher_lows: int
    lower_highs: int
    lower_lows: int
    structure_breaks_bullish: int
    structure_breaks_bearish: int
    
    # Pullback Characteristics
    pullback_count: int
    avg_pullback_depth_pct: float
    failed_pullbacks: int
    
    # Confirming Indicators (what was TRUE during this regime)
    confirming_indicators: List[str] = field(default_factory=list)
    indicator_values: Dict[str, Dict] = field(default_factory=dict)  # name → {mean, std, min, max}
    
    # Price Action Patterns
    price_action_patterns: List[str] = field(default_factory=list)
    
    # Candlestick Patterns
    bullish_patterns: Dict[str, int] = field(default_factory=dict)  # pattern → count
    bearish_patterns: Dict[str, int] = field(default_factory=dict)
    neutral_patterns: Dict[str, int] = field(default_factory=dict)
    
    # Chart Patterns
    chart_patterns: List[str] = field(default_factory=list)
    
    # Outcome Metrics (for causality analysis)
    next_1d_return_pct: float = 0.0
    next_3d_return_pct: float = 0.0
    next_7d_return_pct: float = 0.0
    max_favorable_excursion_1d: float = 0.0
    max_adverse_excursion_1d: float = 0.0
    
    # Regime Quality Scores
    consistency_score: float = 0.0  # How internally consistent is this regime?
    predictability_score: float = 0.0  # How predictable was the outcome?
    
    # Raw Data References (for detailed analysis)
    bar_indices: List[int] = field(default_factory=list)
    swing_indices: List[int] = field(default_factory=list)


class RegimeInstanceEngine:
    """
    Direct regime instance discovery without HMM intermediate step.
    
    Philosophy:
    1. Find natural market segments (3-10 days typically)
    2. Characterize each segment completely
    3. Store everything for future analysis
    """
    
    def __init__(
        self,
        min_bars_per_instance: int = 24,      # 1 day on hourly
        max_bars_per_instance: int = 168,     # 7 days on hourly
        volatility_threshold: float = 0.25,    # 25% change triggers split
        structure_change_window: int = 10,     # Bars to detect structure change
    ):
        self.min_bars = min_bars_per_instance
        self.max_bars = max_bars_per_instance
        self.vol_threshold = volatility_threshold
        self.struct_window = structure_change_window
        
        self.instances: List[RegimeInstance] = []
    
    def discover_instances(self, df: pd.DataFrame, pair: str, timeframe: str) -> List[RegimeInstance]:
        """
        Main entry point: discover all regime instances in the dataset.
        
        Returns list of RegimeInstance objects, each fully characterized.
        """
        print(f"\n{'='*80}")
        print(f"REGIME INSTANCE DISCOVERY: {pair} {timeframe}")
        print(f"{'='*80}")
        print(f"Dataset: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        # Step 1: Segment the data into instances
        segments = self._segment_data(df)
        print(f"\nSegmentation: Found {len(segments)} natural segments")
        
        # Step 2: Characterize each segment completely
        instances = []
        for seg_start, seg_end in segments:
            instance = self._characterize_instance(df, seg_start, seg_end, pair, timeframe)
            if instance:
                instances.append(instance)
        
        print(f"Created: {len(instances)} regime instances")
        
        # Step 3: Calculate outcome metrics
        instances = self._calculate_outcomes(df, instances)
        
        self.instances = instances
        return instances
    
    def _segment_data(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Segment data into natural regime boundaries.
        
        Triggers for new regime:
        1. Significant volatility change (25%+)
        2. Structure rotation (trend → range or vice versa)
        3. Momentum reversal (all momentum indicators flip sign)
        4. Time limit (max 7 days per instance)
        5. Minimum duration enforced (1 day)
        """
        segments = []
        n = len(df)
        
        # Compute segmentation signals
        atr = df['atr'].values if 'atr' in df.columns else np.ones(n)
        close = df['close'].values
        
        # Rolling volatility (20-bar)
        rolling_vol = pd.Series(close).pct_change().rolling(20).std().fillna(0).values
        
        # Detect structure (simple: is price making higher highs?)
        structure = np.zeros(n)
        for i in range(20, n):
            recent_highs = df['high'].iloc[i-20:i].values
            if recent_highs[-1] > recent_highs[:-1].max():
                structure[i] = 1  # Trending up
            elif recent_highs[-1] < recent_highs[:-1].min():
                structure[i] = -1  # Trending down
            else:
                structure[i] = 0  # Ranging
        
        # Momentum composite (RSI + MACD + PPO)
        momentum = np.zeros(n)
        if 'rsi' in df.columns:
            momentum += (df['rsi'].values - 50) / 50  # Normalize to -1 to 1
        if 'macd_hist' in df.columns:
            momentum += np.sign(df['macd_hist'].values)
        if 'ppo' in df.columns:
            momentum += np.sign(df['ppo'].values)
        
        # Segmentation logic
        seg_start = 0
        
        for i in range(self.min_bars, n):
            triggered = False
            reason = ""
            vol_change = 0.0 
            
            # Trigger 1: Volatility jump
            if i > 0 and rolling_vol[i-1] > 0:
                vol_change = abs(rolling_vol[i] - rolling_vol[i-1]) / rolling_vol[i-1]
                if vol_change > self.vol_threshold:
                    triggered = True
                    reason = f"volatility jump {vol_change:.1%}"
            
            # Trigger 2: Structure rotation
            if i >= self.struct_window:
                prev_struct = structure[i - self.struct_window]
                curr_struct = structure[i]
                if prev_struct != 0 and curr_struct != 0 and prev_struct != curr_struct:
                    triggered = True
                    reason = "structure rotation"
            
            # Trigger 3: Momentum reversal (all indicators flip)
            if i >= self.struct_window:
                prev_mom = momentum[i - self.struct_window]
                curr_mom = momentum[i]
                if prev_mom * curr_mom < -0.5:  # Opposite signs, strong signal
                    triggered = True
                    reason = "momentum reversal"
            
            # Trigger 4: Time limit
            bars_in_segment = i - seg_start
            if bars_in_segment >= self.max_bars:
                triggered = True
                reason = f"time limit ({bars_in_segment} bars)"
            
            # Execute split
            if triggered and (i - seg_start) >= self.min_bars:
                segments.append((seg_start, i))
                print(f"  Segment {len(segments)}: bars {seg_start}-{i} ({i-seg_start} bars) - {reason}")
                seg_start = i
        
        # Final segment
        if seg_start < n and (n - seg_start) >= self.min_bars:
            segments.append((seg_start, n))
        
        return segments
    
    def _characterize_instance(
        self, 
        df: pd.DataFrame, 
        start_idx: int, 
        end_idx: int,
        pair: str,
        timeframe: str
    ) -> RegimeInstance:
        """
        Fully characterize a regime instance with ALL available data.
        This is the heart of the system - comprehensive data collection.
        """
        segment = df.iloc[start_idx:end_idx].copy()
        
        if len(segment) < self.min_bars:
            return None
        
        # Generate unique instance ID
        start_time = segment.index[0]
        instance_id = f"{pair}_{timeframe}_{start_time.strftime('%Y%m%d_%H%M')}"
        
        # Calculate duration
        duration_hours = (segment.index[-1] - segment.index[0]).total_seconds() / 3600
        
        # Price action metrics
        price_start = segment['close'].iloc[0]
        price_end = segment['close'].iloc[-1]
        price_change_pct = ((price_end / price_start) - 1) * 100
        
        max_price = segment['high'].max()
        min_price = segment['low'].min()
        max_drawdown_pct = ((min_price / price_start) - 1) * 100
        max_runup_pct = ((max_price / price_start) - 1) * 100
        
        # Count swings in this segment
        swing_count = 0
        if 'swing_high' in segment.columns and 'swing_low' in segment.columns:
            swing_count = segment['swing_high'].sum() + segment['swing_low'].sum()
        
        avg_swing_mag = 0.0
        if swing_count > 1:
            swings = []
            if 'swing_high' in segment.columns:
                swing_highs = segment[segment['swing_high'] == 1]['high'].values
                swings.extend(swing_highs)
            if 'swing_low' in segment.columns:
                swing_lows = segment[segment['swing_low'] == 1]['low'].values
                swings.extend(swing_lows)
            
            if len(swings) > 1:
                swing_changes = np.abs(np.diff(swings)) / swings[:-1] * 100
                avg_swing_mag = float(np.mean(swing_changes))
        
        # Determine dominant structure
        dominant_structure = self._classify_structure(segment)
        
        # Volatility metrics
        volatility_mean = float(segment['atr'].mean()) if 'atr' in segment.columns else 0.0
        volatility_std = float(segment['atr'].std()) if 'atr' in segment.columns else 0.0
        volatility_trend = self._detect_trend(segment['atr'].values) if 'atr' in segment.columns else 'stable'
        
        # Momentum metrics
        rsi_mean = float(segment['rsi'].mean()) if 'rsi' in segment.columns else 50.0
        rsi_std = float(segment['rsi'].std()) if 'rsi' in segment.columns else 0.0
        rsi_trend = self._detect_trend(segment['rsi'].values) if 'rsi' in segment.columns else 'stable'
        
        macd_hist_mean = float(segment['macd_hist'].mean()) if 'macd_hist' in segment.columns else 0.0
        macd_crossovers = self._count_crossovers(segment['macd_hist'].values) if 'macd_hist' in segment.columns else 0
        
        adx_mean = float(segment['adx'].mean()) if 'adx' in segment.columns else 0.0
        adx_trend = self._detect_trend(segment['adx'].values) if 'adx' in segment.columns else 'stable'
        
        # Volume metrics
        volume_mean = float(segment['volume'].mean()) if 'volume' in segment.columns else 0.0
        volume_trend = self._detect_trend(segment['volume'].values) if 'volume' in segment.columns else 'stable'
        volume_spikes = self._count_volume_spikes(segment) if 'volume' in segment.columns else 0
        
        # Structure counts
        higher_highs = self._count_structure_type(segment, 'higher_high')
        higher_lows = self._count_structure_type(segment, 'higher_low')
        lower_highs = self._count_structure_type(segment, 'lower_high')
        lower_lows = self._count_structure_type(segment, 'lower_low')
        
        structure_breaks_bullish = int(segment['structure_break_bullish'].sum()) if 'structure_break_bullish' in segment.columns else 0
        structure_breaks_bearish = int(segment['structure_break_bearish'].sum()) if 'structure_break_bearish' in segment.columns else 0
        
        # Pullback characteristics
        pullback_count, avg_pullback_depth, failed_pullbacks = self._analyze_pullbacks(segment)
        
        # Confirming indicators (comprehensive collection)
        confirming_indicators, indicator_values = self._collect_confirming_indicators(segment)
        
        # Price action patterns
        price_action_patterns = self._collect_price_action_patterns(segment)
        
        # Candlestick patterns
        bullish_patterns, bearish_patterns, neutral_patterns = self._collect_candlestick_patterns(segment)
        
        # Chart patterns
        chart_patterns = self._collect_chart_patterns(segment)
        
        # Build instance
        instance = RegimeInstance(
            instance_id=instance_id,
            pair=pair,
            timeframe=timeframe,
            start_time=start_time,
            end_time=segment.index[-1],
            duration_hours=duration_hours,
            
            swing_count=swing_count,
            avg_swing_magnitude_pct=avg_swing_mag,
            dominant_structure=dominant_structure,
            
            price_change_pct=price_change_pct,
            max_drawdown_pct=max_drawdown_pct,
            max_runup_pct=max_runup_pct,
            
            volatility_mean=volatility_mean,
            volatility_std=volatility_std,
            volatility_trend=volatility_trend,
            
            rsi_mean=rsi_mean,
            rsi_std=rsi_std,
            rsi_trend=rsi_trend,
            
            macd_hist_mean=macd_hist_mean,
            macd_crossovers=macd_crossovers,
            
            adx_mean=adx_mean,
            adx_trend=adx_trend,
            
            volume_mean=volume_mean,
            volume_trend=volume_trend,
            volume_spikes=volume_spikes,
            
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
            structure_breaks_bullish=structure_breaks_bullish,
            structure_breaks_bearish=structure_breaks_bearish,
            
            pullback_count=pullback_count,
            avg_pullback_depth_pct=avg_pullback_depth,
            failed_pullbacks=failed_pullbacks,
            
            confirming_indicators=confirming_indicators,
            indicator_values=indicator_values,
            
            price_action_patterns=price_action_patterns,
            
            bullish_patterns=bullish_patterns,
            bearish_patterns=bearish_patterns,
            neutral_patterns=neutral_patterns,
            
            chart_patterns=chart_patterns,
            
            bar_indices=list(range(start_idx, end_idx)),
            swing_indices=self._get_swing_indices(segment, start_idx)
        )
        
        # Calculate quality scores
        instance.consistency_score = self._calculate_consistency_score(instance, segment)
        
        return instance
    
    def _classify_structure(self, segment: pd.DataFrame) -> str:
        """Classify the dominant market structure."""
        if len(segment) < 20:
            return 'insufficient_data'
        
        close = segment['close'].values
        highs = segment['high'].values
        lows = segment['low'].values
        
        # Linear regression slope
        x = np.arange(len(close))
        slope = np.polyfit(x, close, 1)[0]
        
        # ADX for trend strength
        adx = segment['adx'].mean() if 'adx' in segment.columns else 0
        
        # Volatility relative to range
        atr = segment['atr'].mean() if 'atr' in segment.columns else 0
        price_range = highs.max() - lows.min()
        vol_ratio = atr / price_range if price_range > 0 else 0
        
        # Classification logic
        if adx > 25:  # Strong trend
            if slope > 0:
                return 'trending_up_strong'
            else:
                return 'trending_down_strong'
        elif adx > 15:  # Moderate trend
            if slope > 0:
                return 'trending_up_moderate'
            else:
                return 'trending_down_moderate'
        else:  # Ranging or choppy
            if vol_ratio > 0.4:
                return 'volatile_choppy'
            else:
                return 'ranging_tight'
        
        return 'mixed'
    
    def _detect_trend(self, values: np.ndarray) -> str:
        """Detect if values are trending up, down, or stable."""
        if len(values) < 3:
            return 'stable'
        
        # Linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Normalize slope by value range
        value_range = values.max() - values.min()
        if value_range == 0:
            return 'stable'
        
        normalized_slope = slope / value_range
        
        if normalized_slope > 0.1:
            return 'increasing'
        elif normalized_slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _count_crossovers(self, values: np.ndarray) -> int:
        """Count zero crossovers."""
        if len(values) < 2:
            return 0
        
        signs = np.sign(values)
        sign_changes = np.diff(signs) != 0
        return int(sign_changes.sum())
    
    def _count_volume_spikes(self, segment: pd.DataFrame) -> int:
        """Count significant volume spikes."""
        if 'volume' not in segment.columns:
            return 0
        
        vol = segment['volume'].values
        vol_mean = vol.mean()
        vol_std = vol.std()
        
        if vol_std == 0:
            return 0
        
        # Spike = 2 standard deviations above mean
        spikes = vol > (vol_mean + 2 * vol_std)
        return int(spikes.sum())
    
    def _count_structure_type(self, segment: pd.DataFrame, structure_type: str) -> int:
        """Count occurrences of a specific structure type."""
        # This assumes structure analysis has been run
        # For now, implement basic counting logic
        
        if len(segment) < 10:
            return 0
        
        highs = segment['high'].values
        lows = segment['low'].values
        
        count = 0
        
        if structure_type == 'higher_high':
            for i in range(5, len(highs)):
                if highs[i] > highs[i-5:i].max():
                    count += 1
        
        elif structure_type == 'higher_low':
            for i in range(5, len(lows)):
                if lows[i] > lows[i-5:i].max():
                    count += 1
        
        elif structure_type == 'lower_high':
            for i in range(5, len(highs)):
                if highs[i] < highs[i-5:i].min():
                    count += 1
        
        elif structure_type == 'lower_low':
            for i in range(5, len(lows)):
                if lows[i] < lows[i-5:i].min():
                    count += 1
        
        return count
    
    def _analyze_pullbacks(self, segment: pd.DataFrame) -> Tuple[int, float, int]:
        """Analyze pullback characteristics."""
        # Look for pullback indicators
        pullback_count = 0
        pullback_depths = []
        failed_count = 0
        
        if 'pullback_depth' in segment.columns:
            #pullbacks = segment[segment['pullback_depth'] > 0]
            # Force the column to be numeric; strings/errors become NaN
            numeric_pullback_depth = pd.to_numeric(segment['pullback_depth'], errors='coerce')

            # Now, perform the comparison on the clean, numeric-only Series
            pullbacks = segment[numeric_pullback_depth > 0]

            pullback_count = len(pullbacks)
            if pullback_count > 0:
                pullback_depths = pullbacks['pullback_depth'].tolist()
        
        if 'failed_pullback_bull' in segment.columns:
            failed_count += int(segment['failed_pullback_bull'].sum())
        if 'failed_pullback_bear' in segment.columns:
            failed_count += int(segment['failed_pullback_bear'].sum())
        
        avg_depth = float(np.mean(pullback_depths)) if pullback_depths else 0.0
        
        return pullback_count, avg_depth, failed_count
    
    def _collect_confirming_indicators(self, segment: pd.DataFrame) -> Tuple[List[str], Dict]:
        """
        Collect ALL indicators that are in confirming state during this regime.
        
        Returns:
            - List of indicator names in confirming state
            - Dict of indicator statistics: {name: {mean, std, min, max, trend}}
        """
        confirming = []
        indicator_stats = {}
        
        # Momentum indicators
        if 'rsi' in segment.columns:
            rsi = segment['rsi']
            if rsi.mean() > 55:
                confirming.append('rsi_bullish')
            elif rsi.mean() < 45:
                confirming.append('rsi_bearish')
            
            indicator_stats['rsi'] = {
                'mean': float(rsi.mean()),
                'std': float(rsi.std()),
                'min': float(rsi.min()),
                'max': float(rsi.max()),
                'trend': self._detect_trend(rsi.values)
            }
        
        if 'macd_hist' in segment.columns:
            macd = segment['macd_hist']
            if macd.mean() > 0:
                confirming.append('macd_bullish')
            elif macd.mean() < 0:
                confirming.append('macd_bearish')
            
            indicator_stats['macd_hist'] = {
                'mean': float(macd.mean()),
                'std': float(macd.std()),
                'min': float(macd.min()),
                'max': float(macd.max()),
                'trend': self._detect_trend(macd.values)
            }
        
        if 'ppo' in segment.columns:
            ppo = segment['ppo']
            if ppo.mean() > 0:
                confirming.append('ppo_bullish')
            elif ppo.mean() < 0:
                confirming.append('ppo_bearish')
            
            indicator_stats['ppo'] = {
                'mean': float(ppo.mean()),
                'std': float(ppo.std()),
                'min': float(ppo.min()),
                'max': float(ppo.max()),
                'trend': self._detect_trend(ppo.values)
            }
        
        # Trend indicators
        if 'adx' in segment.columns:
            adx = segment['adx']
            if adx.mean() > 25:
                confirming.append('adx_strong_trend')
            elif adx.mean() < 20:
                confirming.append('adx_ranging')
            
            indicator_stats['adx'] = {
                'mean': float(adx.mean()),
                'std': float(adx.std()),
                'min': float(adx.min()),
                'max': float(adx.max()),
                'trend': self._detect_trend(adx.values)
            }
        
        # Volatility indicators
        if 'atr' in segment.columns:
            atr = segment['atr']
            atr_pct = atr / segment['close'] * 100
            
            if atr_pct.mean() > 3:
                confirming.append('high_volatility')
            elif atr_pct.mean() < 1:
                confirming.append('low_volatility')
            
            indicator_stats['atr_percent'] = {
                'mean': float(atr_pct.mean()),
                'std': float(atr_pct.std()),
                'min': float(atr_pct.min()),
                'max': float(atr_pct.max()),
                'trend': self._detect_trend(atr_pct.values)
            }
        
        if 'bb_width' in segment.columns:
            bb = segment['bb_width']
            if bb.mean() > bb.quantile(0.7):
                confirming.append('bb_expanding')
            elif bb.mean() < bb.quantile(0.3):
                confirming.append('bb_contracting')
            
            indicator_stats['bb_width'] = {
                'mean': float(bb.mean()),
                'std': float(bb.std()),
                'min': float(bb.min()),
                'max': float(bb.max()),
                'trend': self._detect_trend(bb.values)
            }
        
        # Volume indicators
        if 'obv' in segment.columns:
            obv = segment['obv']
            obv_trend = self._detect_trend(obv.values)
            if obv_trend == 'increasing':
                confirming.append('obv_accumulation')
            elif obv_trend == 'decreasing':
                confirming.append('obv_distribution')
            
            indicator_stats['obv'] = {
                'mean': float(obv.mean()),
                'std': float(obv.std()),
                'min': float(obv.min()),
                'max': float(obv.max()),
                'trend': obv_trend
            }
        
        # Add more indicators as needed...
        # (Similar pattern for all 100+ indicators)
        
        return confirming, indicator_stats
    
    def _collect_price_action_patterns(self, segment: pd.DataFrame) -> List[str]:
        """Collect price action patterns present in this regime."""
        patterns = []
        
        # Structure patterns
        if 'structure_break_bullish' in segment.columns:
            if segment['structure_break_bullish'].sum() > 0:
                patterns.append('bullish_structure_break')
        
        if 'structure_break_bearish' in segment.columns:
            if segment['structure_break_bearish'].sum() > 0:
                patterns.append('bearish_structure_break')
        
        # Pullback patterns
        if 'healthy_bull_pullback' in segment.columns:
            if segment['healthy_bull_pullback'].sum() > 0:
                patterns.append('healthy_bull_pullback')
        
        if 'healthy_bear_pullback' in segment.columns:
            if segment['healthy_bear_pullback'].sum() > 0:
                patterns.append('healthy_bear_pullback')
        
        # Fibonacci patterns
        if 'near_fib_618' in segment.columns:
            if segment['near_fib_618'].sum() > 0:
                patterns.append('fibonacci_618_bounce')
        
        # Add more pattern checks...
        
        return patterns
    
    def _collect_candlestick_patterns(self, segment: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
        """Collect candlestick pattern occurrences."""
        bullish = {}
        bearish = {}
        neutral = {}
        
        # Bullish patterns
        bullish_pattern_cols = [
            'pattern_hammer', 'pattern_bullish_engulfing', 'pattern_morning_star',
            'pattern_piercing_line', 'pattern_three_white_soldiers'
        ]
        
        for col in bullish_pattern_cols:
            if col in segment.columns:
                count = int(segment[col].sum())
                if count > 0:
                    pattern_name = col.replace('pattern_', '')
                    bullish[pattern_name] = count
        
        # Bearish patterns
        bearish_pattern_cols = [
            'pattern_shooting_star', 'pattern_bearish_engulfing', 'pattern_evening_star',
            'pattern_dark_cloud_cover', 'pattern_three_black_crows'
        ]
        
        for col in bearish_pattern_cols:
            if col in segment.columns:
                count = int(segment[col].sum())
                if count > 0:
                    pattern_name = col.replace('pattern_', '')
                    bearish[pattern_name] = count
        
        # Neutral patterns
        neutral_pattern_cols = ['pattern_doji', 'pattern_spinning_top']
        
        for col in neutral_pattern_cols:
            if col in segment.columns:
                count = int(segment[col].sum())
                if count > 0:
                    pattern_name = col.replace('pattern_', '')
                    neutral[pattern_name] = count
        
        return bullish, bearish, neutral
    
    def _collect_chart_patterns(self, segment: pd.DataFrame) -> List[str]:
        """Collect chart patterns (double tops, triangles, etc.)."""
        patterns = []
        
        chart_pattern_cols = [
            'double_top', 'double_bottom', 'head_and_shoulders',
            'ascending_triangle', 'descending_triangle', 'symmetrical_triangle',
            'cup_and_handle', 'bullish_flag', 'bearish_flag'
        ]
        
        for col in chart_pattern_cols:
            if col in segment.columns:
                if segment[col].sum() > 0:
                    patterns.append(col)
        
        return patterns
    
    def _get_swing_indices(self, segment: pd.DataFrame, start_idx: int) -> List[int]:
        """Get absolute indices of swings in this segment."""
        swing_indices = []
        
        if 'swing_high' in segment.columns:
            highs = segment[segment['swing_high'] == 1].index
            swing_indices.extend([start_idx + segment.index.get_loc(idx) for idx in highs])
        
        if 'swing_low' in segment.columns:
            lows = segment[segment['swing_low'] == 1].index
            swing_indices.extend([start_idx + segment.index.get_loc(idx) for idx in lows])
        
        return sorted(swing_indices)
    
    def _calculate_consistency_score(self, instance: RegimeInstance, segment: pd.DataFrame) -> float:
        """
        Calculate how internally consistent this regime is.
        
        High score = all indicators align, low variance, clear structure
        Low score = mixed signals, high variance, unclear structure
        """
        score = 0.0
        max_score = 0.0
        
        # 1. Indicator alignment (do momentum indicators agree?)
        momentum_signals = 0
        total_momentum = 0
        
        if instance.rsi_mean > 55:
            momentum_signals += 1
        elif instance.rsi_mean < 45:
            momentum_signals += 1
        total_momentum += 1
        
        if instance.macd_hist_mean > 0:
            momentum_signals += 1
        elif instance.macd_hist_mean < 0:
            momentum_signals += 1
        total_momentum += 1
        
        if total_momentum > 0:
            alignment_score = momentum_signals / total_momentum
            score += alignment_score * 30
        max_score += 30
        
        # 2. Volatility consistency (low std = more consistent)
        if instance.volatility_std > 0:
            vol_consistency = 1 - min(instance.volatility_std / instance.volatility_mean, 1.0)
            score += vol_consistency * 30
        max_score += 30
        
        # 3. Structure clarity (clear trends get higher scores)
        if instance.dominant_structure in ['trending_up_strong', 'trending_down_strong']:
            score += 40
        elif instance.dominant_structure in ['trending_up_moderate', 'trending_down_moderate']:
            score += 25
        elif instance.dominant_structure in ['ranging_tight']:
            score += 20
        else:
            score += 5
        max_score += 40
        
        return (score / max_score) * 100 if max_score > 0 else 0.0
    
    def _calculate_outcomes(self, df: pd.DataFrame, instances: List[RegimeInstance]) -> List[RegimeInstance]:
        """
        Calculate outcome metrics for each instance (what happened AFTER the regime).
        This is critical for causality analysis.
        """
        for instance in instances:
            end_idx = instance.bar_indices[-1] if instance.bar_indices else 0
            
            if end_idx >= len(df) - 1:
                continue  # Not enough future data
            
            end_price = df['close'].iloc[end_idx]
            
            # 1-day outcome
            if end_idx + 24 < len(df):  # Assuming hourly data
                future_1d = df.iloc[end_idx:end_idx+24]
                price_1d = df['close'].iloc[end_idx + 24]
                instance.next_1d_return_pct = ((price_1d / end_price) - 1) * 100
                instance.max_favorable_excursion_1d = ((future_1d['high'].max() / end_price) - 1) * 100
                instance.max_adverse_excursion_1d = ((future_1d['low'].min() / end_price) - 1) * 100
            
            # 3-day outcome
            if end_idx + 72 < len(df):
                price_3d = df['close'].iloc[end_idx + 72]
                instance.next_3d_return_pct = ((price_3d / end_price) - 1) * 100
            
            # 7-day outcome
            if end_idx + 168 < len(df):
                price_7d = df['close'].iloc[end_idx + 168]
                instance.next_7d_return_pct = ((price_7d / end_price) - 1) * 100
            
            # Predictability score (how well did indicators predict outcome?)
            # If bullish indicators and positive outcome, high score
            bullish_score = 0
            if instance.rsi_mean > 50:
                bullish_score += 1
            if instance.macd_hist_mean > 0:
                bullish_score += 1
            if 'macd_bullish' in instance.confirming_indicators:
                bullish_score += 1
            
            if bullish_score > 0 and instance.next_1d_return_pct > 0:
                instance.predictability_score = min((bullish_score / 3) * abs(instance.next_1d_return_pct) * 10, 100)
            elif bullish_score == 0 and instance.next_1d_return_pct < 0:
                instance.predictability_score = min(abs(instance.next_1d_return_pct) * 10, 100)
            else:
                instance.predictability_score = 0
        
        return instances


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Regime Instance Engine - Direct Discovery System")
    print("="*80)
    print("\nThis is the NEW approach: direct instance discovery without HMM.")
    print("Each instance is fully characterized for statistical analysis.")
