# sl_tp_planner.py
import pandas as pd
from typing import Dict, Optional, TypedDict, cast, Tuple, Literal
from trailing import TrailConfig

# Narrow type for trailing mode (module scope for Pylance/mypy)
TrailMode = Literal["percent", "atr", "swing", "chandelier"]


class SLTPPlanner:
    """
    Structured multi-timeframe SL/TP planning class.
    Uses swing points, ATR, MA across multiple timeframes.
    """

    """    class _PlanItem(TypedDict, total=False):
        sl: float
        tp: float
        RRR: float
        valid: bool

    class _TrailingItem(TypedDict):
        distance: float

    _PlanValue = Union["_PlanItem", "_TrailingItem"]"""

    class _PlanItem(TypedDict, total=False):
        sl: float
        tp: float
        RRR: float
        valid: bool
        # also used for TrailingStop entry; make it optional so mypy allows it
        distance: float

    def __init__(
        self,
        entry_price: float,
        symbol: str,
        data_by_timeframe: Dict[str, pd.DataFrame],
        fib_levels: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        :param entry_price: trade entry level
        :param symbol: trading pair (e.g., 'BTC/USDT')
        :param data_by_timeframe: dict of { '15m': df_15m, '1h': df_1h, ... }
        """
        self.entry = entry_price
        self.symbol = symbol
        self.data_by_timeframe = data_by_timeframe
        self.result: Dict[str, SLTPPlanner._PlanItem] = {}
        self.fib_levels: Dict[str, float] = fib_levels or {}

    # Add this method to SLTPPlanner class in sl_tp_planner.py

    def set_mtf_levels(
        self,
        htf_df: pd.DataFrame,
        ttf_df: pd.DataFrame, 
        ltf_df: pd.DataFrame,
        direction: str = 'bullish'
    ) -> None:
        """
        Multi-timeframe SL/TP using hierarchical structure:
        - HTF: Defines overall trend and major SL
        - TTF: Refines TP targets  
        - LTF: Defines precise entry and tight SL
        """
        
        # HTF: Major swing levels (widest SL)
        htf_highs = htf_df['swing_high'].dropna().tail(10)
        htf_lows = htf_df['swing_low'].dropna().tail(10)
        
        # TTF: Intermediate levels
        ttf_highs = ttf_df['swing_high'].dropna().tail(20)
        ttf_lows = ttf_df['swing_low'].dropna().tail(20)
        
        # LTF: Precise levels
        ltf_highs = ltf_df['swing_high'].dropna().tail(30)
        ltf_lows = ltf_df['swing_low'].dropna().tail(30)
        
        if direction == 'bullish':
            # SL: Use HTF swing low (strongest support)
            htf_support = htf_lows[htf_lows < self.entry].max() if not htf_lows[htf_lows < self.entry].empty else self.entry * 0.97
            
            # But tighten with LTF if very close
            ltf_support = ltf_lows[ltf_lows < self.entry].max() if not ltf_lows[ltf_lows < self.entry].empty else htf_support
            
            # Use LTF if it's within 1% of HTF (tighter stop)
            if htf_support == 0:
                sl = ltf_support  # Fallback to LTF support if HTF support is 0
            else:
                sl = ltf_support if abs(ltf_support - htf_support)/htf_support < 0.01 else htf_support
            sl = sl * 0.995  # Small buffer
            
            # TP: Use TTF resistance as primary target
            ttf_resistance = ttf_highs[ttf_highs > self.entry].min() if not ttf_highs[ttf_highs > self.entry].empty else self.entry * 1.03
            tp = ttf_resistance * 0.995
            
        else:  # bearish
            # SL: Use HTF swing high (strongest resistance)
            htf_resistance = htf_highs[htf_highs > self.entry].min() if not htf_highs[htf_highs > self.entry].empty else self.entry * 1.03
            
            # Tighten with LTF if very close
            ltf_resistance = ltf_highs[ltf_highs > self.entry].min() if not ltf_highs[ltf_highs > self.entry].empty else htf_resistance
            
            sl = ltf_resistance if abs(ltf_resistance - htf_resistance)/htf_resistance < 0.01 else htf_resistance
            sl = sl * 1.005
            
            # TP: Use TTF support
            ttf_support = ttf_lows[ttf_lows < self.entry].max() if not ttf_lows[ttf_lows < self.entry].empty else self.entry * 0.97
            tp = ttf_support * 1.005
        
        self.result['MTF_Hierarchical'] = {
            'sl': round(sl, 4),
            'tp': round(tp, 4),
            'htf_level': round(htf_support if direction == 'bullish' else htf_resistance, 4),
            'ttf_level': round(ttf_resistance if direction == 'bullish' else ttf_support, 4),
            'ltf_level': round(ltf_support if direction == 'bullish' else ltf_resistance, 4)
        }

    def set_by_regime(
        self,
        regime_type: str,
        volatility_regime: str = 'normal'
    ) -> None:
        """
        Adjust SL/TP based on market regime.
        
        Regime types:
        - 'strong_trend': Wider stops, larger targets
        - 'ranging': Tighter stops, modest targets
        - 'transition': Very tight stops, breakout targets
        """
        
        # Get ATR for baseline
        atr = None
        for df in self.data_by_timeframe.values():
            if 'high' in df.columns and 'low' in df.columns:
                atr_series = (df['high'] - df['low']).rolling(14).mean()
                atr = atr_series.iloc[-1]
                break
        
        if atr is None:
            atr = self.entry * 0.02  # 2% fallback
        
        # Regime-specific multipliers
        if regime_type == 'strong_trend_high_vol':
            sl_mult, tp_mult = 2.5, 4.0  # Wide stops, large targets
        elif regime_type == 'strong_trend_normal_vol':
            sl_mult, tp_mult = 2.0, 3.0
        elif regime_type == 'ranging_high_vol':
            sl_mult, tp_mult = 1.0, 1.5  # Tight stops, modest targets
        elif regime_type == 'ranging_normal_vol':
            sl_mult, tp_mult = 1.5, 2.0
        elif regime_type == 'transition':
            sl_mult, tp_mult = 0.8, 5.0  # Very tight stop, breakout target
        else:
            sl_mult, tp_mult = 1.5, 2.5  # Default
        
        sl = self.entry - (atr * sl_mult)
        tp = self.entry + (atr * tp_mult)
        
        self.result[f'Regime_{regime_type}'] = {
            'sl': round(sl, 4),
            'tp': round(tp, 4),
            'regime': regime_type,
            'volatility': volatility_regime
        }

    def set_by_swing_levels(self, lookback: int = 50) -> None:
        """
        Uses recent swing highs/lows to define SL/TP for each timeframe.
        """
        for tf, df in self.data_by_timeframe.items():
            lows = df["swing_low"].dropna().tail(lookback)
            highs = df["swing_high"].dropna().tail(lookback)

            support_candidates = lows[lows < self.entry]
            resistance_candidates = highs[highs > self.entry]

            support = (
                support_candidates.max()
                if not support_candidates.empty
                else self.entry * 0.98
            )
            resistance = (
                resistance_candidates.min()
                if not resistance_candidates.empty
                else self.entry * 1.02
            )

            sl = support * 0.995
            tp = resistance * 0.995

            self.result[f"SwingPoints_{tf}"] = {"sl": round(sl, 4), "tp": round(tp, 4)}

    def set_by_atr(
        self,
        atr_period: int = 14,
        multiplier_sl: float = 1.5,
        multiplier_tp: float = 2.5,
    ) -> None:
        """
        Sets SL/TP based on ATR range for each timeframe.
        """
        for tf, df in self.data_by_timeframe.items():
            atr = (
                df["high"].rolling(atr_period).max()
                - df["low"].rolling(atr_period).min()
            )
            latest_atr = atr.iloc[-1]

            sl = self.entry - latest_atr * multiplier_sl
            tp = self.entry + latest_atr * multiplier_tp

            self.result[f"ATR_{tf}"] = {"sl": round(sl, 4), "tp": round(tp, 4)}

    def set_by_moving_average(self, ma_column_name: str = "ma_50") -> None:
        """
        Uses a specified moving average to define SL/TP for each timeframe.
        Assumes MA is precomputed in each DataFrame.
        """
        for tf, df in self.data_by_timeframe.items():
            if ma_column_name not in df.columns:
                continue  # skip if MA not present

            ma_value = df[ma_column_name].iloc[-1]
            sl = ma_value * 0.99
            tp = self.entry + (self.entry - sl) * 2

            self.result[f"MA_{tf}"] = {"sl": round(sl, 4), "tp": round(tp, 4)}

    def set_by_fibonacci(self, fib_levels_by_tf: Dict[str, Dict[str, float]]) -> None:
        """
        Uses externally provided Fibonacci levels (e.g., from swing analysis).
        Format: {'1h': {'fib_61_8': 49.5, 'fib_127_2': 53.0, 'fib_161_8': 54.5}, ...}
        """
        for tf, levels in fib_levels_by_tf.items():
            if tf not in self.data_by_timeframe:
                continue

            fib_61_8: Optional[float] = levels.get("fib_61_8")
            fib_127_2: Optional[float] = levels.get("fib_127_2")
            fib_161_8: Optional[float] = levels.get("fib_161_8")

            if fib_61_8 is not None and fib_127_2 is not None and fib_161_8 is not None:
                sl = fib_61_8 * 0.995
                tp = (
                    fib_127_2
                    if abs(fib_127_2 - self.entry) < abs(fib_161_8 - self.entry)
                    else fib_161_8
                )

                self.result[f"Fibonacci_{tf}"] = {
                    "sl": round(sl, 4),
                    "tp": round(tp, 4),
                }

    def add_trailing_stop(self, distance: float) -> None:
        """
        Adds trailing stop info (applied globally).
        """
        self.result["TrailingStop"] = {"distance": round(distance, 4)}

    def validate_risk_reward(self, min_rr: float = 2.0) -> None:
        """
        Validates all methods in self.result to ensure RR >= min_rr.
        Adds RRR value and flags invalids.
        """
        for method, values in list(self.result.items()):
            # Only apply to entries that have SL/TP:
            if not isinstance(values, dict) or (
                "sl" not in values or "tp" not in values
            ):
                continue

            # Narrow the Union[_PlanItem, _TrailingItem] to _PlanItem
            item = cast(SLTPPlanner._PlanItem, values)
            risk = abs(self.entry - item["sl"])
            reward = abs(item["tp"] - self.entry)

            rr = reward / risk if risk != 0 else 0
            item["RRR"] = round(rr, 2)
            item["valid"] = rr >= min_rr
            self.result[method] = item

    def get_plan(self) -> Dict[str, "SLTPPlanner._PlanItem"]:
        """
        Returns final SL/TP plan across all timeframes.
        """
        return self.result

    # -------------------- TrailingStop -> TrailConfig bridge --------------------
    def _latest_close_atr_decimals(
        self,
    ) -> Tuple[Optional[float], Optional[float], int]:
        """
        Best-effort estimate of (last_close, ATR14, price_decimals) using the
        smallest timeframe DataFrame that has high/low/close.
        """
        # Pick the 'smallest' TF by typical naming if multiple are present
        # (fallback: first df that has the needed columns)
        priority = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
        dfs = [
            (k, v)
            for k, v in self.data_by_timeframe.items()
            if isinstance(v, pd.DataFrame)
        ]
        dfs.sort(
            key=lambda kv: priority.index(kv[0]) if kv[0] in priority else len(priority)
        )
        target = None
        for tf, df in dfs:
            if {"high", "low", "close"}.issubset(df.columns) and not df.empty:
                target = df
                break
        if target is None:
            return None, None, 4
        last_px = float(target["close"].iloc[-1])
        atr = None
        try:
            tr = (target["high"] - target["low"]).abs()
            if len(tr) >= 14:
                atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        except Exception:
            atr = None
        # crude decimal guess from last price (you can swap in exchange_precision later)
        s = f"{last_px:.10f}".rstrip("0").rstrip(".")
        dec = len(s.split(".")[1]) if "." in s else 0
        dec = max(0, min(8, dec))
        return last_px, atr, dec

    def build_trailing_config(
        self,
        *,
        mode: str = "auto",
        distance: Optional[float] = None,
        pct: Optional[float] = None,
        atr_k: float = 3.0,
        be_trigger_atr: float = 1.0,
        min_improve_atr: float = 0.25,
        cooldown_s: float = 10.0,
        spread_buffer: float = 0.0,
        price_scale: Optional[int] = None,
        swing_level: Optional[float] = None,
    ) -> TrailConfig:
        """
        Produce a ready-to-use TrailConfig for TrailingStopEngine.
        - If mode="auto" and you provided `distance`, we prefer:
            * ATR-mode if ATR is available (atr_k ~= distance / ATR)
            * else PERCENT-mode (pct ~= distance / price)
        - If you pass `pct`, we use percent mode.
        - Otherwise we honor the explicit `mode` you set.
        """
        last_px, atr, dec = self._latest_close_atr_decimals()
        if price_scale is None:
            price_scale = dec

        # Auto-map a plain distance into atr_k or pct if requested
        auto_mode = mode == "auto"
        mode_out: TrailMode = "chandelier"  # sensible default

        pct_out: Optional[float] = None
        atr_k_out: Optional[float] = None
        atr_val_out: Optional[float] = atr

        if pct is not None:
            mode_out = "percent"
            pct_out = float(pct)
        elif auto_mode and distance is not None:
            if atr is not None and atr > 0:
                mode_out = "atr"
                atr_k_out = max(0.01, float(distance) / float(atr))
            elif last_px is not None and last_px > 0:
                mode_out = "percent"
                pct_out = max(1e-6, float(distance) / float(last_px))
            else:
                # fallback to chandelier w/ provided atr_k (atr may still be None)
                mode_out = "chandelier"
                atr_k_out = atr_k
        else:
            # honor explicit mode; narrow for mypy
            mode_in = str(mode).lower()
            if mode_in in ("percent", "atr", "swing", "chandelier"):
                mode_out = cast(TrailMode, mode_in)
            else:
                mode_out = "chandelier"
            atr_k_out = atr_k

        # Assemble config
        cfg = TrailConfig(
            mode=mode_out,  # "percent" | "atr" | "swing" | "chandelier"
            pct=pct_out or 0.01,
            atr_k=atr_k_out or atr_k,
            atr_value=atr_val_out,
            swing_level=swing_level,
            be_trigger_atr=be_trigger_atr,
            min_improve_atr=min_improve_atr,
            cooldown_s=cooldown_s,
            spread_buffer=spread_buffer,
            price_scale=price_scale,
        )
        return cfg

    def get_trailing_config(self) -> Optional[TrailConfig]:
        """
        Build a TrailConfig from the planner's TrailingStop entry, if present.
        Returns None if no trailing is configured.
        """
        t = self.result.get("TrailingStop")
        if not isinstance(t, dict) or "distance" not in t:
            return None
        distance = float(t["distance"])
        return self.build_trailing_config(mode="auto", distance=distance)
