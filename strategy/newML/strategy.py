#autoTrader/strategy/newml/strategy.py
"""
This class acts as the bridge between your ML pipeline and your execution engine.

It handles:
    Data Buffering: Maintains the rolling window required for feature extraction.
    Inference: Calls the ML pipeline.
    Action Mapping: Translates the abstract "Regime 2" into concrete "Risk 0.5%" instructions.
    Fail-Safes: Automatically degrades to conservative settings if data is missing or the model is uncertain.
    Monitoring: Records inference results for later analysis.

    
"""
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .pipeline import RegimePipeline
from .features import FeatureRegistry, WindowConfig
from .interpretation import InterpretationRegistry, StrategyParams
from .monitoring import monitor
from ...db_connector import DatabaseConnector

@dataclass
class TradeInstruction:
    """Concrete instructions for the execution engine."""
    can_trade: bool
    risk_per_trade: float
    stop_loss_multiplier: float
    regime_id: int
    regime_name: str           
    confidence: float
    mode: str 
    signal_type: str = "NEUTRAL" # NEW: 'BULLISH_SIGNAL', 'BEARISH_SIGNAL', 'NEUTRAL'

class DynamicLogicEngine:
    """
    Evaluates the logic formulas stored in the database by StrategyFormulator.
    """
    def __init__(self, db_path: str, pair_tf: str):
        self.db = DatabaseConnector(db_path)
        self.pair_tf = pair_tf
        self.strategies = self._load_strategies()
        # Rolling window for defining "HIGH"/"LOW" relative to recent history
        self.history = pd.DataFrame()
        self.window_size = 50 

    def _load_strategies(self):
        """Fetches the JSON logic from strategy_playbook table."""
        try:
            # Load Bullish Strategy
            query = "SELECT confirming_indicators_json FROM strategy_playbook WHERE regime_name = 'Bullish_Swing_Gen'"
            res = self.db.execute(query, fetch=True)
            bull_strat = json.loads(res[0][0]) if res else None

            # Load Bearish Strategy
            query = "SELECT confirming_indicators_json FROM strategy_playbook WHERE regime_name = 'Bearish_Swing_Gen'"
            res = self.db.execute(query, fetch=True)
            bear_strat = json.loads(res[0][0]) if res else None
            
            return {'bull': bull_strat, 'bear': bear_strat}
        except Exception as e:
            print(f"⚠️ Failed to load dynamic strategies: {e}")
            return {'bull': None, 'bear': None}

    def update_data(self, new_data: Dict[str, float]):
        """Updates internal history to calculate relative levels (High/Low)."""
        # Convert dict to df row
        row = pd.DataFrame([new_data])
        self.history = pd.concat([self.history, row]).iloc[-self.window_size:]

    def check_signals(self, current_features: Dict[str, float]) -> str:
        """
        Evaluates the current feature set against the database logic.
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if self.history.empty: return "NEUTRAL"

        # 1. Check Bullish Logic
        if self._evaluate_formula(self.strategies['bull'], current_features):
            return "BULLISH"
            
        # 2. Check Bearish Logic
        if self._evaluate_formula(self.strategies['bear'], current_features):
            return "BEARISH"
            
        return "NEUTRAL"

    def _evaluate_formula(self, strategy_json, features) -> bool:
        if not strategy_json: return False
        
        # We only check 'causing_factors' for the trigger as they are the strong drivers
        causing = strategy_json.get('causing_factors', [])
        if not causing: return False
        
        conditions_met = True
        for factor in causing:
            feat_name = factor['feature']
            relation = factor['relation'] # 'HIGH' or 'LOW'
            
            if feat_name not in features:
                continue # Skip unknown features (safe fallback)
                
            curr_val = features[feat_name]
            avg_val = self.history[feat_name].mean()
            
            # Simple Relative Logic: 
            # HIGH means > Average
            # LOW means < Average
            if relation == 'HIGH' and curr_val <= avg_val:
                conditions_met = False
                break
            elif relation == 'LOW' and curr_val >= avg_val:
                conditions_met = False
                break
                
        return conditions_met

class RegimeAwareStrategy:
    def __init__(self, model_path: str = None, db_path: str = None, pair_tf: str = "btc_usdt_1h", pipeline: RegimePipeline = None):
        # 1. ML Pipeline (Clusters)
        if pipeline:
            self.pipeline = pipeline
        elif model_path:
            import joblib
            self.pipeline = joblib.load(model_path)
        else:
            # Optional: Allow running without ML model if we rely purely on DB logic
            self.pipeline = None

        self.feature_registry = FeatureRegistry()
        self.window_config = WindowConfig(window_size=32)
        self.interpreter = InterpretationRegistry()
        
        # 2. Dynamic Logic Engine (Swing Points Logic)
        if db_path:
            self.logic_engine = DynamicLogicEngine(db_path, pair_tf)
        else:
            self.logic_engine = None
        
        self._buffer_size = self.window_config.window_size + 10
        self.buffer = pd.DataFrame() 

    def on_bar(self, ts, o, h, l, c, v, extra_features: Dict = None) -> TradeInstruction:
        """
        Process a new market bar.
        Args:
            extra_features: Dict of pre-calculated indicators (RSI, etc.) from DB or calculations.
                            Essential for the DynamicLogicEngine.
        """
        # 1. Update Buffer
        new_row = pd.DataFrame({
            'open': [float(o)], 'high': [float(h)], 'low': [float(l)], 
            'close': [float(c)], 'volume': [float(v)]
        }, index=[pd.to_datetime(ts)])
        
        self.buffer = pd.concat([self.buffer, new_row])
        if len(self.buffer) > self._buffer_size:
            self.buffer = self.buffer.iloc[-self._buffer_size:]

        # 2. Update Dynamic Logic Engine with feature data
        if self.logic_engine and extra_features:
            self.logic_engine.update_data(extra_features)

        # 3. ML Inference (GMM)
        regime_id = 0
        confidence = 0.0
        
        if self.pipeline and len(self.buffer) >= self.window_config.window_size:
            try:
                curr_idx = len(self.buffer) - 1
                features = self.feature_registry.build_window(self.buffer, curr_idx, self.window_config)
                X = features.reshape(1, -1)
                pred = self.pipeline.predict(X)
                regime_id = pred['regime']
                confidence = pred['confidence']
                monitor.record_inference(regime_id, confidence)
            except Exception as e:
                print(f"[Strategy] ML Failed: {e}")

        # 4. Interpret ML Result
        strat_params = self.interpreter.get_action(regime_id, confidence)
        
        # 5. OVERRIDE with Causal Logic (The Swing Point Solution)
        signal_type = "NEUTRAL"
        if self.logic_engine and extra_features:
            signal_type = self.logic_engine.check_signals(extra_features)
            
            # If Causal Analysis says BULLISH, we force Aggressive Mode
            if signal_type == "BULLISH":
                strat_params.mode = "aggressive"
                strat_params.stop_multiplier = 2.0
                strat_params.max_risk_per_trade = 0.015
            # If Causal Analysis says BEARISH, we force Flat Mode
            elif signal_type == "BEARISH":
                strat_params.mode = "flat"
        
        return TradeInstruction(
            can_trade=(strat_params.mode != "flat"),
            risk_per_trade=strat_params.max_risk_per_trade,
            stop_loss_multiplier=strat_params.stop_multiplier,
            regime_id=regime_id,
            regime_name=f"R{regime_id}_{strat_params.mode}_{signal_type}",
            confidence=confidence,
            mode=strat_params.mode,
            signal_type=signal_type
        )

    def calculate_position_size(self, balance: float, entry: float, stop: float, instr: TradeInstruction) -> float:
        if not instr.can_trade or balance <= 0: return 0.0
        risk_amount = balance * instr.risk_per_trade
        stop_dist = abs(entry - stop)
        if stop_dist == 0: return 0.0
        return risk_amount / stop_dist

    def _get_fallback_instruction(self, reason: str) -> TradeInstruction:
        return TradeInstruction(True, 0.005, 1.5, 0, f"Fallback_{reason}", 0.0, "conservative")