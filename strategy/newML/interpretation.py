import numpy as np
from dataclasses import dataclass

@dataclass
class StrategyParams:
    mode: str  # 'aggressive', 'conservative', 'flat'
    stop_multiplier: float
    max_risk_per_trade: float

class InterpretationRegistry:
    def __init__(self):
        # MAPPING DERIVED FROM DB PROFILE (2025-11-28)
        # Data showed Regime 2 had highest returns (Bull/Pump).
        # Data showed Regime 1 had negative returns (Bear/Dump).
        # Data showed Regime 0 had low volatility (Chop).
        
        self.rules = {
            # CHOP (Regime 0): Tight stops, standard risk
            0: StrategyParams(mode="conservative", stop_multiplier=1.0, max_risk_per_trade=0.005),
            
            # BEAR (Regime 1): Stay Flat / Cash (or Short if enabled)
            1: StrategyParams(mode="flat", stop_multiplier=1.0, max_risk_per_trade=0.0),
            
            # BULL (Regime 2): Let it run, wide stops for volatility
            2: StrategyParams(mode="aggressive", stop_multiplier=2.5, max_risk_per_trade=0.02),
        }
        
    def get_action(self, regime_id: int, confidence: float) -> StrategyParams:
        # Confidence Gating
        # If model is unsure (<50%), default to conservative (Regime 0 behavior)
        if confidence < 0.50:
            return self.rules[0]
            
        return self.rules.get(regime_id, self.rules[0])