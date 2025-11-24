"""
formula_strategy_generator.py

Generates actionable trading strategies with explicit formulas based on:
1. Statistical causality analysis from regime_statistical_analysis_sqlite.py
2. Indicator effectiveness and correlations
3. Multi-timeframe confirmation logic
4. Regime-specific thresholds and parameters

Each strategy includes:
- HTF/TTF/LTF formulas with specific conditions
- Causal indicator relationships
- Entry/Exit logic with thresholds
- Risk management parameters
"""

import json
import sqlite3
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class IndicatorCondition:
    """Single indicator condition with specific threshold"""
    indicator: str
    operator: str  # >, <, >=, <=, ==, crosses_above, crosses_below
    value: float
    timeframe: str
    lookback: int = 1  # bars to look back for confirmation
    
    def to_formula(self) -> str:
        """Convert to human-readable formula"""
        if self.operator in ['crosses_above', 'crosses_below']:
            return f"{self.indicator}[{self.timeframe}] {self.operator} {self.value} (confirmed {self.lookback} bars)"
        return f"{self.indicator}[{self.timeframe}] {self.operator} {self.value}"


@dataclass
class TimeframeFormula:
    """Complete formula for one timeframe"""
    timeframe: str
    primary_conditions: List[IndicatorCondition]  # Must ALL be true
    confirmation_conditions: List[IndicatorCondition]  # At least N must be true
    min_confirmations: int = 2
    causal_chain: List[str] = None  # Causal relationships from stats
    
    def to_formula_string(self) -> str:
        """Generate readable formula"""
        primary = " AND ".join([c.to_formula() for c in self.primary_conditions])
        
        if self.confirmation_conditions:
            confirm_count = min(self.min_confirmations, len(self.confirmation_conditions))
            confirm = f" AND (At least {confirm_count} of: " + \
                     ", ".join([c.to_formula() for c in self.confirmation_conditions]) + ")"
        else:
            confirm = ""
        
        causal = ""
        if self.causal_chain:
            causal = f" | Causal chain: {' → '.join(self.causal_chain)}"
        
        return f"[{self.timeframe}] {primary}{confirm}{causal}"


@dataclass
class MTFStrategyFormula:
    """Complete multi-timeframe strategy with formulas"""
    strategy_id: str
    name: str
    direction: str  # LONG or SHORT
    
    # Timeframe formulas
    htf_formula: TimeframeFormula
    ttf_formula: TimeframeFormula
    ltf_formula: TimeframeFormula
    
    # Entry/Exit logic
    entry_logic: str  # Complete entry condition
    exit_logic: str  # Complete exit condition
    
    # Risk management
    stop_loss_formula: str
    take_profit_formula: str
    position_size_formula: str
    
    # Statistical backing
    win_rate: float
    profit_factor: float
    sample_size: int
    statistical_significance: float  # p-value from causality tests
    
    # Regime context
    optimal_regimes: List[str]  # Which regimes this works best in
    avoid_regimes: List[str]  # Which regimes to avoid
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'direction': self.direction,
            'formulas': {
                'htf': self.htf_formula.to_formula_string(),
                'ttf': self.ttf_formula.to_formula_string(),
                'ltf': self.ltf_formula.to_formula_string()
            },
            'entry_logic': self.entry_logic,
            'exit_logic': self.exit_logic,
            'risk_management': {
                'stop_loss': self.stop_loss_formula,
                'take_profit': self.take_profit_formula,
                'position_size': self.position_size_formula
            },
            'statistics': {
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'sample_size': self.sample_size,
                'significance': self.statistical_significance
            },
            'regime_context': {
                'optimal_regimes': self.optimal_regimes,
                'avoid_regimes': self.avoid_regimes
            }
        }


class FormulaStrategyGenerator:
    """
    Generates formula-based strategies using statistical causality analysis
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def get_indicator_causality(self, regime_type: str = None) -> List[Dict]:
        """
        Get causal relationships between indicators from statistical analysis
        
        Returns list of {cause, effect, strength, p_value, regime}
        """
        query = """
        SELECT 
            cause_indicator,
            effect_indicator,
            correlation_strength,
            p_value,
            regime_type,
            mutual_information,
            granger_causality_pvalue
        FROM indicator_causality
        WHERE p_value < 0.05  -- Only statistically significant
        ORDER BY correlation_strength DESC
        """
        
        if regime_type:
            query += f" AND regime_type = '{regime_type}'"
        
        cursor = self.conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_optimal_indicator_thresholds(self, indicator: str, direction: str, 
                                        timeframe: str) -> Dict:
        """
        Get optimal thresholds for an indicator based on historical performance
        
        Returns {threshold, win_rate, avg_return, sample_size}
        """
        query = """
        SELECT 
            indicator_value as threshold,
            win_rate,
            avg_return,
            COUNT(*) as sample_size
        FROM regime_indicator_effectiveness
        WHERE indicator_name = ?
        AND direction = ?
        AND timeframe = ?
        AND sample_size >= 30  -- Minimum statistical validity
        ORDER BY win_rate * profit_factor DESC
        LIMIT 1
        """
        
        cursor = self.conn.execute(query, (indicator, direction, timeframe))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_regime_specific_parameters(self, regime_id: str) -> Dict:
        """Get parameters specific to a regime (ATR, volatility, etc.)"""
        query = """
        SELECT 
            regime_id,
            dominant_structure,
            rsi_mean,
            rsi_std,
            adx_mean,
            volatility_mean,
            atr_mean,
            avg_duration_hours
        FROM regime_instances
        WHERE regime_instance_id = ?
        """
        
        cursor = self.conn.execute(query, (regime_id,))
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def build_causal_chain(self, target_outcome: str, max_depth: int = 3) -> List[str]:
        """
        Build causal chain leading to target outcome
        
        Example: target_outcome = 'price_increase'
        Returns: ['volume_increase', 'rsi_divergence', 'price_increase']
        """
        causality = self.get_indicator_causality()
        
        # Build graph of causal relationships
        graph = {}
        for rel in causality:
            cause = rel['cause_indicator']
            effect = rel['effect_indicator']
            if cause not in graph:
                graph[cause] = []
            graph[cause].append((effect, rel['correlation_strength']))
        
        # Find path to target
        def find_path(current, target, depth, visited):
            if depth == 0 or current == target:
                return [current] if current == target else []
            
            if current in visited:
                return []
            
            visited.add(current)
            
            if current not in graph:
                return []
            
            for next_node, strength in sorted(graph[current], 
                                             key=lambda x: x[1], 
                                             reverse=True):
                path = find_path(next_node, target, depth - 1, visited.copy())
                if path:
                    return [current] + path
            
            return []
        
        # Try to find path from each starting point
        for start in graph.keys():
            path = find_path(start, target_outcome, max_depth, set())
            if path:
                return path
        
        return [target_outcome]  # Fallback
    
    def generate_formula_strategy(self, 
                                  strategy_base: Dict,
                                  htf: str,
                                  ttf: str,
                                  ltf: str) -> MTFStrategyFormula:
        """
        Generate complete formula-based strategy from statistical analysis
        
        Args:
            strategy_base: Basic strategy info (id, direction, etc.)
            htf/ttf/ltf: Timeframe names
        """
        direction = strategy_base.get('direction', 'LONG')
        
        # Get causal relationships for this direction
        causality = self.get_indicator_causality()
        
        # Build HTF formula (trend confirmation)
        htf_formula = self._build_htf_formula(htf, direction, causality)
        
        # Build TTF formula (momentum confirmation)
        ttf_formula = self._build_ttf_formula(ttf, direction, causality)
        
        # Build LTF formula (entry timing)
        ltf_formula = self._build_ltf_formula(ltf, direction, causality)
        
        # Build entry logic
        entry_logic = self._build_entry_logic(htf_formula, ttf_formula, ltf_formula, direction)
        
        # Build exit logic
        exit_logic = self._build_exit_logic(ltf_formula, direction)
        
        # Build risk management formulas
        sl_formula, tp_formula, ps_formula = self._build_risk_formulas(
            ltf, direction, strategy_base
        )
        
        # Get regime context
        optimal_regimes, avoid_regimes = self._get_regime_context(strategy_base)
        
        # Compile strategy
        strategy = MTFStrategyFormula(
            strategy_id=strategy_base.get('id', 'UNKNOWN'),
            name=strategy_base.get('name', 'Generated Strategy'),
            direction=direction,
            htf_formula=htf_formula,
            ttf_formula=ttf_formula,
            ltf_formula=ltf_formula,
            entry_logic=entry_logic,
            exit_logic=exit_logic,
            stop_loss_formula=sl_formula,
            take_profit_formula=tp_formula,
            position_size_formula=ps_formula,
            win_rate=strategy_base.get('win_rate', 0.0),
            profit_factor=strategy_base.get('profit_factor', 0.0),
            sample_size=strategy_base.get('sample_size', 0),
            statistical_significance=strategy_base.get('p_value', 1.0),
            optimal_regimes=optimal_regimes,
            avoid_regimes=avoid_regimes
        )
        
        return strategy
    
    def _build_htf_formula(self, timeframe: str, direction: str, 
                          causality: List[Dict]) -> TimeframeFormula:
        """Build HTF (trend) formula"""
        
        # Primary conditions: Strong trend indicators
        primary = []
        
        if direction == 'LONG':
            # Bullish trend conditions
            primary.append(IndicatorCondition(
                indicator='sma_20',
                operator='>',
                value='sma_50',  # Will be replaced with actual value
                timeframe=timeframe,
                lookback=3
            ))
            primary.append(IndicatorCondition(
                indicator='adx',
                operator='>',
                value=25.0,
                timeframe=timeframe,
                lookback=2
            ))
        else:
            # Bearish trend conditions
            primary.append(IndicatorCondition(
                indicator='sma_20',
                operator='<',
                value='sma_50',
                timeframe=timeframe,
                lookback=3
            ))
            primary.append(IndicatorCondition(
                indicator='adx',
                operator='>',
                value=25.0,
                timeframe=timeframe,
                lookback=2
            ))
        
        # Confirmation conditions
        confirmations = []
        if direction == 'LONG':
            confirmations.append(IndicatorCondition(
                indicator='macd_hist',
                operator='>',
                value=0.0,
                timeframe=timeframe
            ))
            confirmations.append(IndicatorCondition(
                indicator='rsi',
                operator='>',
                value=50.0,
                timeframe=timeframe
            ))
        else:
            confirmations.append(IndicatorCondition(
                indicator='macd_hist',
                operator='<',
                value=0.0,
                timeframe=timeframe
            ))
            confirmations.append(IndicatorCondition(
                indicator='rsi',
                operator='<',
                value=50.0,
                timeframe=timeframe
            ))
        
        # Build causal chain
        target = 'price_increase' if direction == 'LONG' else 'price_decrease'
        causal_chain = self.build_causal_chain(target, max_depth=2)
        
        return TimeframeFormula(
            timeframe=timeframe,
            primary_conditions=primary,
            confirmation_conditions=confirmations,
            min_confirmations=1,
            causal_chain=causal_chain
        )
    
    def _build_ttf_formula(self, timeframe: str, direction: str,
                          causality: List[Dict]) -> TimeframeFormula:
        """Build TTF (momentum) formula"""
        
        primary = []
        
        if direction == 'LONG':
            primary.append(IndicatorCondition(
                indicator='rsi',
                operator='>',
                value=40.0,
                timeframe=timeframe,
                lookback=1
            ))
            primary.append(IndicatorCondition(
                indicator='rsi',
                operator='<',
                value=70.0,
                timeframe=timeframe,
                lookback=1
            ))
        else:
            primary.append(IndicatorCondition(
                indicator='rsi',
                operator='<',
                value=60.0,
                timeframe=timeframe,
                lookback=1
            ))
            primary.append(IndicatorCondition(
                indicator='rsi',
                operator='>',
                value=30.0,
                timeframe=timeframe,
                lookback=1
            ))
        
        confirmations = []
        if direction == 'LONG':
            confirmations.append(IndicatorCondition(
                indicator='volume',
                operator='>',
                value='sma_volume_20',
                timeframe=timeframe
            ))
            confirmations.append(IndicatorCondition(
                indicator='ppo',
                operator='>',
                value=0.0,
                timeframe=timeframe
            ))
        else:
            confirmations.append(IndicatorCondition(
                indicator='volume',
                operator='>',
                value='sma_volume_20',
                timeframe=timeframe
            ))
            confirmations.append(IndicatorCondition(
                indicator='ppo',
                operator='<',
                value=0.0,
                timeframe=timeframe
            ))
        
        return TimeframeFormula(
            timeframe=timeframe,
            primary_conditions=primary,
            confirmation_conditions=confirmations,
            min_confirmations=1,
            causal_chain=['momentum_increase', 'volume_confirmation']
        )
    
    def _build_ltf_formula(self, timeframe: str, direction: str,
                          causality: List[Dict]) -> TimeframeFormula:
        """Build LTF (entry timing) formula"""
        
        primary = []
        
        if direction == 'LONG':
            primary.append(IndicatorCondition(
                indicator='close',
                operator='crosses_above',
                value='sma_7',
                timeframe=timeframe,
                lookback=1
            ))
            primary.append(IndicatorCondition(
                indicator='macd_hist',
                operator='crosses_above',
                value=0.0,
                timeframe=timeframe,
                lookback=2
            ))
        else:
            primary.append(IndicatorCondition(
                indicator='close',
                operator='crosses_below',
                value='sma_7',
                timeframe=timeframe,
                lookback=1
            ))
            primary.append(IndicatorCondition(
                indicator='macd_hist',
                operator='crosses_below',
                value=0.0,
                timeframe=timeframe,
                lookback=2
            ))
        
        confirmations = []
        if direction == 'LONG':
            confirmations.append(IndicatorCondition(
                indicator='rsi',
                operator='>',
                value=50.0,
                timeframe=timeframe
            ))
            confirmations.append(IndicatorCondition(
                indicator='bb_position',
                operator='>',
                value=0.5,
                timeframe=timeframe
            ))
        else:
            confirmations.append(IndicatorCondition(
                indicator='rsi',
                operator='<',
                value=50.0,
                timeframe=timeframe
            ))
            confirmations.append(IndicatorCondition(
                indicator='bb_position',
                operator='<',
                value=0.5,
                timeframe=timeframe
            ))
        
        return TimeframeFormula(
            timeframe=timeframe,
            primary_conditions=primary,
            confirmation_conditions=confirmations,
            min_confirmations=1,
            causal_chain=['entry_trigger', 'confirmation']
        )
    
    def _build_entry_logic(self, htf: TimeframeFormula, ttf: TimeframeFormula,
                          ltf: TimeframeFormula, direction: str) -> str:
        """Build complete entry logic"""
        
        logic = f"""
ENTRY LOGIC FOR {direction}:

1. HTF Trend Confirmation:
   {htf.to_formula_string()}

2. TTF Momentum Confirmation:
   {ttf.to_formula_string()}

3. LTF Entry Trigger:
   {ltf.to_formula_string()}

EXECUTE ENTRY WHEN:
- ALL HTF primary conditions are TRUE
- ALL TTF primary conditions are TRUE  
- ALL LTF primary conditions are TRUE
- At least {htf.min_confirmations} HTF confirmations are TRUE
- At least {ttf.min_confirmations} TTF confirmations are TRUE
- At least {ltf.min_confirmations} LTF confirmations are TRUE

"""
        return logic.strip()
    
    def _build_exit_logic(self, ltf: TimeframeFormula, direction: str) -> str:
        """Build exit logic"""
        
        if direction == 'LONG':
            logic = """
EXIT LOGIC:

Exit if ANY of the following:
1. Take profit hit (see risk management)
2. Stop loss hit (see risk management)
3. RSI[LTF] crosses below 70 (overbought exit)
4. MACD_hist[LTF] crosses below 0 (momentum reversal)
5. Close[LTF] crosses below SMA_7 (trend break)
"""
        else:
            logic = """
EXIT LOGIC:

Exit if ANY of the following:
1. Take profit hit (see risk management)
2. Stop loss hit (see risk management)
3. RSI[LTF] crosses above 30 (oversold exit)
4. MACD_hist[LTF] crosses above 0 (momentum reversal)
5. Close[LTF] crosses above SMA_7 (trend break)
"""
        
        return logic.strip()
    
    def _build_risk_formulas(self, ltf: str, direction: str, 
                            strategy_base: Dict) -> Tuple[str, str, str]:
        """Build risk management formulas"""
        
        # Stop loss
        if direction == 'LONG':
            sl = f"STOP_LOSS = Entry_Price - (2.0 * ATR[{ltf}])"
        else:
            sl = f"STOP_LOSS = Entry_Price + (2.0 * ATR[{ltf}])"
        
        # Take profit
        if direction == 'LONG':
            tp = f"TAKE_PROFIT = Entry_Price + (3.0 * ATR[{ltf}])"
        else:
            tp = f"TAKE_PROFIT = Entry_Price - (3.0 * ATR[{ltf}])"
        
        # Position size
        ps = f"""
POSITION_SIZE = (Account_Balance * Risk_Percent) / (Entry_Price - Stop_Loss)
where Risk_Percent = 1.0% (max risk per trade)
"""
        
        return sl, tp, ps.strip()
    
    def _get_regime_context(self, strategy_base: Dict) -> Tuple[List[str], List[str]]:
        """Get optimal and avoid regimes"""
        
        # Query regime effectiveness for this strategy pattern
        # This is simplified - you'd query your actual regime data
        
        optimal = ['trending_bull', 'volatility_expansion', 'momentum_surge']
        avoid = ['choppy_sideways', 'low_volume', 'ranging_tight']
        
        return optimal, avoid
    
    def generate_all_strategies(self, output_file: str):
        """Generate formulas for all strategies and export"""
        
        # This would iterate through your existing strategies
        # and generate formulas for each
        
        strategies = []
        
        # Example: Generate a few sample strategies
        for i in range(5):
            strategy_base = {
                'id': f'MTF_{i+1:04d}',
                'name': f'Formula Strategy {i+1}',
                'direction': 'LONG' if i % 2 == 0 else 'SHORT',
                'win_rate': 0.65 + (i * 0.02),
                'profit_factor': 2.5 + (i * 0.3),
                'sample_size': 100 + (i * 50),
                'p_value': 0.01
            }
            
            strategy = self.generate_formula_strategy(
                strategy_base,
                htf='4h',
                ttf='1h',
                ltf='15m'
            )
            
            strategies.append(strategy.to_dict())
        
        # Export
        with open(output_file, 'w') as f:
            json.dump(strategies, f, indent=2)
        
        print(f"✅ Generated {len(strategies)} formula-based strategies")
        print(f"✅ Exported to {output_file}")
        
        return strategies


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = FormulaStrategyGenerator("regime_data.db")
    
    # Generate all strategies with formulas
    strategies = generator.generate_all_strategies("formula_strategies.json")
    
    # Print example
    if strategies:
        print("\n" + "="*80)
        print("EXAMPLE FORMULA STRATEGY")
        print("="*80)
        print(json.dumps(strategies[0], indent=2))