# strategy_formulator.py
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import json
import uuid
from tqdm import tqdm
from typing import List, Dict

from ...db_connector import DatabaseConnector
# We import the Statistical Analyzer
from .statistical_analysis import RegimeStatisticalAnalyzer

class StrategyFormulator:
    """
    Automated Strategy Generator (Swing-Point Edition).
    
    Workflow:
    1. Swing Detection: Identify HH/LL/HL/LH.
    2. Regime Labelling: +/- 2% change between swings determines regime.
    3. DB Persistence: Convert these labeled segments into 'Regime Instances' in DB.
    4. Statistical Mining: Use Granger Causality to find drivers for these specific regimes.
    """

    def __init__(self, db_path: str, pair_tf: str):
        self.db_path = db_path
        self.pair_tf = pair_tf
        self.db = DatabaseConnector(db_path)
        
        # Initialize the advanced analyzer
        self.analyzer = RegimeStatisticalAnalyzer(self.db)
        
        # Config
        self.swing_order = 5      # Lookback for swing detection
        self.cf_threshold = 0.70  # (Legacy threshold, we now use p-values)
        self.mi_threshold_cf = 0.15 
        self.mi_threshold_af = 0.05

    def generate_strategy(self):
        print(f"🚀 Starting Swing-Based Strategy Formulation for {self.pair_tf}...")
        
        # STEP 1: LOAD & LABEL DATA (Memory)
        df = self._load_and_label_data()
        if df is None: return

        # STEP 2: POPULATE DB (ETL)
        # We must push these swing-based regimes to the DB so the Statistical Analyzer can query them
        print("2. Populating Database with Swing-Based Regimes...")
        self._persist_regimes_to_db(df)
        
        # STEP 3: STATISTICAL MINING
        print("3. Running Causal Analysis...")
        
        # Get indicators available in the feature set
        indicators = self._get_feature_columns(df)
        print(f"   Analyzing {len(indicators)} potential factors...")
        
        # A. Bullish (Regime 2)
        print("\n--- Analysing Bullish Factors (Regime 2) ---")
        bull_factors = self._analyze_regime_drivers(indicators, target_regime=2)
        if bull_factors['CF'] or bull_factors['AF']:
            self._save_formula("Bullish_Swing_Gen", 2, "UP", bull_factors)
        else:
            print("   No significant factors found for Bullish Swing Regime.")

        # B. Bearish (Regime 1)
        print("\n--- Analysing Bearish Factors (Regime 1) ---")
        bear_factors = self._analyze_regime_drivers(indicators, target_regime=1)
        if bear_factors['CF'] or bear_factors['AF']:
            self._save_formula("Bearish_Swing_Gen", 1, "DOWN", bear_factors)
        else:
            print("   No significant factors found for Bearish Swing Regime.")

    # --- PART 1: SWING LOGIC ---

    def _load_and_label_data(self) -> pd.DataFrame:
        print("1. Loading and Labeling Data...")
        df = self.db.load_full_features(self.pair_tf)
        if df is None or df.empty:
            print(f"❌ No data found for {self.pair_tf}")
            return None
            
        # Ensure numeric
        df = df.apply(pd.to_numeric, errors='ignore')
        
        # 1. Identify Swings
        df['swing_high'] = df.iloc[argrelextrema(df['high'].values, np.greater_equal, order=self.swing_order)[0]]['high']
        df['swing_low'] = df.iloc[argrelextrema(df['low'].values, np.less_equal, order=self.swing_order)[0]]['low']
        
        swings = []
        for ts, row in df.iterrows():
            if pd.notna(row['swing_high']):
                swings.append({'timestamp': ts, 'price': row['swing_high'], 'type': 'high'})
            elif pd.notna(row['swing_low']):
                swings.append({'timestamp': ts, 'price': row['swing_low'], 'type': 'low'})
        
        if not swings:
            print("❌ No swings identified.")
            return None
            
        swings_df = pd.DataFrame(swings)
        
        # 2. Label Regimes based on 2% Rule
        regime_map = pd.Series(index=df.index, data=0) # Default Neutral
        
        for i in range(len(swings_df) - 1):
            curr = swings_df.iloc[i]
            nxt = swings_df.iloc[i+1]
            
            # Calculate % change between swing points
            pct_change = (nxt['price'] - curr['price']) / curr['price']
            
            label = 0
            if pct_change >= 0.02: # Bullish Regime
                label = 2
            elif pct_change <= -0.02: # Bearish Regime
                label = 1
            
            # Apply label to all bars BETWEEN these two swings
            if label != 0:
                mask = (df.index >= curr['timestamp']) & (df.index < nxt['timestamp'])
                regime_map[mask] = label
                
        df['regime'] = regime_map
        
        dist = df['regime'].value_counts()
        print(f"   Regime Distribution: {dist.to_dict()}")
        return df

    # --- PART 2: DB PERSISTENCE ---

    def _persist_regimes_to_db(self, df: pd.DataFrame):
        """
        Groups consecutive bars of the same regime into 'Instances' and saves to DB.
        This enables the StatisticalAnalyzer to work its magic.
        """
        # Group consecutive regimes
        df['grp'] = (df['regime'] != df['regime'].shift()).cumsum()
        grouped = df.groupby('grp')
        
        regime_instances = []
        indicator_records = []
        feature_cols = self._get_feature_columns(df)
        
        for _, group in tqdm(grouped, desc="Persisting Regimes"):
            regime_id = int(group.iloc[0]['regime'])
            
            # We only care about Bull(2) or Bear(1) for causal analysis
            # Neutral(0) is usually noise, but we can store it if needed. 
            # Let's focus on 1 and 2 to keep DB clean.
            if regime_id == 0: continue
            
            start_row = group.iloc[0]
            end_row = group.iloc[-1]
            
            instance_id = str(uuid.uuid4())
            duration_bars = len(group)
            
            # Calculate 'Outcome' for Granger Test
            # The return of the regime itself is the primary outcome
            regime_return = (end_row['close'] - start_row['open']) / start_row['open']
            
            regime_instances.append((
                instance_id,
                self.pair_tf.split('_')[0],
                self.pair_tf.split('_')[-1],
                group.index[0].isoformat(),
                group.index[-1].isoformat(),
                float(duration_bars),
                str(regime_id),
                float(regime_return) # stored in 'next_1d_return_pct' column for compatibility
            ))
            
            # Store Feature Stats for this Instance
            # The Analyzer needs to know the state of indicators during this regime
            for col in feature_cols:
                avg_val = group[col].mean()
                indicator_records.append((
                    instance_id,
                    col,
                    float(avg_val)
                ))
        
        # Batch Insert
        self._bulk_insert_instances(regime_instances)
        self._bulk_insert_indicators(indicator_records)
        print(f"   ✅ Saved {len(regime_instances)} regime instances to DB.")

    def _bulk_insert_instances(self, records):
        if not records: return
        query = """
        INSERT OR REPLACE INTO regime_instances 
        (instance_id, pair, timeframe, start_time, end_time, duration_hours, dominant_structure, next_1d_return_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        chunk = 1000
        for i in range(0, len(records), chunk):
            self.db.execute_batch(query, records[i:i+chunk])

    def _bulk_insert_indicators(self, records):
        if not records: return
        query = """
        INSERT OR REPLACE INTO regime_confirming_indicators 
        (instance_id, indicator_name, confirmation_strength)
        VALUES (?, ?, ?)
        """
        chunk = 5000
        for i in range(0, len(records), chunk):
            self.db.execute_batch(query, records[i:i+chunk])

    # --- PART 3: ANALYSIS & SYNTHESIS ---

    def _analyze_regime_drivers(self, indicators, target_regime):
        cfs = []
        afs = []
        
        for ind in tqdm(indicators, desc=f"Analyzing Drivers (R{target_regime})"):
            # 1. Direction (High vs Low)
            direction = self.analyzer.get_correlation_direction(ind, target_regime)
            if direction == "UNKNOWN": continue

            # 2. Causal Tests
            # Note: 'next_1d_return_pct' in DB now holds the Regime Return
            mi_res = self.analyzer.mutual_info_test(ind, outcome='next_1d_return_pct')
            granger_res = self.analyzer.granger_causality_test(ind, outcome='next_1d_return_pct')
            
            is_cf = False
            is_af = False
            reason = ""
            
            # Classification Logic
            if granger_res.get('is_causal'):
                is_cf = True
                reason = f"Granger Causal (p={granger_res.get('p_value'):.4f})"
            elif mi_res.get('valid') and mi_res['mutual_info'] > self.mi_threshold_cf:
                is_cf = True
                reason = f"High Mutual Info ({mi_res['mutual_info']:.3f})"
            elif mi_res.get('valid') and mi_res['mutual_info'] > self.mi_threshold_af:
                is_af = True
                reason = f"Moderate Mutual Info ({mi_res['mutual_info']:.3f})"
                
            if is_cf or is_af:
                factor = {
                    'feature': ind,
                    'relation': direction,
                    'reason': reason
                }
                if is_cf: cfs.append(factor)
                else: afs.append(factor)
                
        return {'CF': cfs, 'AF': afs}

    def _save_formula(self, name: str, regime_id: int, direction: str, factors: Dict):
        cfs = factors.get('CF', [])
        afs = factors.get('AF', [])
        
        # Build Logic String
        cf_str = " AND ".join([f"{f['feature']} is {f['relation']}" for f in cfs])
        af_str = " OR ".join([f"{f['feature']} is {f['relation']}" for f in afs])
        
        formula = ""
        if cfs: formula += f"({cf_str})"
        if cfs and afs: formula += " AND "
        if afs: formula += f"({af_str})"
            
        print(f"\n✅ Formula Generated for {name}:")
        print(formula)
        
        # Save to Playbook
        strategy_json = {
            "logic_version": "2.0-Swing-Causal",
            "causing_factors": cfs,
            "affecting_factors": afs,
            "raw_formula": formula
        }
        
        record = {
            "regime_name": name,
            "regime_id": regime_id,
            "trend_direction": direction,
            "volatility_level": "AUTO",
            "confirming_indicators_json": json.dumps(strategy_json),
            "strategy_patterns_json": json.dumps({"source": "swing_statistical_mining"}),
            "last_updated": pd.Timestamp.now().isoformat()
        }
        self.db.upsert_strategy_playbook(record)

    def _get_feature_columns(self, df):
        exclude = ['open','high','low','close','volume','timestamp','regime','grp','swing_high','swing_low']
        return [c for c in df.columns if c not in exclude]

if __name__ == "__main__":
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Update with your actual DB path
    DB_PATH = "./data/auto_trader_db.sqlite"
    PAIR = "btc_usdt_1h"
    
    try:
        formulator = StrategyFormulator(DB_PATH, PAIR)
        formulator.generate_strategy()
    except Exception as e:
        print(f"Formulator failed: {e}")