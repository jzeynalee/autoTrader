import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from chart_patterns import CHART_PATTERN_FUNCS

class LBankDataFetcher:
    """Fetches K-line data from LBANK exchange and calculates technical indicators from scratch"""    
    def __init__(self):
        self.MAX_LIMIT = 2000
        self.BASE_URL = "https://api.lbkex.com/v2/kline.do"
        self.pairs = ['btc_usdt', 'eth_usdt', 'sol_usdt', 'trx_usdt', 'doge_usdt']
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.TIMEFRAME_MAP = {
            '1m': 'minute1',
            '5m': 'minute5',
            '15m': 'minute15',
            '1h': 'hour1',
            '4h': 'hour4'
        }
        self.data = {}
    
    
    def fetch_klines_paginated(self, symbol, timeframe, days=30):  # Start with 30 days instead of 365
        """
        Fetch K-line data from LBANK for specified number of days with pagination
        """
        all_data = []
        
        # Calculate time ranges in SECONDS
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60)
        
        print(f"Fetching {symbol} {timeframe} data for {days} days...")
        print(f"Time range: {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
        
        # Try single request first
        params = {
            'symbol': symbol,
            'type': self.TIMEFRAME_MAP[timeframe],
            'size': self.MAX_LIMIT,
            'time': start_time  # Use start_time, not end_time
        }
        
        try:
            print(f"  Request 1: time={datetime.fromtimestamp(start_time)}, size={self.MAX_LIMIT}")
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            '''if data.get('result') == 'true' and 'data' in data and data['data']:
                batch_data = data['data']
                
                if not batch_data:
                    print(f"    ✗ Empty data batch")
                
                # DEBUG: Print first candle to see structure
                print(f"    DEBUG: First candle raw data: {batch_data[0]}")
                print(f"    DEBUG: Timestamp: {batch_data[0][0]}, Date: {datetime.fromtimestamp(batch_data[0][0]/1000)}")
                
                # Add batch data
                all_data.extend(batch_data)'''
            
            # DEBUG: Print raw response
            print(f"    DEBUG: API response result={data.get('result')}")
            print(f"    DEBUG: API response keys={data.keys()}")
            
            if data.get('result') == 'true' and 'data' in data and data['data']:
                batch_data = data['data']
                
                print(f"    DEBUG: Received {len(batch_data)} candles")
                if len(batch_data) > 0:
                    print(f"    DEBUG: First candle: {batch_data[0]}")
                    print(f"    DEBUG: Last candle: {batch_data[-1]}")
                
                all_data.extend(batch_data)
                    
            else:
                error_msg = data.get('msg', 'Unknown error')
                error_code = data.get('error_code', 'Unknown')
                print(f"    ✗ API Error {error_code}: {error_msg}")
                print(f"    DEBUG: Full response: {data}")
                return None
                
        except Exception as e:
            print(f"    ✗ Exception: {str(e)}")
            return None
        
        if not all_data:
            print(f"  ✗ No data fetched for {symbol} {timeframe}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        print(f"    DEBUG: DataFrame shape before processing: {df.shape}")
        print(f"    DEBUG: DataFrame columns: {df.columns.tolist()}")
        print(f"    DEBUG: First few timestamp values: {df[0].head().tolist()}")
        
        # Remove duplicates based on timestamp
        df = df.drop_duplicates(subset=[0])
        
        # Remove invalid timestamps (0 or negative)
        df = df[df[0] > 0]
        
        print(f"    DEBUG: DataFrame shape after filtering: {df.shape}")
        
        if len(df) == 0:
            print(f"  ✗ All data filtered out - timestamps were invalid")
            return None
        
        # Rename columns
        df = df.rename(columns={
            0: 'timestamp',
            1: 'open',
            2: 'high',
            3: 'low',
            4: 'close',
            5: 'volume'
        })
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  ✓ Final dataset: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df


    def fetch_all_data(self, use_pagination=True):
        """Fetch all trading pairs and timeframes"""
        print("Fetching K-line data from LBANK...")
        
        for pair in self.pairs:
            print(f"\nProcessing {pair.upper()}...")
            self.data[pair] = {}
            
            for tf in self.timeframes:
                print(f"  Fetching {tf} data...")
                
                if use_pagination:
                    df = self.fetch_klines_paginated(pair, tf, days=365)
                else:
                    df = self.fetch_klines(pair, tf)
                
                if df is not None and len(df) > 0:
                    print(f"  Calculating indicators for {tf}...")
                    df = self.calculate_indicators(df)
                    self.data[pair][tf] = df
                    print(f"  ✓ {pair} {tf}: {len(df)} candles with {len(df.columns)} features")
                else:
                    print(f"  ✗ Failed to fetch {pair} {tf}")
                
                # Be respectful to API
                time.sleep(1)
        
        return self.data
    
    def fetch_klines(self, symbol, timeframe, limit=2000):
        """Single request version with correct time parameter"""
        try:
            # Calculate start time (1 year ago from now) in SECONDS
            current_time = int(time.time())
            start_time = current_time - (365 * 24 * 60 * 60)  # 1 year ago in SECONDS
            
            params = {
                'symbol': symbol,
                'type': self.TIMEFRAME_MAP[timeframe],
                'size': limit,
                'time': start_time  # In SECONDS
            }
            
            print(f"Fetching {symbol} {timeframe} from {datetime.fromtimestamp(start_time)}")
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('result') == 'true' and 'data' in data:
                klines = data['data']
                if not klines:
                    print(f"No data returned for {symbol} {timeframe}")
                    return None
                    
                df = pd.DataFrame(klines)
                
                # Rename columns to standard format
                df = df.rename(columns={
                    0: 'timestamp',
                    1: 'open',
                    2: 'high',
                    3: 'low',
                    4: 'close',
                    5: 'volume'
                })
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"✓ Fetched {len(df)} candles for {symbol} {timeframe}")
                return df
            else:
                error_msg = data.get('msg', 'Unknown error')
                error_code = data.get('error_code', 'Unknown')
                print(f"Error fetching {symbol} {timeframe}: {error_code} - {error_msg}")
                return None
                
        except Exception as e:
            print(f"Exception fetching {symbol} {timeframe}: {str(e)}")
            return None
    
    # ============ HELPER FUNCTIONS ============
    @staticmethod
    def sma(series, period):
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series, period):
        """Return EMA; auto-convert numpy arrays to pandas Series."""
        import pandas as pd
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(series, period):
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    @staticmethod
    def rma(series, period):
        """Rolling Moving Average (Wilder's smoothing)"""
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def std(series, period):
        """Standard Deviation"""
        return series.rolling(window=period).std()
    
    @staticmethod
    def typical_price(high, low, close):
        """Typical Price"""
        return (high + low + close) / 3
    
    @staticmethod
    def true_range(high, low, close):
        """True Range"""
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr
    
    # ============ INDICATOR CALCULATIONS ============
    
    def calculate_indicators(self, df):
        """Calculate comprehensive technical indicators from scratch"""
        if df is None or len(df) < 50:
            print("⚠️  Insufficient data for indicator calculation")
            return df
        
        df = df.copy()
        new_cols = {}     # collect all new indicator columns here

        # Calculate True Range first as it's used in multiple indicators
        tr = self.true_range(df['high'], df['low'], df['close'])
        # ============ Trend Indicators ============
        
        # ============ MOVING AVERAGES ============        
        # Simple Moving Averages # 1 ---> 5
        new_cols['sma_7'] = self.sma(df['close'], 7)
        new_cols['sma_20'] = self.sma(df['close'], 20)
        new_cols['sma_50'] = self.sma(df['close'], 50)
        new_cols['sma_100'] = self.sma(df['close'], 100)
        new_cols['sma_200'] = self.sma(df['close'], 200)
        
        # Exponential Moving Averages # 6 ---> 10
        new_cols['ema_9'] = self.ema(df['close'], 9)
        new_cols['ema_12'] = self.ema(df['close'], 12)
        new_cols['ema_26'] = self.ema(df['close'], 26)
        new_cols['ema_50'] = self.ema(df['close'], 50)
        new_cols['ema_200'] = self.ema(df['close'], 200)
        
        # Weighted Moving Average # 11
        new_cols['wma_20'] = self.wma(df['close'], 20)
        
        # Double EMA (DEMA) # 12
        ema = self.ema(df['close'], 20)
        ema_ema = self.ema(ema, 20)
        new_cols['dema'] = 2 * ema - ema_ema
        
        # Triple EMA (TEMA) # 13
        ema1 = self.ema(df['close'], 20)
        ema2 = self.ema(ema1, 20)
        ema3 = self.ema(ema2, 20)
        new_cols['tema'] = 3 * ema1 - 3 * ema2 + ema3
        
        # Hull Moving Average (HMA) # 14
        half_length = 10
        sqrt_length = int(np.sqrt(20))
        wma1 = self.wma(df['close'], half_length)
        wma2 = self.wma(df['close'], 20)
        raw_hma = 2 * wma1 - wma2
        new_cols['hma'] = self.wma(raw_hma, sqrt_length)
        
        # Smoothed Moving Average (SMMA/RMA) # 15
        new_cols['smma'] = self.rma(df['close'], 14)
        
        # Least Squares Moving Average (LSMA) # 16
        def lsma(series, period):
            result = []
            for i in range(len(series)):
                if i < period - 1:
                    result.append(np.nan)
                else:
                    y = series.iloc[i-period+1:i+1].values
                    x = np.arange(period)
                    z = np.polyfit(x, y, 1)
                    result.append(z[0] * (period - 1) + z[1])
            return pd.Series(result, index=series.index)
        
        new_cols['lsma'] = lsma(df['close'], 25)
                
        # ============ MACD ============        
        macd_line = new_cols['ema_12'] - new_cols['ema_26']
        new_cols['macd'] = macd_line
        new_cols['macd_signal'] = self.ema(macd_line, 9)
        new_cols['macd_hist'] = macd_line - new_cols['macd_signal']

        # ============ ADX (Average Directional Index) ============        
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        pos_dm = pd.Series(pos_dm, index=df.index)
        neg_dm = pd.Series(neg_dm, index=df.index)
        
        atr_14 = self.rma(tr, 14)
        
        di_plus = 100 * self.rma(pos_dm, 14) / atr_14
        di_minus = 100 * self.rma(neg_dm, 14) / atr_14
        new_cols['di_plus'] = di_plus  # 20
        new_cols['di_minus'] = di_minus  # 21
        # compute DX / ADX from the DI series (use local vars)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        new_cols['adx'] = self.rma(dx, 14)  # 45

        # ============ AROON ============        
        def aroon_up(high, period=25):
            result = []
            for i in range(len(high)):
                if i < period - 1:
                    result.append(np.nan)
                else:
                    window = high.iloc[i-period+1:i+1]
                    periods_since_high = period - 1 - window.values.argmax()
                    result.append(((period - periods_since_high) / period) * 100)
            return pd.Series(result, index=high.index)
        
        def aroon_down(low, period=25):
            result = []
            for i in range(len(low)):
                if i < period - 1:
                    result.append(np.nan)
                else:
                    window = low.iloc[i-period+1:i+1]
                    periods_since_low = period - 1 - window.values.argmin()
                    result.append(((period - periods_since_low) / period) * 100)
            return pd.Series(result, index=low.index)
        
        new_cols['aroon_up'] = aroon_up(df['high'], 25) # 22
        new_cols['aroon_down'] = aroon_down(df['low'], 25) # 23
        aroon_up = new_cols['aroon_up']
        aroon_down = new_cols['aroon_down']
        new_cols['aroon_osc'] = aroon_up - aroon_down  # 24

        # Collection Stage 1 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        # ============ ICHIMOKU CLOUD ============        
        # Tenkan-sen (Conversion Line)
        period9_high = df['high'].rolling(window=9).max()
        period9_low = df['low'].rolling(window=9).min()
        new_cols['ichimoku_conversion'] = (period9_high + period9_low) / 2 # 25
        
        # Kijun-sen (Base Line)
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        new_cols['ichimoku_base'] = (period26_high + period26_low) / 2 # 26

        # Collection Stage 2 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        
        # Senkou Span A (Leading Span A)
        new_cols['ichimoku_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(26) # 27
        
        # Senkou Span B (Leading Span B)
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        new_cols['ichimoku_b'] = ((period52_high + period52_low) / 2).shift(26) # 28
        
        # Chikou Span (Lagging Span)
        new_cols['ichimoku_lagging'] = df['close'].shift(-26) # 29

        # ============ PARABOLIC SAR ============        
        def parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
            psar = close.copy()
            bull = True
            af = af_start
            ep = low.iloc[0]
            hp = high.iloc[0]
            lp = low.iloc[0]            
            for i in range(1, len(close)):
                if bull:
                    psar.iloc[i] = psar.iloc[i-1] + af * (hp - psar.iloc[i-1])
                    if low.iloc[i] < psar.iloc[i]:
                        bull = False
                        psar.iloc[i] = hp
                        lp = low.iloc[i]
                        af = af_start
                else:
                    psar.iloc[i] = psar.iloc[i-1] + af * (lp - psar.iloc[i-1])
                    if high.iloc[i] > psar.iloc[i]:
                        bull = True
                        psar.iloc[i] = lp
                        hp = high.iloc[i]
                        af = af_start                
                if bull:
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
                        af = min(af + af_increment, af_max)
                    if low.iloc[i-1] < psar.iloc[i]:
                        psar.iloc[i] = low.iloc[i-1]
                    if low.iloc[i-2] < psar.iloc[i]:
                        psar.iloc[i] = low.iloc[i-2]
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + af_increment, af_max)
                    if high.iloc[i-1] > psar.iloc[i]:
                        psar.iloc[i] = high.iloc[i-1]
                    if high.iloc[i-2] > psar.iloc[i]:
                        psar.iloc[i] = high.iloc[i-2]            
            return psar        
        new_cols['psar'] = parabolic_sar(df['high'], df['low'], df['close']) # 30
                
        # ============ TRIX ============        
        ema1 = self.ema(df['close'], 15)
        ema2 = self.ema(ema1, 15)
        ema3 = self.ema(ema2, 15)
        new_cols['trix'] = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100 # 31
        # Collection Stage 3 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        new_cols['trix_signal'] = self.sma(df['trix'], 9) # 32

        # ============ MASS INDEX ============
        hl_range = df['high'] - df['low']
        ema9 = self.ema(hl_range, 9)
        ema9_ema9 = self.ema(ema9, 9)
        mass_ratio = ema9 / ema9_ema9
        new_cols['mass_index'] = mass_ratio.rolling(window=25).sum() # 33

        # ===============DPO (Detrended Price Oscillator)=========
        def dpo(series: pd.Series, period: int = 20) -> pd.Series:
            """
            Calculate Detrended Price Oscillator (DPO).
            
            Args:
                series (pd.Series): Price series (e.g., close prices).
                period (int): Lookback period for SMA.
            
            Returns:
                pd.Series: DPO values.
            """
            shift = int(period / 2 + 1)
            sma = series.rolling(window=period).mean()
            dpo = series.shift(shift) - sma
            return dpo
        df["dpo"] = dpo(df["close"], period=20) # 34
                
        # ============ KST (Know Sure Thing) ============        
        roc1 = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        roc2 = ((df['close'] - df['close'].shift(15)) / df['close'].shift(15)) * 100
        roc3 = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
        roc4 = ((df['close'] - df['close'].shift(30)) / df['close'].shift(30)) * 100        
        new_cols['kst'] = (self.sma(roc1, 10) * 1) + (self.sma(roc2, 10) * 2) + (self.sma(roc3, 10) * 3) + (self.sma(roc4, 15) * 4) # 69
        # Collection Stage 4 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        new_cols['kst_signal'] = self.sma(df['kst'], 9) # 35
        
        # Collection Stage 1 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # ============ Momentum Indicators ============
        
        # McGinley Dynamic # 17
        new_cols['mcginley'] = df['close'].copy()
        # Collection Stage 1 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        for i in range(1, len(df)):
            if pd.notna(df['mcginley'].iloc[i-1]):
                df.loc[df.index[i], 'mcginley'] = df['mcginley'].iloc[i-1] + (
                    (df['close'].iloc[i] - df['mcginley'].iloc[i-1]) / 
                    (20 * (df['close'].iloc[i] / df['mcginley'].iloc[i-1]) ** 4)
                )
        
        # Volume Weighted Moving Average (VWMA) # 18
        new_cols['vwma'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()  # 36

        
        # ============ RSI ============        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)        
        avg_gain = self.rma(gain, 14)
        avg_loss = self.rma(loss, 14)        
        rs = avg_gain / avg_loss
        new_cols['rsi'] = 100 - (100 / (1 + rs)) # 37
        new_cols['rsi_30'] = 100 - (100 / (1 + self.rma(gain, 30) / self.rma(loss, 30))) # 38
        
        # Collection Stage 6 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        # Connors RSI
        # Component 1: Standard RSI
        rsi_component = df['rsi'].copy()        
        # Component 2: RSI of streak
        streak = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                streak.iloc[i] = max(1, streak.iloc[i-1] + 1) if streak.iloc[i-1] > 0 else 1
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                streak.iloc[i] = min(-1, streak.iloc[i-1] - 1) if streak.iloc[i-1] < 0 else -1
        
        streak_delta = streak.diff()
        streak_gain = streak_delta.where(streak_delta > 0, 0)
        streak_loss = -streak_delta.where(streak_delta < 0, 0)
        streak_avg_gain = self.rma(streak_gain, 2)
        streak_avg_loss = self.rma(streak_loss, 2)
        streak_rs = streak_avg_gain / streak_avg_loss
        rsi_streak = 100 - (100 / (1 + streak_rs))
        
        # Component 3: Percent rank of ROC
        roc = df['close'].pct_change(1) * 100
        pct_rank = roc.rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
        
        new_cols['connors_rsi'] = (rsi_component + rsi_streak + pct_rank) / 3 # 39
        
        # ============ STOCHASTIC ============        
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        
        new_cols['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min) # 40
        # Collection Stage 7 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        new_cols['stoch_d'] = self.sma(df['stoch_k'], 3) # 41
        
        # Stochastic RSI
        rsi_min = df['rsi'].rolling(window=14).min()
        rsi_max = df['rsi'].rolling(window=14).max()
        new_cols['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min) # 42
        # Collection Stage 8 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        new_cols['stoch_rsi_k'] = self.sma(df['stoch_rsi'], 3) * 100 # 43
        # Collection Stage 9 ,merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        new_cols['stoch_rsi_d'] = self.sma(df['stoch_rsi_k'], 3) # 44
        
        # Stochastic Momentum Index (SMI)
        ll = df['low'].rolling(window=14).min()
        hh = df['high'].rolling(window=14).max()
        diff = hh - ll
        rdiff = df['close'] - (hh + ll) / 2        
        avgrel = self.ema(self.ema(rdiff, 3), 3)
        avgdiff = self.ema(self.ema(diff, 3), 3)        
        new_cols['smi'] = np.where(avgdiff != 0, (avgrel / (avgdiff / 2)) * 100, 0) # 45
        new_cols['smi_signal'] = self.ema(new_cols['smi'], 3) # 46
        # SMI Ergodic
        new_cols['smi_ergodic'] = new_cols['smi'] # 47
        new_cols['smi_ergodic_signal'] = self.ema(new_cols['smi_ergodic'], 5) # 48
        
        # ============ WILLIAMS %R ========        
        highest_high = df['high'].rolling(window=14).max()
        lowest_low = df['low'].rolling(window=14).min()
        new_cols['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low) # 49

        # ============ ROC (Rate of Change) ============        
        new_cols['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100  # 50
        new_cols['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100 # 51
        
        # ============ AWESOME OSCILLATOR ============        
        median_price = (df['high'] + df['low']) / 2
        new_cols['ao'] = self.sma(median_price, 5) - self.sma(median_price, 34) # 52

        # ===========KAMA (Kaufman's Adaptive Moving Average)=================
        def kama(series: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
            """
            Kaufman's Adaptive Moving Average (KAMA).            
            Args:
                series (pd.Series): Price series (e.g., close prices).
                er_period (int): Lookback period for efficiency ratio.
                fast (int): Fast EMA constant (default=2).
                slow (int): Slow EMA constant (default=30).
            
            Returns:
                pd.Series: KAMA values.
            """
            # Precompute smoothing constants
            sc_fast = 2 / (fast + 1)
            sc_slow = 2 / (slow + 1)

            kama = np.zeros(len(series))
            kama[:] = np.nan

            for i in range(er_period, len(series)):
                # Efficiency Ratio (ER)
                change = abs(series.iloc[i] - series.iloc[i - er_period])
                volatility = np.sum(np.abs(series.iloc[i - er_period + 1:i + 1].diff()))
                er = change / volatility if volatility != 0 else 0

                # Smoothing Constant (SC)
                sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2

                # Initialize first KAMA value
                if np.isnan(kama[i - 1]):
                    kama[i] = series.iloc[i]
                else:
                    kama[i] = kama[i - 1] + sc * (series.iloc[i] - kama[i - 1])

            return pd.Series(kama, index=series.index)
        df["kama"] = kama(df["close"], er_period=10, fast=2, slow=30)  # 53

        # ==========Percentage Price Oscillator (PPO)========================
        def ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
            """
            Percentage Price Oscillator (PPO).
            
            Args:
                series (pd.Series): Price series (e.g., close prices).
                fast (int): Fast EMA period (default=12).
                slow (int): Slow EMA period (default=26).
                signal (int): Signal EMA period (default=9).
            
            Returns:
                tuple: (ppo, ppo_signal, ppo_hist)
            """
            ema_fast = self.ema(series, fast)
            ema_slow = self.ema(series, slow)

            ppo_line = (ema_fast - ema_slow) / ema_slow * 100
            ppo_signal = self.ema(ppo_line, signal)
            ppo_hist = ppo_line - ppo_signal

            return ppo_line, ppo_signal, ppo_hist
        df["ppo"], df["ppo_signal"], df["ppo_hist"] = ppo(df["close"], fast=12, slow=26, signal=9)   # 54

        # ============ ULTIMATE OSCILLATOR ============        
        bp = df['close'] - pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)        
        avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
        avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
        avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()        
        new_cols['ultimate_osc'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7 # 55

        # ===============TSI (True Strength Index)=======================
        def tsi(series: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
            """
            True Strength Index (TSI).
            
            Args:
                series (pd.Series): Price series (e.g., close prices).
                r (int): First EMA period (default=25).
                s (int): Second EMA period (default=13).
            
            Returns:
                pd.Series: TSI values.
            """
            # Momentum
            m = series.diff()
            # Double-smoothed momentum
            ema1 = self.ema(m, r)
            ema2 = self.ema(ema1, s)
            # Double-smoothed absolute momentum
            ema1_abs = self.ema(m.abs(), r)
            ema2_abs = self.ema(ema1_abs, s)
            tsi = 100 * (ema2 / ema2_abs)
            return tsi
        df["tsi"] = tsi(df["close"], r=25, s=13)   # 56
        
        
        # Collection Stage 12, merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        
        # ============ Volatility Indicators ============
        # ============ BOLLINGER BANDS ============        
        new_cols['bb_middle'] = self.sma(df['close'], 20) # 57
        bb_std = self.std(df['close'], 20)
        new_cols['bb_upper'] = new_cols['bb_middle'] + (bb_std * 2) # 58
        new_cols['bb_lower'] = new_cols['bb_middle'] - (bb_std * 2) # 59
        new_cols['bb_width'] = (new_cols['bb_upper'] - new_cols['bb_lower']) / new_cols['bb_middle'] # 60
        new_cols['bb_pct'] = (df['close'] - new_cols['bb_lower']) / (new_cols['bb_upper'] - new_cols['bb_lower']) # 61   
        # Bollinger Bands Width
        new_cols['bb_bandwidth'] = (new_cols['bb_upper'] - new_cols['bb_lower']) / new_cols['bb_middle'] # 62
        
        # ============ ATR (Average True Range) ============        
        tr = self.true_range(df['high'], df['low'], df['close'])
        new_cols['atr'] = self.rma(tr, 14) # 63
        new_cols['atr_percent'] = (new_cols['atr'] / df['close']) * 100 # 64
        # Average Day Range
        new_cols['adr'] = (df['high'] - df['low']).rolling(window=14).mean() # 65
        
        # ============ KELTNER CHANNELS ============        
        new_cols['kc_middle'] = self.ema(df['close'], 20) # 66
        new_cols['kc_upper'] = new_cols['kc_middle'] + (2 * new_cols['atr']) # 67
        new_cols['kc_lower'] = new_cols['kc_middle'] - (2 * new_cols['atr']) # 68
        new_cols['kc_width'] = (new_cols['kc_upper'] - new_cols['kc_lower']) / new_cols['kc_middle'] # 69
        new_cols['kc_pct'] = (df['close'] - new_cols['kc_lower']) / (new_cols['kc_upper'] - new_cols['kc_lower']) # 70

        # ============ DONCHIAN CHANNELS ============        
        new_cols['dc_upper'] = df['high'].rolling(window=20).max() # 71
        new_cols['dc_lower'] = df['low'].rolling(window=20).min() # 72
        new_cols['dc_middle'] = (new_cols['dc_upper'] + new_cols['dc_lower']) / 2 # 73
        new_cols['dc_width'] = (new_cols['dc_upper'] - new_cols['dc_lower']) / new_cols['dc_middle'] # 74
        new_cols['dc_pct'] = (df['close'] - new_cols['dc_lower']) / (new_cols['dc_upper'] - new_cols['dc_lower']) # 75


        #============ Ulcer Index ========================================
        def ulcer_index(series: pd.Series, period: int = 14) -> pd.Series:
            """
            Ulcer Index (UI) - downside risk indicator.            
            Args:
                series (pd.Series): Price series (e.g., close prices).
                period (int): Lookback period (default=14).            
            Returns:
                pd.Series: Ulcer Index values.

            ✅ Notes
                Default period=14 matches common practice (like RSI).
                UI rises when prices fall below recent highs and stays elevated until recovery.
                Unlike volatility measures (like standard deviation), UI focuses only on downside risk.
            """
            # Rolling maximum (peak)
            rolling_max = series.rolling(window=period, min_periods=1).max()            
            # Percentage drawdown from peak
            drawdown = (series - rolling_max) / rolling_max * 100            
            # Squared drawdowns
            squared_dd = drawdown.pow(2)            
            # Ulcer Index = sqrt(mean of squared drawdowns)
            ui = np.sqrt(squared_dd.rolling(window=period, min_periods=1).mean())            
            return ui
        # --- Usage with your df ---
        df["ulcer_index"] = ulcer_index(df["close"], period=14)  # 76

        # ============ OBV (On Balance Volume) ============        
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])        
        new_cols['obv'] = obv # 77
           
        # ============ CHAIKIN MONEY FLOW ============
        # Define CLV (Close Location Value)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)

        # Calculate CMF (Chaikin Money Flow)
        mf_volume = clv * df['volume']
        new_cols['cmf'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        
        # ============ FORCE INDEX ============        
        new_cols['force_index'] = df['close'].diff() * df['volume'] # 79
        new_cols['force_index_ema'] = self.ema(new_cols['force_index'], 13) # 80
        
        # ============ EASE OF MOVEMENT ============        
        distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        box_ratio = (df['volume'] / 100000000) / (df['high'] - df['low'])
        new_cols['eom'] = distance / box_ratio # 81
        new_cols['eom_sma'] = self.sma(new_cols['eom'], 14) # 82

        #=============Volume Price Trend =============================
        def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
            """
            Volume Price Trend (VPT).
            
            Args:
                close (pd.Series): Closing prices.
                volume (pd.Series): Volume values.            
            Returns:
                pd.Series: VPT values.

            VPT is cumulative, so it grows over time. The absolute value isn’t as important as the trend (rising vs falling).
                Often used to confirm price trends:
                Price ↑ and VPT ↑ → strong bullish confirmation.
                Price ↑ but VPT ↓ → divergence (possible weakness).
            """
            # Percentage price change
            pct_change = close.pct_change()
            # VPT = cumulative sum of (pct_change * volume)
            vpt = (pct_change * volume).cumsum()
            return vpt
        new_cols["vpt"] = volume_price_trend(df["close"], df["volume"]) # 83
        
        # ============ negative_volume_index ================================
        def negative_volume_index(close: pd.Series, volume: pd.Series, start_value: float = 1000) -> pd.Series:
            """
            Negative Volume Index (NVI).            
            Args:
                close (pd.Series): Closing prices.
                volume (pd.Series): Volume values.
                start_value (float): Initial NVI value (default=1000).            
            Returns:
                pd.Series: NVI values.

            ✅ Notes
                Starts at 1000 by convention, but you can choose another base.
                Updates only on lower‑volume days.
                Often paired with a moving average (e.g., 255‑day EMA of NVI) to generate signals.
            """
            nvi = np.zeros(len(close))
            nvi[:] = np.nan
            nvi[0] = start_value
            for i in range(1, len(close)):
                if volume.iloc[i] < volume.iloc[i - 1]:
                    pct_change = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
                    nvi[i] = nvi[i - 1] + pct_change * nvi[i - 1]
                else:
                    nvi[i] = nvi[i - 1]
            return pd.Series(nvi, index=close.index)
        new_cols["nvi"] = negative_volume_index(df["close"], df["volume"], start_value=1000) # 84

        # ============ ACCUMULATION/DISTRIBUTION ============        
        clv_ad = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv_ad = clv_ad.fillna(0)
        new_cols['ad'] = (clv_ad * df['volume']).cumsum()  # 85
                
        # ============ MFI (Money Flow Index) ============        
        tp = self.typical_price(df['high'], df['low'], df['close'])
        raw_money_flow = tp * df['volume']        
        positive_flow = np.where(tp > tp.shift(1), raw_money_flow, 0)
        negative_flow = np.where(tp < tp.shift(1), raw_money_flow, 0)        
        positive_flow = pd.Series(positive_flow, index=df.index)
        negative_flow = pd.Series(negative_flow, index=df.index)        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()        
        mfi_ratio = positive_mf / negative_mf
        new_cols['mfi'] = 100 - (100 / (1 + mfi_ratio)) # 86

        # ====================VWAP ===========================

        def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
            """
            Volume Weighted Average Price (VWAP).
    
            Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Close prices
            volume (pd.Series): Volume
    
            Returns:
            pd.Series: VWAP values
            """
            typical_price = (high + low + close) / 3
            cum_vol_price = (typical_price * volume).cumsum()
            cum_volume = volume.cumsum()
            vwap = cum_vol_price / cum_volume
            return vwap
        new_cols["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])  # 87

        
        # Collection Stage 3, merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        
        # ============ Additional Custom Indicators ============

        # ============ CCI (Commodity Channel Index) ============
        
        tp = self.typical_price(df['high'], df['low'], df['close'])
        tp_sma = self.sma(tp, 20)
        mean_dev = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        new_cols['cci'] = (tp - tp_sma) / (0.015 * mean_dev) # 88
        
        # Woodies CCI
        new_cols['woodies_cci'] = new_cols['cci'] # 89
        new_cols['woodies_cci_signal'] = self.ema(new_cols['cci'], 6) # 90
        
        # ============ MOMENTUM ============        
        new_cols['momentum'] = df['close'] - df['close'].shift(10) # 91
        new_cols['momentum_pct'] = (df['close'] / df['close'].shift(10) - 1) * 100 # 92

        
        # ============ CHAIKIN MONEY FLOW ============        
        mf_multiplier = clv
        mf_volume = mf_multiplier * df['volume']
        new_cols['cmf'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum() # 93
        
        # ============ CHAIKIN OSCILLATOR ============        
        new_cols['chaikin_osc'] = self.ema(df['ad'], 3) - self.ema(df['ad'], 10) # 94
        
        # ============ VORTEX INDICATOR ============        
        vmp = np.abs(df['high'] - df['low'].shift(1))
        vmm = np.abs(df['low'] - df['high'].shift(1))        
        vmp = pd.Series(vmp, index=df.index)
        vmm = pd.Series(vmm, index=df.index)        
        vmp_sum = vmp.rolling(window=14).sum()
        vmm_sum = vmm.rolling(window=14).sum()
        tr_sum = tr.rolling(window=14).sum()        
        new_cols['vortex_pos'] = vmp_sum / tr_sum # 94
        new_cols['vortex_neg'] = vmm_sum / tr_sum # 95
        new_cols['vortex_diff'] = new_cols['vortex_pos'] - new_cols['vortex_neg'] # 96

        # ============ BALANCE OF POWER ============        
        new_cols['bop'] = (df['close'] - df['open']) / (df['high'] - df['low']) # 97
        new_cols['bop'] = new_cols['bop'].replace([np.inf, -np.inf], 0) # 98
        
        # ============ BULL BEAR POWER (Elder Ray) ============        
        new_cols['bull_power'] = df['high'] - self.ema(df['close'], 13) # 99
        new_cols['bear_power'] = df['low'] - self.ema(df['close'], 13) # 100
        
        # ============ CHOPPINESS INDEX ============        
        atr_sum = df['atr'].rolling(window=14).sum()
        high_low_range = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
        new_cols['choppiness'] = 100 * np.log10(atr_sum / high_low_range) / np.log10(14) # 101
        
        # Collection Stage 4, merge the indicators built so far, then reset collector
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
                        
        if df is None:
            raise ValueError("Indicator calculation returned None unexpectedly.")
    
        # =======================CandleSticks Patterns ======================
        # Helpers
        def _body(df): 
            return (df["close"] - df["open"]).abs()

        def _range(df): 
            return df["high"] - df["low"]

        def _upper(df): 
            return df["high"] - df[["open","close"]].max(axis=1)

        def _lower(df): 
            return df[["open","close"]].min(axis=1) - df["low"]

        def _bull(df): 
            return df["close"] > df["open"]

        def _bear(df): 
            return df["close"] < df["open"]

        def _small_body(df, ratio=0.25):
            return _body(df) <= ratio * _range(df)

        def _near_equal(s1, s2, tol_ratio=0.001):
            # tolerance relative to average range to avoid float issues
            return (s1 - s2).abs() <= tol_ratio * (_range(df).rolling(2).mean().fillna(_range(df)))

        # -----------------------------
        # Single-candle patterns
        # -----------------------------
        def pattern_hammer(df):
            return (_lower(df) > 2 * _body(df)) & (_upper(df) <= _body(df))

        def pattern_inverted_hammer(df):
            return (_upper(df) > 2 * _body(df)) & (_lower(df) <= _body(df)) & _bull(df)

        def pattern_hanging_man(df):
            return (_lower(df) > 2 * _body(df)) & (_upper(df) <= _body(df)) & _bear(df)

        def pattern_shooting_star(df):
            return (_upper(df) > 2 * _body(df)) & (_lower(df) <= _body(df))

        def pattern_doji(df, thr=0.1):
            return _body(df) <= thr * _range(df)

        def pattern_long_legged_doji(df):
            return (_small_body(df, 0.1)) & (_upper(df) >= 0.4 * _range(df)) & (_lower(df) >= 0.4 * _range(df))

        def pattern_dragonfly_doji(df):
            return (_small_body(df, 0.1)) & (_lower(df) >= 0.6 * _range(df)) & (_upper(df) <= 0.1 * _range(df))

        def pattern_gravestone_doji(df):
            return (_small_body(df, 0.1)) & (_upper(df) >= 0.6 * _range(df)) & (_lower(df) <= 0.1 * _range(df))

        def pattern_marubozu_bull(df):
            return _bull(df) & (_upper(df) <= 0.05 * _range(df)) & (_lower(df) <= 0.05 * _range(df))

        def pattern_marubozu_bear(df):
            return _bear(df) & (_upper(df) <= 0.05 * _range(df)) & (_lower(df) <= 0.05 * _range(df))

        def pattern_belt_hold_bull(df):
            return _bull(df) & (_open_equals_low(df:=df))

        def pattern_belt_hold_bear(df):
            return _bear(df) & (_open_equals_high(df:=df))

        def _open_equals_low(df, tol_ratio=0.001):
            return _near_equal(df["open"], df["low"], tol_ratio)

        def _open_equals_high(df, tol_ratio=0.001):
            return _near_equal(df["open"], df["high"], tol_ratio)

        # -----------------------------
        # Two-candle patterns
        # -----------------------------
        def pattern_bullish_engulfing(df):
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            return (c1 < o1) & _bull(df) & (df["close"] >= o1) & (df["open"] <= c1)

        def pattern_bearish_engulfing(df):
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            return (c1 > o1) & _bear(df) & (df["open"] >= c1) & (df["close"] <= o1)

        def pattern_tweezer_top(df, tol_ratio=0.001):
            h1, h2 = df["high"].shift(1), df["high"]
            return _bull(df.shift(1)) & _bear(df) & _near_equal(h1, h2, tol_ratio)

        def pattern_tweezer_bottom(df, tol_ratio=0.001):
            l1, l2 = df["low"].shift(1), df["low"]
            return _bear(df.shift(1)) & _bull(df) & _near_equal(l1, l2, tol_ratio)

        def pattern_bullish_harami(df):
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            body1_low, body1_high = pd.Series(np.minimum(o1, c1), index=df.index), pd.Series(np.maximum(o1, c1), index=df.index)
            body2_low, body2_high = df[["open","close"]].min(axis=1), df[["open","close"]].max(axis=1)
            return (c1 < o1) & _bull(df) & (body2_low >= body1_low) & (body2_high <= body1_high)

        def pattern_bearish_harami(df):
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            body1_low, body1_high = pd.Series(np.minimum(o1, c1), index=df.index), pd.Series(np.maximum(o1, c1), index=df.index)
            body2_low, body2_high = df[["open","close"]].min(axis=1), df[["open","close"]].max(axis=1)
            return (c1 > o1) & _bear(df) & (body2_low >= body1_low) & (body2_high <= body1_high)

        def pattern_piercing_line(df):
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            return (c1 < o1) & _bull(df) & (df["open"] < c1) & (df["close"] > (o1 + c1) / 2) & (df["close"] < o1)

        def pattern_dark_cloud_cover(df):
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            return (c1 > o1) & _bear(df) & (df["open"] > c1) & (df["close"] < (o1 + c1) / 2) & (df["close"] > o1)

        def pattern_on_neck(df):
            # Bearish candle then small up candle closing near prior low
            o1, c1, l1 = df["open"].shift(1), df["close"].shift(1), df["low"].shift(1)
            return (c1 < o1) & _bull(df) & _near_equal(df["close"], l1, 0.001)

        def pattern_in_neck(df):
            o1, c1, l1 = df["open"].shift(1), df["close"].shift(1), df["low"].shift(1)
            return (c1 < o1) & _bull(df) & (df["close"] > l1) & (df["close"] < o1)

        def pattern_thrusting(df):
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            mid1 = (o1 + c1) / 2.0
            return (c1 < o1) & _bull(df) & (df["open"] > c1) & (df["close"] < mid1)

        # -----------------------------
        # Three-candle patterns
        # -----------------------------
        def pattern_morning_star(df):
            o2, c2 = df["open"].shift(1), df["close"].shift(1)
            small2 = (df["high"].shift(1) - df["low"].shift(1)) * 0.25
            return _bear(df.shift(2)) & ((o2 - c2).abs() <= small2) & _bull(df) & (df["close"] >= (df["open"].shift(2) + df["close"].shift(2)) / 2)

        def pattern_evening_star(df):
            o2, c2 = df["open"].shift(1), df["close"].shift(1)
            small2 = (df["high"].shift(1) - df["low"].shift(1)) * 0.25
            return _bull(df.shift(2)) & ((o2 - c2).abs() <= small2) & _bear(df) & (df["close"] <= (df["open"].shift(2) + df["close"].shift(2)) / 2)

        def pattern_morning_doji_star(df):
            return pattern_morning_star(df) & pattern_doji(df.shift(1))

        def pattern_evening_doji_star(df):
            return pattern_evening_star(df) & pattern_doji(df.shift(1))

        def pattern_three_white_soldiers(df):
            return _bull(df) & _bull(df.shift(1)) & _bull(df.shift(2)) & (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))

        def pattern_three_black_crows(df):
            return _bear(df) & _bear(df.shift(1)) & _bear(df.shift(2)) & (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))

        def pattern_three_inside_up(df):
            return pattern_bearish_harami(df.shift(1)) & _bull(df) & (df["close"] >= df["open"].shift(2))

        def pattern_three_inside_down(df):
            return pattern_bullish_harami(df.shift(1)) & _bear(df) & (df["close"] <= df["open"].shift(2))

        def pattern_three_outside_up(df):
            return pattern_bullish_engulfing(df.shift(1)) & _bull(df) & (df["close"] >= df["close"].shift(1))

        def pattern_three_outside_down(df):
            return pattern_bearish_engulfing(df.shift(1)) & _bear(df) & (df["close"] <= df["close"].shift(1))

        # -----------------------------
        # Continuation / gaps
        # -----------------------------
        def pattern_rising_three_methods(df):
            cond_inside = (df["high"].shift(3) <= df["high"].shift(4)) & (df["low"].shift(3) >= df["low"].shift(4)) & \
                        (df["high"].shift(2) <= df["high"].shift(4)) & (df["low"].shift(2) >= df["low"].shift(4)) & \
                        (df["high"].shift(1) <= df["high"].shift(4)) & (df["low"].shift(1) >= df["low"].shift(4))
            return _bull(df.shift(4)) & _small_body(df.shift(3)) & _small_body(df.shift(2)) & _small_body(df.shift(1)) & cond_inside & _bull(df) & (df["close"] > df["high"].shift(4))

        def pattern_falling_three_methods(df):
            cond_inside = (df["high"].shift(3) <= df["high"].shift(4)) & (df["low"].shift(3) >= df["low"].shift(4)) & \
                        (df["high"].shift(2) <= df["high"].shift(4)) & (df["low"].shift(2) >= df["low"].shift(4)) & \
                        (df["high"].shift(1) <= df["high"].shift(4)) & (df["low"].shift(1) >= df["low"].shift(4))
            return _bear(df.shift(4)) & _small_body(df.shift(3)) & _small_body(df.shift(2)) & _small_body(df.shift(1)) & cond_inside & _bear(df) & (df["close"] < df["low"].shift(4))

        def pattern_upside_tasuki_gap(df):
            return _bull(df.shift(1)) & _bear(df) & _bull(df.shift(-1))

        def pattern_downside_tasuki_gap(df):
            return _bear(df.shift(1)) & _bull(df) & _bear(df.shift(-1))

        # -----------------------------
        # Other multi-candle patterns
        # -----------------------------
        def pattern_abandoned_baby_bull(df):
            return _bear(df.shift(2)) & pattern_doji(df.shift(1)) & _bull(df)

        def pattern_abandoned_baby_bear(df):
            return _bull(df.shift(2)) & pattern_doji(df.shift(1)) & _bear(df)

        def pattern_stick_sandwich(df):
            return _bear(df.shift(2)) & _bull(df.shift(1)) & _bear(df) & _near_equal(df["close"], df["close"].shift(2))

        def pattern_matching_low(df):
            return _bear(df.shift(1)) & _bear(df) & _near_equal(df["close"], df["close"].shift(1))

        def pattern_ladder_bottom(df):
            return _bear(df.shift(4)) & _bear(df.shift(3)) & _bear(df.shift(2)) & _bull(df.shift(1)) & _bull(df)

        def pattern_counterattack_bull(df):
            return _bear(df.shift(1)) & _bull(df) & _near_equal(df["close"], df["close"].shift(1))

        def pattern_counterattack_bear(df):
            return _bull(df.shift(1)) & _bear(df) & _near_equal(df["close"], df["close"].shift(1))

        def pattern_breakaway_bull(df):
            return _bear(df.shift(4)) & _bull(df)

        def pattern_breakaway_bear(df):
            return _bull(df.shift(4)) & _bear(df)

        def pattern_separating_lines_bull(df):
            # Bullish separating lines: down candle then up candle opening equal to previous open
            return _bear(df.shift(1)) & _bull(df) & _near_equal(df["open"], df["open"].shift(1))

        def pattern_separating_lines_bear(df):
            return _bull(df.shift(1)) & _bear(df) & _near_equal(df["open"], df["open"].shift(1))

        def pattern_side_by_side_white_lines(df):
            # Two consecutive long bullish candles with similar opens
            return _bull(df) & _bull(df.shift(1)) & _near_equal(df["open"], df["open"].shift(1)) & (_small_body(df, 0.2) == False) & (_small_body(df.shift(1), 0.2) == False)

        def pattern_homing_pigeon(df):
            # Two bullish candles with the second contained within the first (variation exists)
            o1, c1 = df["open"].shift(1), df["close"].shift(1)
            body1_low, body1_high = pd.Series(np.minimum(o1, c1), index=df.index), pd.Series(np.maximum(o1, c1), index=df.index)
            body2_low, body2_high = df[["open","close"]].min(axis=1), df[["open","close"]].max(axis=1)
            return _bull(df.shift(1)) & _bull(df) & (body2_low >= body1_low) & (body2_high <= body1_high)

        def pattern_matching_high(df):
            return _bull(df.shift(1)) & _bull(df) & _near_equal(df["close"], df["close"].shift(1))

        def pattern_kicking_bull(df):
            # Bullish: marubozu down followed by gap up marubozu up (approximation)
            return pattern_marubozu_bear(df.shift(1)) & pattern_marubozu_bull(df)

        def pattern_kicking_bear(df):
            return pattern_marubozu_bull(df.shift(1)) & pattern_marubozu_bear(df)

        def pattern_kicking_by_length_bull(df):
            return pattern_kicking_bull(df)  # simplified

        def pattern_kicking_by_length_bear(df):
            return pattern_kicking_bear(df)  # simplified

        def pattern_doji_star_bull(df):
            return pattern_doji(df.shift(1)) & _bull(df)

        def pattern_doji_star_bear(df):
            return pattern_doji(df.shift(1)) & _bear(df)

        def pattern_rickshaw_man(df):
            return pattern_long_legged_doji(df)

        # -----------------------------
        # Registry (~40 patterns)
        # -----------------------------
        CANDLE_PATTERN_FUNCS = {
            # Single
            "pattern_hammer": pattern_hammer,
            "pattern_inverted_hammer": pattern_inverted_hammer,
            "pattern_hanging_man": pattern_hanging_man,
            "pattern_shooting_star": pattern_shooting_star,
            "pattern_doji": pattern_doji,
            "pattern_long_legged_doji": pattern_long_legged_doji,
            "pattern_dragonfly_doji": pattern_dragonfly_doji,
            "pattern_gravestone_doji": pattern_gravestone_doji,
            "pattern_marubozu_bull": pattern_marubozu_bull,
            "pattern_marubozu_bear": pattern_marubozu_bear,
            "pattern_belt_hold_bull": pattern_belt_hold_bull,
            "pattern_belt_hold_bear": pattern_belt_hold_bear,

            # Two
            "pattern_bullish_engulfing": pattern_bullish_engulfing,
            "pattern_bearish_engulfing": pattern_bearish_engulfing,
            "pattern_tweezer_top": pattern_tweezer_top,
            "pattern_tweezer_bottom": pattern_tweezer_bottom,
            "pattern_bullish_harami": pattern_bullish_harami,
            "pattern_bearish_harami": pattern_bearish_harami,
            "pattern_piercing_line": pattern_piercing_line,
            "pattern_dark_cloud_cover": pattern_dark_cloud_cover,
            "pattern_on_neck": pattern_on_neck,
            "pattern_in_neck": pattern_in_neck,
            "pattern_thrusting": pattern_thrusting,

            # Three
            "pattern_morning_star": pattern_morning_star,
            "pattern_evening_star": pattern_evening_star,
            "pattern_morning_doji_star": pattern_morning_doji_star,
            "pattern_evening_doji_star": pattern_evening_doji_star,
            "pattern_three_white_soldiers": pattern_three_white_soldiers,
            "pattern_three_black_crows": pattern_three_black_crows,
            "pattern_three_inside_up": pattern_three_inside_up,
            "pattern_three_inside_down": pattern_three_inside_down,
            "pattern_three_outside_up": pattern_three_outside_up,
            "pattern_three_outside_down": pattern_three_outside_down,
            "pattern_rising_three_methods": pattern_rising_three_methods,
            "pattern_falling_three_methods": pattern_falling_three_methods,
            "pattern_upside_tasuki_gap": pattern_upside_tasuki_gap,
            "pattern_downside_tasuki_gap": pattern_downside_tasuki_gap,

            # Other multi
            "pattern_abandoned_baby_bull": pattern_abandoned_baby_bull,
            "pattern_abandoned_baby_bear": pattern_abandoned_baby_bear,
            "pattern_stick_sandwich": pattern_stick_sandwich,
            "pattern_matching_low": pattern_matching_low,
            "pattern_matching_high": pattern_matching_high,
            "pattern_ladder_bottom": pattern_ladder_bottom,
            "pattern_counterattack_bull": pattern_counterattack_bull,
            "pattern_counterattack_bear": pattern_counterattack_bear,
            "pattern_breakaway_bull": pattern_breakaway_bull,
            "pattern_breakaway_bear": pattern_breakaway_bear,
            "pattern_separating_lines_bull": pattern_separating_lines_bull,
            "pattern_separating_lines_bear": pattern_separating_lines_bear,
            "pattern_side_by_side_white_lines": pattern_side_by_side_white_lines,
            "pattern_homing_pigeon": pattern_homing_pigeon,
            "pattern_doji_star_bull": pattern_doji_star_bull,
            "pattern_doji_star_bear": pattern_doji_star_bear,
            "pattern_rickshaw_man": pattern_rickshaw_man,
        }

        # ============ Run Candlestick patterns ============
        for name, func in CANDLE_PATTERN_FUNCS.items():
            try:
                df[name] = func(df).fillna(False).astype(int)  # 1 if pattern present, else 0
            except Exception:
                df[name] = 0

        # ============ Run Chart patterns ============
        for name, func in CHART_PATTERN_FUNCS.items():
            try:
                df[name] = func(df).fillna(False).astype(int)  # 1 if pattern present else 0
            except Exception:
                df[name] = 0
        #--- PullBack Logic
        """
        Add these pullback calculations to calculator.py in the calculate_indicators method,
        after the chart patterns section and before the final return statement.
        """

        # ============ PULLBACK INDICATORS ============
        print("  Calculating pullback indicators...")

        # 1. Swing High/Low Detection
        def detect_swing_points(df, lookback=5):
            """
            Detect swing highs and swing lows.
            A swing high is a high that is higher than lookback bars before and after.
            A swing low is a low that is lower than lookback bars before and after.
            """
            swing_highs = pd.Series(False, index=df.index)
            swing_lows = pd.Series(False, index=df.index)
            
            for i in range(lookback, len(df) - lookback):
                # Check if current high is a swing high
                is_swing_high = True
                for j in range(1, lookback + 1):
                    if df['high'].iloc[i] <= df['high'].iloc[i-j] or df['high'].iloc[i] <= df['high'].iloc[i+j]:
                        is_swing_high = False
                        break
                swing_highs.iloc[i] = is_swing_high
                
                # Check if current low is a swing low
                is_swing_low = True
                for j in range(1, lookback + 1):
                    if df['low'].iloc[i] >= df['low'].iloc[i-j] or df['low'].iloc[i] >= df['low'].iloc[i+j]:
                        is_swing_low = False
                        break
                swing_lows.iloc[i] = is_swing_low
            
            return swing_highs, swing_lows

        new_cols['swing_high'] = detect_swing_points(df, 5)[0].astype(int)
        new_cols['swing_low'] = detect_swing_points(df, 5)[1].astype(int)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 2. Last Swing High/Low Values
        def get_last_swing_values(df, swing_highs, swing_lows):
            """Get the price level of the most recent swing high and low"""
            last_swing_high = pd.Series(np.nan, index=df.index)
            last_swing_low = pd.Series(np.nan, index=df.index)
            
            current_high = np.nan
            current_low = np.nan
            
            for i in range(len(df)):
                if swing_highs.iloc[i]:
                    current_high = df['high'].iloc[i]
                if swing_lows.iloc[i]:
                    current_low = df['low'].iloc[i]
                
                last_swing_high.iloc[i] = current_high
                last_swing_low.iloc[i] = current_low
            
            return last_swing_high, last_swing_low

        last_high, last_low = get_last_swing_values(df, df['swing_high'].astype(bool), df['swing_low'].astype(bool))
        new_cols['last_swing_high'] = last_high
        new_cols['last_swing_low'] = last_low
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 3. Pullback Percentage from Swing High/Low
        new_cols['pullback_from_high_pct'] = ((df['close'] - df['last_swing_high']) / df['last_swing_high'] * 100)
        new_cols['pullback_from_low_pct'] = ((df['close'] - df['last_swing_low']) / df['last_swing_low'] * 100)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 4. Fibonacci Retracement Levels
        def calculate_fib_levels(df, last_high, last_low):
            """Calculate Fibonacci retracement levels from last swing"""
            diff = last_high - last_low
            
            fib_236 = last_high - (diff * 0.236)
            fib_382 = last_high - (diff * 0.382)
            fib_500 = last_high - (diff * 0.500)
            fib_618 = last_high - (diff * 0.618)
            fib_786 = last_high - (diff * 0.786)
            
            return fib_236, fib_382, fib_500, fib_618, fib_786

        fib_236, fib_382, fib_500, fib_618, fib_786 = calculate_fib_levels(df, df['last_swing_high'], df['last_swing_low'])
        new_cols['fib_23.6'] = fib_236
        new_cols['fib_38.2'] = fib_382
        new_cols['fib_50.0'] = fib_500
        new_cols['fib_61.8'] = fib_618
        new_cols['fib_78.6'] = fib_786
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 5. Price Position Relative to Fib Levels
        new_cols['near_fib_236'] = (np.abs(df['close'] - fib_236) / df['close'] < 0.005).astype(int)  # Within 0.5%
        new_cols['near_fib_382'] = (np.abs(df['close'] - fib_382) / df['close'] < 0.005).astype(int)
        new_cols['near_fib_500'] = (np.abs(df['close'] - fib_500) / df['close'] < 0.005).astype(int)
        new_cols['near_fib_618'] = (np.abs(df['close'] - fib_618) / df['close'] < 0.005).astype(int)
        new_cols['near_fib_786'] = (np.abs(df['close'] - fib_786) / df['close'] < 0.005).astype(int)

        # 6. Trend Direction (for identifying pullbacks in trends)
        # Using multiple timeframes of moving averages
        new_cols['trend_short'] = np.where(df['close'] > df['ema_9'], 1, -1)  # Short-term trend
        new_cols['trend_medium'] = np.where(df['close'] > df['ema_50'], 1, -1)  # Medium-term trend
        new_cols['trend_long'] = np.where(df['close'] > df['ema_200'], 1, -1)  # Long-term trend
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 7. Pullback Depth Classification
        def classify_pullback_depth(pullback_pct):
            """Classify pullback as shallow, moderate, or deep"""
            result = pd.Series('none', index=pullback_pct.index)
            result[pullback_pct.between(-5, -1)] = 'shallow'
            result[pullback_pct.between(-10, -5)] = 'moderate'
            result[pullback_pct < -10] = 'deep'
            return result

        new_cols['pullback_depth'] = classify_pullback_depth(df['pullback_from_high_pct'])
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 8. Higher Lows in Uptrend (Bullish Pullback Pattern)
        def detect_higher_lows(df, swing_lows, window=3):
            """Detect if recent swing lows are making higher lows"""
            higher_lows = pd.Series(False, index=df.index)
            
            swing_low_indices = df.index[swing_lows].tolist()
            
            for i in range(len(swing_low_indices) - window + 1):
                recent_lows = [df.loc[swing_low_indices[i+j], 'low'] for j in range(window)]
                if all(recent_lows[j] < recent_lows[j+1] for j in range(len(recent_lows)-1)):
                    # Mark all candles after the pattern
                    start_idx = df.index.get_loc(swing_low_indices[i+window-1])
                    if start_idx < len(df):
                        higher_lows.iloc[start_idx:] = True
            
            return higher_lows

        new_cols['higher_lows_pattern'] = detect_higher_lows(df, df['swing_low'].astype(bool), window=3).astype(int)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 9. Lower Highs in Downtrend (Bearish Pullback Pattern)
        def detect_lower_highs(df, swing_highs, window=3):
            """Detect if recent swing highs are making lower highs"""
            lower_highs = pd.Series(False, index=df.index)
            
            swing_high_indices = df.index[swing_highs].tolist()
            
            for i in range(len(swing_high_indices) - window + 1):
                recent_highs = [df.loc[swing_high_indices[i+j], 'high'] for j in range(window)]
                if all(recent_highs[j] > recent_highs[j+1] for j in range(len(recent_highs)-1)):
                    start_idx = df.index.get_loc(swing_high_indices[i+window-1])
                    if start_idx < len(df):
                        lower_highs.iloc[start_idx:] = True
            
            return lower_highs

        new_cols['lower_highs_pattern'] = detect_lower_highs(df, df['swing_high'].astype(bool), window=3).astype(int)

        # 10. Pullback to Moving Average
        new_cols['pullback_to_ema9'] = ((df['close'] - df['ema_9']).abs() / df['ema_9'] < 0.01).astype(int)  # Within 1%
        new_cols['pullback_to_ema20'] = ((df['close'] - df['ema_26']).abs() / df['ema_26'] < 0.01).astype(int)
        new_cols['pullback_to_sma50'] = ((df['close'] - df['sma_50']).abs() / df['sma_50'] < 0.01).astype(int)

        # 11. Pullback Momentum (RSI during pullback)
        # Healthy pullback: RSI stays above 40 in uptrend, below 60 in downtrend
        new_cols['healthy_bull_pullback'] = ((df['trend_medium'] == 1) & 
                                            (df['rsi'] > 40) & 
                                            (df['rsi'] < 60) & 
                                            (df['close'] < df['close'].shift(1))).astype(int)

        new_cols['healthy_bear_pullback'] = ((df['trend_medium'] == -1) & 
                                            (df['rsi'] < 60) & 
                                            (df['rsi'] > 40) & 
                                            (df['close'] > df['close'].shift(1))).astype(int)

        # 12. Exhaustion Detection (Potential End of Pullback)
        # Volume decreases during pullback, RSI divergence
        new_cols['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}
        
        new_cols['volume_decreasing'] = (df['volume'] < df['volume_ma5']).astype(int)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 13. Pullback Duration (How many bars since last swing)
        def calculate_pullback_duration(df, swing_highs, swing_lows):
            """Calculate how many bars since the last swing high/low"""
            bars_since_high = pd.Series(0, index=df.index)
            bars_since_low = pd.Series(0, index=df.index)
            
            last_high_idx = -999
            last_low_idx = -999
            
            for i in range(len(df)):
                if swing_highs.iloc[i]:
                    last_high_idx = i
                if swing_lows.iloc[i]:
                    last_low_idx = i
                
                bars_since_high.iloc[i] = i - last_high_idx if last_high_idx >= 0 else 999
                bars_since_low.iloc[i] = i - last_low_idx if last_low_idx >= 0 else 999
            
            return bars_since_high, bars_since_low

        bars_high, bars_low = calculate_pullback_duration(df, df['swing_high'].astype(bool), df['swing_low'].astype(bool))
        new_cols['bars_since_swing_high'] = bars_high
        new_cols['bars_since_swing_low'] = bars_low
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 14. ABC Pullback Pattern Detection
        def detect_abc_pullback(df, window=20):
            """
            Detect ABC pullback pattern:
            A = Initial move
            B = Pullback
            C = Continuation
            """
            abc_bull = pd.Series(False, index=df.index)
            abc_bear = pd.Series(False, index=df.index)
            
            for i in range(window, len(df)):
                window_data = df.iloc[i-window:i+1]
                
                # Bullish ABC: Low -> High -> Higher Low
                highs = window_data['high']
                lows = window_data['low']
                
                point_a_idx = lows.idxmin()  # Lowest point
                point_a_pos = window_data.index.get_loc(point_a_idx)
                
                if point_a_pos < len(window_data) - 5:
                    after_a = window_data.iloc[point_a_pos:]
                    point_b_idx = after_a['high'].idxmax()  # Highest point after A
                    point_b_pos = after_a.index.get_loc(point_b_idx)
                    
                    if point_b_pos < len(after_a) - 2:
                        after_b = after_a.iloc[point_b_pos:]
                        point_c_low = after_b['low'].min()
                        point_a_low = window_data.loc[point_a_idx, 'low']
                        
                        # C should be higher than A (higher low)
                        if point_c_low > point_a_low * 1.005:  # 0.5% higher
                            abc_bull.iloc[i] = True
                
                # Bearish ABC: High -> Low -> Lower High
                point_a_idx = highs.idxmax()
                point_a_pos = window_data.index.get_loc(point_a_idx)
                
                if point_a_pos < len(window_data) - 5:
                    after_a = window_data.iloc[point_a_pos:]
                    point_b_idx = after_a['low'].idxmin()
                    point_b_pos = after_a.index.get_loc(point_b_idx)
                    
                    if point_b_pos < len(after_a) - 2:
                        after_b = after_a.iloc[point_b_pos:]
                        point_c_high = after_b['high'].max()
                        point_a_high = window_data.loc[point_a_idx, 'high']
                        
                        if point_c_high < point_a_high * 0.995:
                            abc_bear.iloc[i] = True
            
            return abc_bull, abc_bear

        abc_bull, abc_bear = detect_abc_pullback(df, window=20)
        new_cols['abc_pullback_bull'] = abc_bull.astype(int)
        new_cols['abc_pullback_bear'] = abc_bear.astype(int)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 15. Pullback Quality Score (0-100)
        def calculate_pullback_quality(df):
            """
            Calculate pullback quality score based on multiple factors:
            - Trend strength
            - Pullback depth (ideal: 38.2%-61.8%)
            - Volume decrease during pullback
            - RSI levels
            - Moving average support
            """
            quality_score = pd.Series(0, index=df.index)
            
            # Factor 1: Strong trend (30 points)
            trend_strength = (df['trend_short'] == df['trend_medium']) & (df['trend_medium'] == df['trend_long'])
            quality_score += trend_strength.astype(int) * 30
            
            # Factor 2: Ideal pullback depth (30 points)
            ideal_depth = df['pullback_from_high_pct'].between(-12, -3)
            quality_score += ideal_depth.astype(int) * 30
            
            # Factor 3: Volume decrease (20 points)
            quality_score += df['volume_decreasing'] * 20
            
            # Factor 4: RSI not oversold (20 points) - healthy pullback
            rsi_healthy = (df['rsi'] > 40) & (df['rsi'] < 60)
            quality_score += rsi_healthy.astype(int) * 20
            
            return quality_score

        new_cols['pullback_quality_score'] = calculate_pullback_quality(df)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 16. Support/Resistance Confluence at Pullback Level
        def detect_sr_confluence(df):
            """
            Detect if pullback is happening at a support/resistance level.
            Confluence = Multiple indicators at same level (MA, Fib, previous swing)
            """
            confluence = pd.Series(0, index=df.index)
            
            tolerance = 0.01  # 1% tolerance
            
            # Check confluence with each level
            near_ema9 = (df['close'] - df['ema_9']).abs() / df['close'] < tolerance
            near_ema26 = (df['close'] - df['ema_26']).abs() / df['close'] < tolerance
            near_sma50 = (df['close'] - df['sma_50']).abs() / df['close'] < tolerance
            near_swing_low = (df['close'] - df['last_swing_low']).abs() / df['close'] < tolerance
            
            confluence += near_ema9.astype(int)
            confluence += near_ema26.astype(int)
            confluence += near_sma50.astype(int)
            confluence += near_swing_low.astype(int)
            confluence += df['near_fib_382']
            confluence += df['near_fib_500']
            confluence += df['near_fib_618']
            
            return confluence

        new_cols['sr_confluence_score'] = detect_sr_confluence(df)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 17. Pullback Completion Signal
        # Combination of factors suggesting pullback is complete
        new_cols['pullback_complete_bull'] = (
            (df['trend_medium'] == 1) &  # In uptrend
            (df['pullback_quality_score'] >= 50) &  # Good quality pullback
            (df['rsi'] < 50) &  # RSI pulled back
            (df['sr_confluence_score'] >= 2) &  # At support level
            (df['volume_decreasing'] == 1) &  # Volume dried up
            (df['close'] > df['open'])  # Current candle bullish
        ).astype(int)

        new_cols['pullback_complete_bear'] = (
            (df['trend_medium'] == -1) &  # In downtrend
            (df['pullback_quality_score'] >= 50) &
            (df['rsi'] > 50) &  # RSI pulled back up
            (df['sr_confluence_score'] >= 2) &  # At resistance
            (df['volume_decreasing'] == 1) &
            (df['close'] < df['open'])  # Current candle bearish
        ).astype(int)

        # 18. Failed Pullback Detection (Pullback that becomes reversal)
        new_cols['failed_pullback_bull'] = (
            (df['trend_medium'] == 1) &  # Was in uptrend
            (df['close'] < df['last_swing_low']) &  # Broke below support
            (df['volume'] > df['volume_ma5'])  # With volume
        ).astype(int)

        new_cols['failed_pullback_bear'] = (
            (df['trend_medium'] == -1) &  # Was in downtrend  
            (df['close'] > df['last_swing_high']) &  # Broke above resistance
            (df['volume'] > df['volume_ma5'])
        ).astype(int)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 19. Measured Move Target (Expected continuation after pullback)
        def calculate_measured_move(df):
            """
            Calculate expected target after pullback completion.
            Target = Last swing high + (Last swing high - Last swing low)
            """
            move_size = df['last_swing_high'] - df['last_swing_low']
            bull_target = df['last_swing_high'] + move_size
            bear_target = df['last_swing_low'] - move_size
            
            return bull_target, bear_target

        bull_target, bear_target = calculate_measured_move(df)
        new_cols['measured_move_bull_target'] = bull_target
        new_cols['measured_move_bear_target'] = bear_target
        new_cols['distance_to_bull_target_pct'] = ((bull_target - df['close']) / df['close'] * 100)
        new_cols['distance_to_bear_target_pct'] = ((df['close'] - bear_target) / df['close'] * 100)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # 20. Pullback Stage Classification
        def classify_pullback_stage(df):
            """
            Classify current stage: 
            - 'impulse': Strong directional move
            - 'pullback_early': Just started pulling back
            - 'pullback_mid': In middle of pullback
            - 'pullback_late': Near completion
            - 'resumption': Trend resuming
            """
            stage = pd.Series('neutral', index=df.index)
            
            for i in range(10, len(df)):
                # Check recent price action
                recent_close = df['close'].iloc[i-10:i+1]
                
                # Impulse: Strong move in one direction
                if df['trend_medium'].iloc[i] == 1:
                    if recent_close.iloc[-1] > recent_close.iloc[0] * 1.03:  # 3% up
                        stage.iloc[i] = 'impulse_up'
                elif df['trend_medium'].iloc[i] == -1:
                    if recent_close.iloc[-1] < recent_close.iloc[0] * 0.97:  # 3% down
                        stage.iloc[i] = 'impulse_down'
                
                # Pullback stages based on bars since swing and depth
                bars_since = df['bars_since_swing_high'].iloc[i]
                pullback_depth = abs(df['pullback_from_high_pct'].iloc[i])
                
                if df['trend_medium'].iloc[i] == 1 and df['close'].iloc[i] < df['close'].iloc[i-1]:
                    if bars_since <= 3:
                        stage.iloc[i] = 'pullback_early'
                    elif bars_since <= 7 and pullback_depth < 8:
                        stage.iloc[i] = 'pullback_mid'
                    elif pullback_depth >= 5:
                        stage.iloc[i] = 'pullback_late'
                
                # Resumption: Pullback complete, trend resuming
                if df['pullback_complete_bull'].iloc[i] == 1:
                    stage.iloc[i] = 'resumption_bull'
                elif df['pullback_complete_bear'].iloc[i] == 1:
                    stage.iloc[i] = 'resumption_bear'
            
            return stage

        new_cols['pullback_stage'] = classify_pullback_stage(df)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        print("  ✓ Pullback indicators calculated successfully")

        # Merge all pullback indicators
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        new_cols = {}

        # --- HMM-READY SWING FIELDS ---
        print("  Building HMM swing features...")

        new_cols["swing_type"] = np.where(df["swing_high"] == 1, 1,
                            np.where(df["swing_low"] == 1, -1, 0))

        # Extract only swing points
        swings = df[df["swing_type"] != 0].copy().reset_index()

        # Initialize new observation columns
        swings["swing_index"] = np.arange(len(swings))
        swings["swing_price"] = np.where(swings["swing_type"] == 1,
                                        swings["high"], swings["low"])
        swings["swing_timestamp"] = swings["timestamp"]

        # Compute HMM observation features
        swings["swing_direction"] = np.sign(swings["swing_price"].diff())
        swings["swing_magnitude"] = swings["swing_price"].pct_change().abs() * 100
        swings["swing_duration"] = (swings["swing_timestamp"].diff().dt.total_seconds() / 60.0)
        swings["swing_volatility"] = df["atr"].reindex(swings["index"]).rolling(5, min_periods=1).mean().values
        swings["swing_mid_rsi"] = df["rsi"].reindex(swings["index"]).rolling(5, min_periods=1).mean().values

        # Merge back (optional)
        new_cols = new_cols.merge(swings[["index","swing_index","swing_type","swing_price",
                            "swing_direction","swing_magnitude",
                            "swing_duration","swing_volatility","swing_mid_rsi"]],
                    left_index=True, right_on="index", how="left").drop(columns=["index"])
        
        # Final df Generation- Merge any remaining indicators if any
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        # Return defragmented DataFrame
        df = df.copy()   # fully compact
        return df

    def get_dataframe(self, pair, timeframe):
        """Get a specific dataframe"""
        return self.data.get(pair, {}).get(timeframe)
    
    def save_to_csv(self, output_dir='lbank_data'):
        """Save all dataframes to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for pair in self.data:
            for tf in self.data[pair]:
                if self.data[pair][tf] is not None:
                    filename = f"{output_dir}/{pair}_{tf}.csv"
                    self.data[pair][tf].to_csv(filename, index=False)
                    print(f"Saved {filename}")
    
    def get_summary(self):
        """Print a summary of fetched data"""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        for pair in self.data:
            print(f"\n{pair.upper()}:")
            for tf in self.data[pair]:
                if self.data[pair][tf] is not None:
                    df = self.data[pair][tf]
                    latest = df.iloc[-1]
                    print(f"  {tf:5s}: {len(df):4d} candles | "
                          f"Latest Close: ${latest['close']:.4f} | "
                          f"RSI: {latest['rsi']:.2f}")


if __name__ == "__main__":
    # Initialize fetcher
    fetcher = LBankDataFetcher()
    
    # Fetch all data with pagination (recommended for full historical data)
    data = fetcher.fetch_all_data(use_pagination=True)
    
    # Print summary
    fetcher.get_summary()
    
    # Save to CSV files
    fetcher.save_to_csv()
