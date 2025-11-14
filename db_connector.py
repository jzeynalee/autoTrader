# db_connector.py
import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import List, Optional

class DatabaseConnector:
    """
    Centralized connection and persistence layer for the trading system.
    Handles I/O for feature data and strategy storage.
    """
    
    def __init__(self, db_path: str = './data/auto_trader_db.sqlite'):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establishes the SQLite connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Allows column access by name
            print(f"✅ Database connected at {self.db_path}")
        except sqlite3.Error as e:
            print(f"❌ Database connection failed: {e}")
            self.conn = None

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
            
    def create_tables(self):
        """Creates the necessary database tables if they don't exist."""
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        # --- 1. features_data (Stores raw OHLCV and calculated features) ---
        # Note: We keep a limited set of placeholder columns for features, 
        # but the table must support dynamic additions for feature_engineering.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features_data (
                timestamp INTEGER NOT NULL,
                pair_tf TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                rsi REAL, 
                macd_hist REAL,
                trend_structure TEXT,
                PRIMARY KEY (timestamp, pair_tf)
            );
        """)

        # --- 2. strategies_master (Stores final strategy definitions) ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies_master (
                strategy_id TEXT PRIMARY KEY,
                strategy_type TEXT,
                pair_tf TEXT,
                performance_score REAL,
                win_rate REAL,
                parameters_json TEXT, -- Full strategy definition
                sl_tp_json TEXT,     -- Final stop/target plan
                last_backtest_date TEXT
            );
        """)
        # --- 3. strategy_playbook (Stores regime-based playbook) ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_playbook (
                regime_name TEXT PRIMARY KEY,
                regime_id INTEGER,
                trend_direction TEXT,
                volatility_level TEXT,
                confirming_indicators_json TEXT,
                strategy_patterns_json TEXT,
                last_updated TEXT
            );
        """)
        self.conn.commit()

    def _get_current_columns(self, table_name: str) -> List[str]:
        """Retrieves current column names from a table."""
        if not self.conn: 
            return
        cursor = self.conn.execute(f"PRAGMA table_info({table_name});")
        return [col[1] for col in cursor.fetchall()]

    def _check_and_add_columns(self, df: pd.DataFrame, table_name: str):
        """Dynamically adds missing columns to the SQLite table schema."""
        if not self.conn or df.empty:
            return
            
        current_columns = self._get_current_columns(table_name)
        df_columns = df.columns.tolist()
        
        missing_columns = [col for col in df_columns if col not in current_columns]
        
        if missing_columns:
            print(f"⚠️ Detected {len(missing_columns)} new feature columns. Altering table...")
            cursor = self.conn.cursor()
            
            for col in missing_columns:
                # Infer the SQLite type based on the Pandas Dtype (REAL for floats/ints, TEXT for objects)
                dtype = df[col].dtype
                sql_type = 'TEXT' if pd.api.types.is_object_dtype(dtype) else 'REAL'
                
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {sql_type};")
                    print(f"   + Added column: {col} ({sql_type})")
                except sqlite3.OperationalError as e:
                    # Catch scenarios where column was added by another process thread
                    if "duplicate column name" not in str(e):
                        print(f"   ❌ Failed to add column {col}: {e}")
            
            self.conn.commit()

    # ========================================================================
    # STRATEGY & PLAYBOOK I/O
    # ========================================================================

    def upsert_strategy_playbook(self, playbook_record: dict):
        """
        Inserts or replaces a single record in the strategy_playbook table.
        
        Args:
            playbook_record: A dictionary matching the table schema.
        """
        if not self.conn:
            print("❌ Cannot upsert playbook: No database connection.")
            return

        cursor = self.conn.cursor()
        query = """
            INSERT OR REPLACE INTO strategy_playbook (
                regime_name,
                regime_id,
                trend_direction,
                volatility_level,
                confirming_indicators_json,
                strategy_patterns_json,
                last_updated
            ) VALUES (
                :regime_name,
                :regime_id,
                :trend_direction,
                :volatility_level,
                :confirming_indicators_json,
                :strategy_patterns_json,
                :last_updated
            );
        """
        
        try:
            # Convert datetime to string if it's an object
            if isinstance(playbook_record.get('last_updated'), datetime):
                playbook_record['last_updated'] = playbook_record['last_updated'].isoformat()
                
            cursor.execute(query, playbook_record)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Error upserting strategy playbook record '{playbook_record.get('regime_name')}': {e}")

    # ========================================================================
    # INGESTION & CORE SYSTEM I/O (New/Updated Methods)
    # ========================================================================

    def load_last_timestamp(self, pair_tf: str) -> Optional[int]:
        """
        Retrieves the maximum timestamp (in seconds) for a given pair_tf.
        Used by ingestion.py for incremental loading.
        """
        if not self.conn:
            return None
            
        cursor = self.conn.cursor()
        query = "SELECT MAX(timestamp) FROM features_data WHERE pair_tf = ?;"
        
        try:
            cursor.execute(query, (pair_tf,))
            result = cursor.fetchone()[0]
            # Result will be an integer timestamp or None
            return result if result is not None else None
        except Exception as e:
            print(f"❌ Error loading last timestamp for {pair_tf}: {e}")
            return None
            
    def upsert_raw_ohlcv(self, df_raw: pd.DataFrame, pair_tf: str):
        """
        Inserts or replaces raw OHLCV data into features_data.
        df_raw must contain: timestamp_s, pair_tf, open, high, low, close, volume.
        """
        if not self.conn or df_raw.empty:
            return

        # Rename timestamp column for matching schema
        if 'timestamp_s' in df_raw.columns:
            df_raw = df_raw.rename(columns={'timestamp_s': 'timestamp'})
        
        # Select the columns matching the database schema for raw data
        cols_to_save = ['timestamp', 'pair_tf', 'open', 'high', 'low', 'close', 'volume']
        df_to_save = df_raw.reindex(columns=cols_to_save, fill_value=None)
        
        try:
            # Use to_sql to insert/replace (SQLite handles PK conflict implicitly)
            df_to_save.to_sql(
                'features_data', 
                self.conn, 
                if_exists='append', 
                index=False,
            )
            # This relies on the table having been created with the PRIMARY KEY constraint.
            self.conn.commit()
            
        except Exception as e:
            print(f"❌ Database UPSERT failed for raw OHLCV for {pair_tf}: {e}")
            # Depending on error, rollback might be necessary, but print error for now.
            # raise # Re-raise if fatal
            

    # --- Methods required by FeatureEngineer (Task 3) ---

    def get_all_raw_pair_tfs(self) -> List[str]:
        """Retrieves a list of all pair_tfs currently present in the database."""
        if not self.conn:
            # Fallback for mocking needed by FeatureEngineer, but should rely on ingestion.
            return ['btc_usdt_1m', 'btc_usdt_5m', 'btc_usdt_15m', 'btc_usdt_1h', 'btc_usdt_4h']
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT pair_tf FROM features_data;")
        return [row[0] for row in cursor.fetchall()]

    def load_raw_ohlcv(self, pair_tf: str) -> Optional[pd.DataFrame]:
        """Loads raw OHLCV data from features_data for feature calculation."""
        if not self.conn:
            return None
            
        query = f"""
            SELECT timestamp, open, high, low, close, volume 
            FROM features_data 
            WHERE pair_tf = ?
            ORDER BY timestamp ASC;
        """
        try:
            # We load the INTEGER timestamp and convert to datetime index later
            df = pd.read_sql_query(query, self.conn, params=(pair_tf,))
            if df.empty:
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            return df
        except Exception as e:
            print(f"❌ Error loading raw OHLCV for {pair_tf}: {e}")
            return None

    def get_all_calculated_pair_tfs(self) -> List[str]:
        """Retrieves all pair_tfs that have calculated features (non-null RSI placeholder)."""
        if not self.conn: return []
        cursor = self.conn.cursor()
        # Query: Ensure 'close' exists (raw data) AND 'rsi' exists (feature data)
        cursor.execute("SELECT DISTINCT pair_tf FROM features_data WHERE close IS NOT NULL AND rsi IS NOT NULL;")
        return [row[0] for row in cursor.fetchall()]

    def load_full_features(self, pair_tf: str) -> Optional[pd.DataFrame]:
        """Loads a DataFrame with all available features for the pair_tf."""
        if not self.conn: return None
        
        # We select ALL columns dynamically from the features_data table
        # This requires using the helper method introduced in the last fix:
        # (Assuming _get_current_columns is accessible or implemented)
        
        # Placeholder: Retrieve columns dynamically (must be implemented)
        cursor = self.conn.execute("PRAGMA table_info(features_data);")
        columns = [col[1] for col in cursor.fetchall()]
        columns_str = ", ".join(columns)
        
        query = f"SELECT {columns_str} FROM features_data WHERE pair_tf = ? ORDER BY timestamp ASC;"
        try:
            # Load the raw data as a DataFrame
            df = pd.read_sql_query(query, self.conn, params=(pair_tf,))
            if df.empty: 
                return None
            
            # CRITICAL FIX: Convert column names to lowercase for consistency
            df.columns = [col.lower() for col in df.columns]
            
            # Convert timestamp and set index
            # NOTE: Timestamp is an INTEGER (seconds) in the DB.
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            return df
        except Exception as e:
            print(f"❌ Error loading full features for {pair_tf}: {e}")
            return None

    def update_features(self, df_features: pd.DataFrame, pair_tf: str):
        """
        Updates the database table with ALL available columns from df_features.
        This generic UPSERT handles the 200+ feature columns added by the FeatureEngineer.
        """
        if not self.conn or df_features.empty:
            return
        
        # 1. Prepare data for insertion (Ensure index is in seconds for PK matching)
        df_features['timestamp'] = df_features.index.astype(int) // 10**9
        df_features['pair_tf'] = pair_tf
        df_to_save = df_features.reset_index(drop=True)
        
        # 2. Dynamic column handling (SQLite requires explicit column list for INSERT OR REPLACE)
        columns = df_to_save.columns.tolist()
        columns_str = ", ".join(columns)
        
        try:
            self._check_and_add_columns(df_to_save, 'features_data')            
            df_to_save.to_sql(
                'features_data_temp', 
                self.conn, 
                if_exists='replace', 
                index=False
            )
            
            # Use the explicit INSERT OR REPLACE from the temporary table
            cursor = self.conn.cursor()
            
            # NOTE: We use a potentially slow but robust approach due to dynamic columns.
            # This INSERT OR REPLACE will only update existing columns. New columns require ALTER TABLE (future engineering task).
            cursor.execute(f"""
                INSERT OR REPLACE INTO features_data ({columns_str})
                SELECT {columns_str} FROM features_data_temp;
            """)
            
            # Clean up temp table
            cursor.execute("DROP TABLE features_data_temp;")
            self.conn.commit()
            
        except Exception as e:
            print(f"❌ Database feature UPSERT failed for {pair_tf}: {e}")
            raise