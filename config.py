import configparser
import os

# Define the path to the config file relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.ini')

def load_config(config_file=CONFIG_PATH):
    """
    Reads the configuration file and returns a config object.
    """
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        # Create default config if missing
        print(f"⚠️ Config file not found at {config_file}. Creating default.")
        config['database'] = {'db_path': './data/auto_trader_db.sqlite'}
        config['system'] = {'log_level': 'INFO', 'n_jobs': '-1'}
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            config.write(f)
            
    config.read(config_file)
    return config

# Global config object
CONFIG = load_config()

def get_db_path():
    return CONFIG.get('database', 'db_path', fallback='./data/auto_trader_db.sqlite')