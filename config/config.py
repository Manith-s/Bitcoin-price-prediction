"""Configuration parameters for the Bitcoin price prediction project"""

import os
from datetime import datetime, timedelta

# Data collection parameters
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CRYPTO_SYMBOL = "BTC"
QUOTE_CURRENCY = "USD"
TIME_INTERVAL = "1d"  # 1d, 1h, 15m, etc.

# Feature engineering parameters
WINDOW_SIZES = [7, 14, 30, 60, 90]
TECHNICAL_INDICATORS = [
    "RSI", "MACD", "Bollinger", "ATR", "OBV",
    "Ichimoku", "ROC", "Williams_R", "MFI", "CCI"
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
ENSEMBLE_MODELS = [
    "RandomForest", "GradientBoosting", "XGBoost", 
    "LightGBM", "CatBoost"
]
RL_ALGORITHM = "PPO"  # PPO, A2C, DQN, etc.

# Training parameters
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Define project paths
def setup_paths():
    """Setup project paths and create directories if they don't exist."""
    # Get project base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define directories
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Define file paths
    RAW_DATA_PATH = os.path.join(DATA_DIR, f"{CRYPTO_SYMBOL}_{QUOTE_CURRENCY}_{TIME_INTERVAL}_raw.csv")
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, f"{CRYPTO_SYMBOL}_{QUOTE_CURRENCY}_{TIME_INTERVAL}_processed.csv")
    FEATURED_DATA_PATH = os.path.join(DATA_DIR, f"{CRYPTO_SYMBOL}_{QUOTE_CURRENCY}_{TIME_INTERVAL}_featured.csv")
    
    paths = {
        'BASE_DIR': BASE_DIR,
        'DATA_DIR': DATA_DIR,
        'MODEL_DIR': MODEL_DIR,
        'RESULTS_DIR': RESULTS_DIR,
        'LOG_DIR': LOG_DIR,
        'RAW_DATA_PATH': RAW_DATA_PATH,
        'PROCESSED_DATA_PATH': PROCESSED_DATA_PATH,
        'FEATURED_DATA_PATH': FEATURED_DATA_PATH
    }
    
    return paths

# Get paths
PATHS = setup_paths()

# Extract paths into variables
BASE_DIR = PATHS['BASE_DIR']
DATA_DIR = PATHS['DATA_DIR']
MODEL_DIR = PATHS['MODEL_DIR']
RESULTS_DIR = PATHS['RESULTS_DIR']
LOG_DIR = PATHS['LOG_DIR']
RAW_DATA_PATH = PATHS['RAW_DATA_PATH']
PROCESSED_DATA_PATH = PATHS['PROCESSED_DATA_PATH']
FEATURED_DATA_PATH = PATHS['FEATURED_DATA_PATH']