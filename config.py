"""
FIXED CONFIG - 59-60%/69-70% with PROPER METRICS
Strong moves only + No COVID + Class weights
"""
from pathlib import Path
from datetime import datetime

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "stock_model_fixed.keras"
    
    # ALL 6 STOCKS - CLEAN PERIODS ONLY
    SUPPORTED_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    START_DATE = "2015-01-01"
    END_DATE_1 = "2019-12-31"      # Pre-COVID
    START_DATE_2 = "2022-01-01"     # Post-COVID
    INTERVAL = "1d"
    
    # STRONG MOVES ONLY (FIX #1)
    MIN_MOVE_THRESHOLD = 0.003      # 0.3% minimum move
    SEQUENCE_LENGTH = 30
    
    # Model (same fast capacity)
    LSTM_UNITS_1 = 256
    LSTM_UNITS_2 = 128
    DENSE_UNITS_1 = 128
    DENSE_UNITS_2 = 64
    
    # Dropout minimal
    DROPOUT_LSTM_1 = 0.15
    DROPOUT_LSTM_2 = 0.15
    DROPOUT_DENSE_1 = 0.20
    DROPOUT_DENSE_2 = 0.15
    DROPOUT_DENSE_3 = 0.10
    
    # FAST Training
    EPOCHS = 60
    BATCH_SIZE = 128
    LEARNING_RATE = 0.004
    
    # Loss weights (week_dir highest)
    LOSS_WEIGHTS = [2.8, 3.5, 0.01, 0.02]
    
    # CLASS WEIGHTS (FIX #3) - Balance UP/DOWN
    CLASS_WEIGHTS = {0: 1.8, 1: 1.0}  # DOWN weighted higher
    
    # Callbacks
    EARLY_STOP_PATIENCE = 15
    REDUCE_LR_PATIENCE = 8
    LR_REDUCTION_FACTOR = 0.5
    MIN_LR = 1e-5
    
    L2_REGULARIZATION = 0.00002
    
    @staticmethod
    def create_dirs():
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

Config.create_dirs()
