"""
Feature Engineering - Create technical indicators and targets
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import Config

def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators from OHLCV data
    
    Args:
        df: DataFrame with OHLCV columns
    
    Returns:
        DataFrame with technical indicators added
    """
    
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_mid'] = sma_20
    
    # ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    # Price changes
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for prediction
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame with target columns added
    """
    
    df = df.copy()
    
    # Tomorrow's direction (1 if up, 0 if down)
    df['tomorrow_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Tomorrow's return
    df['tomorrow_return'] = df['close'].shift(-1) / df['close'] - 1
    
    # 1-week (5 trading days) direction
    df['week_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # 1-week return
    df['week_return'] = df['close'].shift(-5) / df['close'] - 1
    
    # Drop rows with NaN targets
    df = df.dropna()
    
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Build normalized feature matrix for model input
    
    Args:
        df: DataFrame with all features and targets
    
    Returns:
        Tuple of (X, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, scaler)
    """
    
    # Feature columns (exclude targets)
    feature_cols = [col for col in df.columns if col not in [
        'tomorrow_direction', 'tomorrow_return', 'week_direction', 'week_return',
        'tr1', 'tr2', 'tr3', 'tr'  # Drop intermediate ATR columns
    ]]
    
    # Extract targets
    y_tom_dir = df['tomorrow_direction'].values.astype(float)
    y_week_dir = df['week_direction'].values.astype(float)
    y_tom_ret = df['tomorrow_return'].values.astype(float)
    y_week_ret = df['week_return'].values.astype(float)
    
    # Normalize features
    X = df[feature_cols].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    return X, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, scaler


def make_sequences(X, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, seq_len=60):
    """
    Create sequences for LSTM
    
    Args:
        X: Feature matrix
        y_tom_dir: Tomorrow direction targets
        y_week_dir: Week direction targets
        y_tom_ret: Tomorrow return targets
        y_week_ret: Week return targets
        seq_len: Sequence length (default: 60 days)
    
    Returns:
        Tuple of (X_seq, y_tom_dir_seq, y_week_dir_seq, y_tom_ret_seq, y_week_ret_seq)
    """
    
    X_seq = []
    y_tom_dir_seq = []
    y_week_dir_seq = []
    y_tom_ret_seq = []
    y_week_ret_seq = []
    
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_tom_dir_seq.append(y_tom_dir[i + seq_len])
        y_week_dir_seq.append(y_week_dir[i + seq_len])
        y_tom_ret_seq.append(y_tom_ret[i + seq_len])
        y_week_ret_seq.append(y_week_ret[i + seq_len])
    
    X_seq = np.array(X_seq)
    y_tom_dir_seq = np.array(y_tom_dir_seq).reshape(-1, 1)
    y_week_dir_seq = np.array(y_week_dir_seq).reshape(-1, 1)
    y_tom_ret_seq = np.array(y_tom_ret_seq).reshape(-1, 1)
    y_week_ret_seq = np.array(y_week_ret_seq).reshape(-1, 1)
    
    return X_seq, y_tom_dir_seq, y_week_dir_seq, y_tom_ret_seq, y_week_ret_seq