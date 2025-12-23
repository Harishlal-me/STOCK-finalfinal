#!/usr/bin/env python3
"""
Enhanced Stock Predictor - With Advanced Risk Management
✅ Uses LATEST available price from CSV (automatically gets most recent date)
✅ Dynamic probability thresholds based on market conditions
✅ CSV logging for analysis
✅ Improved SPY/index download with fallbacks
✅ Simplified output without position sizing
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_SUPPRESS_LOGS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)

import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
from typing import List
import argparse
import csv

tf.get_logger().setLevel('ERROR')

# ============================================================================
# DYNAMIC PROBABILITY THRESHOLDS
# ============================================================================
class ProbabilityThresholds:
    """Dynamic risk-based probability thresholds"""
    
    @staticmethod
    def get_dynamic_threshold(volatility: float, market_regime: str) -> dict:
        """
        Dynamic thresholds based on market conditions
        - High volatility = require higher probability
        - Bear market = require higher probability
        - Bull market = can accept slightly lower probability
        """
        base_threshold = 0.60
        
        # Volatility adjustment
        if volatility > 0.04:  # Very high volatility
            vol_adj = 0.08
        elif volatility > 0.03:  # High volatility
            vol_adj = 0.05
        elif volatility > 0.02:  # Moderate volatility
            vol_adj = 0.03
        else:  # Low volatility
            vol_adj = 0.00
        
        # Market regime adjustment
        if "BULL STRONG" in market_regime:
            regime_adj = -0.03  # Can be more aggressive
        elif "BULL" in market_regime:
            regime_adj = -0.02
        elif "BEAR" in market_regime:
            regime_adj = 0.05  # More conservative
        elif "SIDEWAYS" in market_regime:
            regime_adj = 0.03
        else:
            regime_adj = 0.02
        
        adjusted_threshold = base_threshold + vol_adj + regime_adj
        adjusted_threshold = max(0.55, min(0.75, adjusted_threshold))  # Clamp between 55-75%
        
        return {
            'threshold': adjusted_threshold,
            'vol_adjustment': vol_adj,
            'regime_adjustment': regime_adj
        }
    
    @staticmethod
    def get_confidence_label(prob: float, threshold: float) -> str:
        """Map probability to confidence with dynamic threshold"""
        margin = prob - threshold
        
        if margin >= 0.10:
            return "🟢 VERY HIGH"
        elif margin >= 0.05:
            return "🟢 HIGH"
        elif margin >= 0.00:
            return "🟡 MEDIUM"
        elif margin >= -0.05:
            return "🟠 LOW"
        else:
            return "🔴 VERY LOW"

# ============================================================================
# IMPROVED MARKET DATA FETCHING
# ============================================================================
class MarketDataFetcher:
    """Fetch market data with fallbacks"""
    
    @staticmethod
    def fetch_market_trend(df: pd.DataFrame, retries: int = 3) -> pd.DataFrame:
        """Fetch SPY data with multiple fallbacks"""
        
        # Suppress yfinance warnings
        import logging
        yf_logger = logging.getLogger('yfinance')
        yf_logger.setLevel(logging.CRITICAL)
        
        # Try SPY first
        for attempt in range(retries):
            try:
                import yfinance as yf
                
                # Get last 2 years of data to ensure enough for 200 EMA
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    market_df = yf.download(
                        'SPY', 
                        start=start_date,
                        end=end_date,
                        progress=False,
                        timeout=10,
                        show_errors=False
                    )
                
                if not market_df.empty and len(market_df) >= 200:
                    market_df['ema_200'] = market_df['Close'].ewm(span=200, adjust=False).mean()
                    market_df['market_trend'] = (market_df['Close'] > market_df['ema_200']).astype(int)
                    market_df = market_df[['market_trend']]
                    
                    if not isinstance(market_df.index, pd.DatetimeIndex):
                        market_df.index = pd.to_datetime(market_df.index)
                    
                    df = df.join(market_df, how='left')
                    df['market_trend'] = df['market_trend'].fillna(method='ffill').fillna(1).astype(int)
                    
                    return df
                    
            except Exception as e:
                if attempt < retries - 1:
                    continue
        
        # Fallback 1: Try QQQ
        try:
            import yfinance as yf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                market_df = yf.download('QQQ', period="2y", progress=False, timeout=10, show_errors=False)
            
            if not market_df.empty and len(market_df) >= 200:
                market_df['ema_200'] = market_df['Close'].ewm(span=200, adjust=False).mean()
                market_df['market_trend'] = (market_df['Close'] > market_df['ema_200']).astype(int)
                market_df = market_df[['market_trend']]
                
                if not isinstance(market_df.index, pd.DatetimeIndex):
                    market_df.index = pd.to_datetime(market_df.index)
                
                df = df.join(market_df, how='left')
                df['market_trend'] = df['market_trend'].fillna(method='ffill').fillna(1).astype(int)
                
                return df
        except:
            pass
        
        # Fallback 2: Use stock's own trend (silent fallback)
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['market_trend'] = (df['close'] > df['ema_200']).astype(int)
        df = df.drop('ema_200', axis=1)
        
        return df

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    up_move = high.diff()
    down_move = -low.diff()
    
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
    
    di_diff = abs(pos_di - neg_di)
    di_sum = pos_di + neg_di
    
    adx = 100 * di_diff.rolling(window=period).mean() / di_sum
    return adx.fillna(0)

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================
class MarketRegime:
    """Detect bull/bear/sideways market"""
    
    @staticmethod
    def get_regime(df: pd.DataFrame) -> str:
        """Determine market regime"""
        if len(df) < 50:
            return "UNKNOWN"
        
        close = df['close'].iloc[-50:]
        
        ma_50 = close.mean()
        ma_200 = df['close'].mean() if len(df) >= 200 else close.mean()
        
        current = close.iloc[-1]
        
        volatility = df['close'].pct_change().std() * 100
        
        if current > ma_50 > ma_200 and volatility < 2.5:
            return "📈 BULL STRONG"
        elif current > ma_50 and volatility < 3:
            return "📈 BULL"
        elif current < ma_50 < ma_200 and volatility < 2.5:
            return "📉 BEAR STRONG"
        elif current < ma_50 and volatility < 3:
            return "📉 BEAR"
        elif abs(current - ma_50) < (ma_50 * 0.02) and volatility < 1.5:
            return "⚖️ SIDEWAYS"
        else:
            return "🔄 MIXED"

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
class RiskManagement:
    """Calculate realistic targets and stop-losses"""
    
    @staticmethod
    def calculate_targets(current_price: float, 
                         atr: float, 
                         volatility: float,
                         direction_prob: float) -> dict:
        """Use ATR for realistic targets"""
        
        if direction_prob > 0.5:  # UP signal
            target_high = current_price + (atr * 1.5)
            target_low = current_price + (atr * 0.5)
            stop_loss = current_price - (atr * 1.0)
        else:  # DOWN signal
            target_low = current_price - (atr * 1.5)
            target_high = current_price - (atr * 0.5)
            stop_loss = current_price + (atr * 1.0)
        
        upside_risk = abs(target_high - current_price)
        downside_risk = abs(current_price - stop_loss)
        risk_reward = upside_risk / downside_risk if downside_risk > 0 else 0
        
        target_return = ((target_high + target_low) / 2 - current_price) / current_price * 100
        max_loss = abs(current_price - stop_loss) / current_price * 100
        
        return {
            'target_high': target_high,
            'target_low': target_low,
            'stop_loss': stop_loss,
            'risk_reward': risk_reward,
            'expected_return': target_return,
            'max_loss': max_loss,
            'atr_pct': (atr / current_price) * 100
        }

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = df['atr'] / df['close']
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['trend_strength'] = abs(df['close'] / df['ma_50'] - 1)
    df['roc_10'] = df['close'].pct_change(10)
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    df['sma_7'] = df['close'].rolling(7).mean()
    df['sma_7'] = (df['close'] - df['sma_7']) / df['sma_7']
    
    ema_7 = df['close'].ewm(span=7, adjust=False).mean()
    df['ema_7'] = (df['close'] - ema_7) / ema_7
    
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['rsi_14'] = df['rsi_14'] / 100
    
    df['volume_ma_7'] = df['volume'].rolling(7).mean()
    df['volume_ma_30'] = df['volume'].rolling(30).mean()
    df['volume_trend_week'] = df['volume_ma_7'] / df['volume_ma_30']
    
    df['weekly_return'] = df['close'].pct_change(5)
    df['weekly_volatility'] = df['close'].pct_change().rolling(5).std()
    
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_diff'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
    
    df['adx_14'] = calculate_adx(df, period=14)
    df['adx_14'] = df['adx_14'] / 100
    
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    df = MarketDataFetcher.fetch_market_trend(df)
    
    return df

# ============================================================================
# PREDICTION DATA CLASS
# ============================================================================
@dataclass
class StockPrediction:
    symbol: str
    current_price: float
    price_date: str
    
    week_prob_up: float
    week_direction: str
    confidence: str
    
    target_high: float
    target_low: float
    stop_loss: float
    risk_reward: float
    expected_return: float
    max_loss: float
    
    market_regime: str
    atr_pct: float
    volatility: float
    
    dynamic_threshold: float
    
    action: str
    signal_strength: str
    reasoning: List[str]
    warnings: List[str]

# ============================================================================
# PREDICTION ENGINE
# ============================================================================
def load_and_prepare_data(symbol: str):
    """Load data from local CSV and automatically use the LATEST date available"""
    
    # Try multiple possible CSV locations
    csv_paths = [
        Path(f"data/stock_data/{symbol}.csv"),
        Path(f"data/{symbol}.csv"),
        Path(f"stock_data/{symbol}.csv"),
        Path(f"{symbol}.csv"),
        Path(f"data/stock_data/{symbol}_daily.csv"),
    ]
    
    df = None
    csv_found = None
    
    # Try to load from CSV first
    for csv_path in csv_paths:
        if csv_path.exists():
            try:
                # Read CSV without setting index initially
                df = pd.read_csv(csv_path)
                csv_found = csv_path
                print(f" [Loading from: {csv_path}]", end="")
                break
            except Exception as e:
                continue
    
    # If CSV not found, fall back to fetch_stock_data
    if df is None:
        sys.path.append(str(Path(__file__).parent))
        try:
            from src.data_loader import fetch_stock_data
            df = fetch_stock_data(symbol, use_cache=True)
        except:
            raise ValueError(f"Could not load data for {symbol}. CSV not found in: {[str(p) for p in csv_paths]}")
    
    if df.empty:
        raise ValueError(f"Could not fetch data for {symbol}")
    
    # Normalize column names to lowercase FIRST
    df.columns = df.columns.str.lower()
    
    # Handle date column - try different possible names
    date_columns = ['date', 'datetime', 'timestamp', 'time']
    date_col = None
    
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    # If no date column found, check if first column looks like dates
    if date_col is None and len(df.columns) > 0:
        first_col = df.columns[0]
        # Check if first column contains date-like strings
        if df[first_col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any():
            date_col = first_col
    
    if date_col:
        # Parse dates and set as index
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])  # Remove rows with invalid dates
        df.set_index(date_col, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert index to datetime
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]  # Remove rows with invalid dates
        except:
            raise ValueError(f"Could not parse dates from CSV for {symbol}")
    
    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Sort by date to ensure latest is at the end
    df = df.sort_index()
    
    # Remove any duplicate dates (keep last)
    df = df[~df.index.duplicated(keep='last')]
    
    # Debug: Print the actual latest date in CSV
    if len(df) > 0:
        print(f" [{len(df)} rows, latest: {df.index[-1].strftime('%Y-%m-%d')}]", end="")
    else:
        raise ValueError(f"No valid data found in CSV for {symbol}")
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}. Found: {list(df.columns)}")
    
    # Convert numeric columns to float (they might be strings in CSV)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN values in required columns
    df = df.dropna(subset=required_cols)
    
    if len(df) == 0:
        raise ValueError(f"No valid numeric data found in CSV for {symbol}")
    
    df = create_prediction_features(df)
    
    feature_cols = [
        'atr_pct', 'volatility', 'trend_strength', 'roc_10', 'volume_ratio',
        'sma_7', 'ema_7', 'rsi_14', 'volume_trend_week',
        'weekly_return', 'weekly_volatility',
        'ema_diff', 'adx_14', 'price_vwap', 'market_trend'
    ]
    
    return df, feature_cols

def predict_stock(symbol: str) -> StockPrediction:
    """Make prediction with full risk management - uses LATEST price from CSV"""
    symbol = symbol.upper()
    
    model_paths = [
        Path("models/stock_model_fixed.keras"),
        Path("./models/stock_model_fixed.keras"),
        Path(__file__).parent / "models" / "stock_model_fixed.keras",
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("Model not found. Run: python train_fixed.py")
    
    model = tf.keras.models.load_model(str(model_path))
    
    # Load data - will automatically get the latest available date
    df, feature_cols = load_and_prepare_data(symbol)
    
    # Get LATEST price and date from CSV (automatically uses the most recent row)
    current_price = float(df['close'].iloc[-1])
    price_date = df.index[-1].strftime('%Y-%m-%d')
    current_atr = float(df['atr'].iloc[-1]) or 1.0
    current_volatility = float(df['volatility'].iloc[-1]) or 0.02
    
    # Market regime
    market_regime = MarketRegime.get_regime(df)
    
    # Dynamic threshold
    threshold_info = ProbabilityThresholds.get_dynamic_threshold(
        current_volatility, market_regime
    )
    dynamic_threshold = threshold_info['threshold']
    
    # Prepare features
    from sklearn.preprocessing import RobustScaler
    
    X = df[feature_cols].values.astype(float)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    seq_len = 60
    if len(X_scaled) < seq_len:
        seq_len = min(30, len(X_scaled))
    
    if len(X_scaled) < seq_len:
        raise ValueError(f"Insufficient data")
    
    X_seq = X_scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))
    
    # Predict
    predictions = model.predict(X_seq, verbose=0)
    week_prob_up = float(predictions[2][0, 0])
    
    # Direction
    week_direction = "UP" if week_prob_up > 0.5 else "DOWN"
    week_direction_emoji = "📈 UP" if week_direction == "UP" else "📉 DOWN"
    
    # Confidence
    confidence = ProbabilityThresholds.get_confidence_label(week_prob_up, dynamic_threshold)
    
    # Risk management
    risk_mgmt = RiskManagement.calculate_targets(
        current_price, current_atr, current_volatility, week_prob_up
    )
    
    # Decision logic
    reasoning = []
    warnings = []
    
    prob_margin = week_prob_up - dynamic_threshold
    
    if prob_margin <= 0:
        reasoning.append(f"❌ Probability too low ({week_prob_up:.1%} < {dynamic_threshold:.1%} threshold)")
        action = "❌ NO TRADE"
        signal_strength = "REJECTED"
    else:
        signal_strength = "VALID"
        reasoning.append(f"✅ Probability above threshold ({week_prob_up:.1%} > {dynamic_threshold:.1%})")
        
        if risk_mgmt['risk_reward'] < 1.5:
            reasoning.append(f"⚠️ Poor risk-reward ({risk_mgmt['risk_reward']:.2f}:1 < 1.5:1)")
            warnings.append(f"Risk-reward below 1.5:1")
        else:
            reasoning.append(f"✅ Good risk-reward ({risk_mgmt['risk_reward']:.2f}:1)")
        
        if "BULL" in market_regime and week_direction == "UP":
            reasoning.append(f"✅ Aligned with bull market")
        elif "BEAR" in market_regime and week_direction == "DOWN":
            reasoning.append(f"✅ Aligned with bear market")
        elif "SIDEWAYS" in market_regime:
            warnings.append("Market is sideways - choppy trading expected")
            reasoning.append(f"⚠️ Sideways market detected")
        else:
            warnings.append("Signal conflicts with market regime")
        
        if current_volatility > 0.03:
            warnings.append(f"High volatility ({current_volatility*100:.1f}%)")
        
        if signal_strength == "VALID" and len(warnings) == 0:
            action = f"🟢 BUY" if week_direction == "UP" else f"🔴 SELL"
        elif signal_strength == "VALID" and len(warnings) <= 1:
            action = f"⚡ CAUTION BUY" if week_direction == "UP" else f"⚡ CAUTION SELL"
        else:
            action = "⏸️ WAIT"
            reasoning.append(f"❌ Too many warnings")
    
    return StockPrediction(
        symbol=symbol,
        current_price=current_price,
        price_date=price_date,
        week_prob_up=week_prob_up,
        week_direction=week_direction_emoji,
        confidence=confidence,
        target_high=risk_mgmt['target_high'],
        target_low=risk_mgmt['target_low'],
        stop_loss=risk_mgmt['stop_loss'],
        risk_reward=risk_mgmt['risk_reward'],
        expected_return=risk_mgmt['expected_return'],
        max_loss=risk_mgmt['max_loss'],
        market_regime=market_regime,
        atr_pct=risk_mgmt['atr_pct'],
        volatility=current_volatility,
        dynamic_threshold=dynamic_threshold,
        action=action,
        signal_strength=signal_strength,
        reasoning=reasoning,
        warnings=warnings
    )

# ============================================================================
# CSV LOGGING
# ============================================================================
def log_to_csv(predictions: List[StockPrediction], filename: str = "predictions_log.csv"):
    """Log predictions to CSV for analysis"""
    
    csv_path = Path(filename)
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'symbol', 'price_date', 'current_price',
            'week_prob_up', 'week_direction', 'confidence',
            'target_high', 'target_low', 'stop_loss',
            'risk_reward', 'expected_return', 'max_loss',
            'market_regime', 'atr_pct', 'volatility',
            'dynamic_threshold', 'action', 'signal_strength', 'warnings'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for pred in predictions:
            # Remove emojis for CSV compatibility
            week_direction_clean = pred.week_direction.replace('📈', '').replace('📉', '').strip()
            market_regime_clean = pred.market_regime.replace('📈', '').replace('📉', '').replace('⚖️', '').replace('🔄', '').strip()
            confidence_clean = pred.confidence.replace('🟢', '').replace('🟡', '').replace('🟠', '').replace('🔴', '').strip()
            action_clean = pred.action.replace('🟢', '').replace('🔴', '').replace('⚡', '').replace('❌', '').replace('⏸️', '').strip()
            
            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': pred.symbol,
                'price_date': pred.price_date,
                'current_price': pred.current_price,
                'week_prob_up': pred.week_prob_up,
                'week_direction': week_direction_clean,
                'confidence': confidence_clean,
                'target_high': pred.target_high,
                'target_low': pred.target_low,
                'stop_loss': pred.stop_loss,
                'risk_reward': pred.risk_reward,
                'expected_return': pred.expected_return,
                'max_loss': pred.max_loss,
                'market_regime': market_regime_clean,
                'atr_pct': pred.atr_pct,
                'volatility': pred.volatility,
                'dynamic_threshold': pred.dynamic_threshold,
                'action': action_clean,
                'signal_strength': pred.signal_strength,
                'warnings': '; '.join(pred.warnings) if pred.warnings else ''
            }
            writer.writerow(row)
    
    print(f"\n📊 Predictions logged to: {csv_path.absolute()}")

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def print_detailed(pred: StockPrediction):
    """Print detailed prediction - SIMPLIFIED"""
    print("\n" + "="*110)
    print(f"🔮 {pred.symbol} - WEEKLY PREDICTION (Latest Data: {pred.price_date})")
    print("="*110)
    
    print(f"\n{'CURRENT SITUATION':─^110}")
    print(f"Price:        ${pred.current_price:.2f}")
    print(f"Data Date:    {pred.price_date}")
    print(f"Direction:    {pred.week_direction} (Probability: {pred.week_prob_up:.1%})")
    print(f"Confidence:   {pred.confidence}")
    print(f"Threshold:    {pred.dynamic_threshold:.1%} (dynamic)")
    print(f"Volatility:   {pred.volatility*100:.2f}% | ATR: {pred.atr_pct:.2f}%")
    print(f"Market:       {pred.market_regime}")
    
    print(f"\n{'TARGETS & RISK MANAGEMENT':─^110}")
    print(f"Entry:             ${pred.current_price:.2f}")
    print(f"Target Range:      ${pred.target_low:.2f} - ${pred.target_high:.2f}")
    print(f"Expected Return:   {pred.expected_return:+.2f}%")
    print(f"Stop Loss:         ${pred.stop_loss:.2f}")
    print(f"Max Loss:          -{pred.max_loss:.2f}%")
    print(f"Risk-Reward:       {pred.risk_reward:.2f}:1")
    
    print(f"\n{'DECISION & REASONING':─^110}")
    print(f"Action: {pred.action}")
    print(f"Signal: {pred.signal_strength}")
    
    for reason in pred.reasoning:
        print(f"  {reason}")
    
    if pred.warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in pred.warnings:
            print(f"  • {warning}")
    
    print("="*110 + "\n")

def print_portfolio(predictions: List[StockPrediction]):
    """Print portfolio summary"""
    print("\n" + "="*150)
    print("📊 PORTFOLIO ANALYSIS - DYNAMIC THRESHOLDS")
    print("="*150)
    
    print(f"\n{'Stock':<8} {'Date':<12} {'Price':<12} {'Dir':<10} {'Prob':<10} {'Threshold':<12} {'Target Range':<22} {'R:R':<8} {'Action':<20}")
    print("-"*150)
    
    for p in predictions:
        print(f"{p.symbol:<8} "
              f"{p.price_date:<12} "
              f"${p.current_price:<11.2f} "
              f"{p.week_direction:<10} "
              f"{p.week_prob_up:<9.1%} "
              f"{p.dynamic_threshold:<11.1%} "
              f"${p.target_low:.0f}-${p.target_high:.0f}         "
              f"{p.risk_reward:<7.2f} "
              f"{p.action:<20}")
    
    print("="*150)
    
    # Summary statistics
    trades = sum(1 for p in predictions if "BUY" in p.action or "SELL" in p.action)
    no_trade = sum(1 for p in predictions if "NO TRADE" in p.action or "WAIT" in p.action)
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total Analyzed: {len(predictions)} stocks")
    print(f"   🟢 TRADE Signals: {trades} | ⏸️ NO TRADE: {no_trade}")
    print(f"   Average Probability: {np.mean([p.week_prob_up for p in predictions]):.1%}")
    print(f"   Average Risk-Reward: {np.mean([p.risk_reward for p in predictions]):.2f}:1")
    print("="*150 + "\n")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Stock Predictor - Reads from LOCAL CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py -s AAPL
  python predict.py -s AAPL MSFT GOOGL
  python predict.py --portfolio
  python predict.py --check
  
Note: 
  - The model reads from CSV files in: data/stock_data/{SYMBOL}.csv
  - Automatically uses the LATEST date available in your CSV
  - Update your CSV daily and the model will always use the most recent price
  
CSV Locations Checked (in order):
  1. data/stock_data/AAPL.csv
  2. data/AAPL.csv
  3. stock_data/AAPL.csv
  4. AAPL.csv
        """
    )
    
    parser.add_argument("-s", "--stocks", nargs="+", help="Stock symbols")
    parser.add_argument("-p", "--portfolio", action="store_true", help="Default portfolio")
    parser.add_argument("--check", action="store_true", help="Check setup")
    parser.add_argument("--no-log", action="store_true", help="Don't log to CSV")
    parser.add_argument("--refresh", action="store_true", help="Force refresh data (ignore cache)")
    
    args = parser.parse_args()
    
    if args.check:
        print("\n" + "="*80)
        print("✅ SETUP CHECK")
        print("="*80)
        
        model_paths = [
            Path("models/stock_model_fixed.keras"),
            Path("./models/stock_model_fixed.keras"),
            Path(__file__).parent / "models" / "stock_model_fixed.keras",
        ]
        
        found = False
        for p in model_paths:
            status = "✅" if p.exists() else "❌"
            print(f"   {status} {p}")
            if p.exists():
                found = True
        
        print("\n" + "="*80)
        if found:
            print("✅ Ready! Run: python predict.py --portfolio")
        else:
            print("❌ Run: python train_fixed.py")
        print("="*80 + "\n")
        return
    
    if args.portfolio:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']
    elif args.stocks:
        symbols = args.stocks
    else:
        print("\n❌ Usage:")
        print("   python predict.py -s AAPL MSFT")
        print("   python predict.py --portfolio")
        print("   python predict.py --check\n")
        sys.exit(1)
    
    print(f"\n🚀 Analyzing {len(symbols)} stocks (using LATEST data from CSV)")
    print(f"   Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    predictions = []
    for symbol in symbols:
        try:
            print(f"   {symbol}...", end=" ", flush=True)
            pred = predict_stock(symbol)
            predictions.append(pred)
            print(f"✅ (Data: {pred.price_date})")
        except Exception as e:
            print(f"❌ ({str(e)})")
    
    if not predictions:
        print("\n❌ No predictions")
        sys.exit(1)
    
    # Display results
    if len(predictions) > 1:
        print_portfolio(predictions)
    
    for pred in predictions:
        print_detailed(pred)
    
    # Log to CSV
    if not args.no_log:
        log_to_csv(predictions)

if __name__ == "__main__":
    main()