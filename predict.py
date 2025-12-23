#!/usr/bin/env python3
"""
Enhanced Stock Predictor v2 - Advanced Risk Management & Analysis
✅ Improved R:R calculation (ensures > 1.5:1)
✅ Per-stock adaptive thresholds
✅ Better market regime detection
✅ Comparative table format for multiple stocks
✅ Weighted confidence scoring
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
from typing import List, Dict
import argparse
import csv

tf.get_logger().setLevel('ERROR')

# ============================================================================
# ADAPTIVE PROBABILITY THRESHOLDS (Per-Stock)
# ============================================================================
class AdaptiveThresholds:
    """Per-stock adaptive thresholds based on historical behavior"""
    
    @staticmethod
    def calculate_stock_threshold(df: pd.DataFrame, volatility: float, market_regime: str) -> dict:
        """
        Calculate adaptive threshold per stock based on:
        - Stock's historical volatility pattern
        - Market regime
        - Trend consistency
        """
        base_threshold = 0.58  # Slightly lower base
        
        # Historical volatility pattern (last 60 days)
        if len(df) >= 60:
            recent_vol = df['close'].pct_change().iloc[-60:].std()
            long_vol = df['close'].pct_change().std()
            vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0
            
            # If recent volatility much higher than normal, be more conservative
            if vol_ratio > 1.5:
                vol_adj = 0.06
            elif vol_ratio > 1.2:
                vol_adj = 0.04
            elif vol_ratio < 0.8:
                vol_adj = -0.02  # More aggressive in stable periods
            else:
                vol_adj = 0.02
        else:
            # Fallback to absolute volatility
            if volatility > 0.04:
                vol_adj = 0.07
            elif volatility > 0.03:
                vol_adj = 0.05
            elif volatility > 0.02:
                vol_adj = 0.03
            else:
                vol_adj = 0.00
        
        # Trend consistency (helps differentiate MIXED from real trends)
        if len(df) >= 20:
            closes = df['close'].iloc[-20:]
            returns = closes.pct_change().dropna()
            
            # Count directional consistency
            positive_days = (returns > 0).sum()
            trend_consistency = abs(positive_days - 10) / 10  # 0 = random, 1 = strong trend
            
            if trend_consistency > 0.6:  # Strong trend
                regime_adj = -0.03  # Can be more aggressive
            elif trend_consistency < 0.3:  # Choppy/mixed
                regime_adj = 0.05  # Be conservative
            else:
                regime_adj = 0.02
        else:
            regime_adj = 0.02
        
        # Market regime fine-tuning
        if "BULL STRONG" in market_regime:
            regime_adj -= 0.02
        elif "BEAR STRONG" in market_regime:
            regime_adj += 0.03
        elif "MIXED" in market_regime or "SIDEWAYS" in market_regime:
            regime_adj += 0.04  # Much more conservative in choppy markets
        
        adjusted_threshold = base_threshold + vol_adj + regime_adj
        adjusted_threshold = max(0.52, min(0.75, adjusted_threshold))  # Clamp 52-75%
        
        return {
            'threshold': adjusted_threshold,
            'vol_adjustment': vol_adj,
            'regime_adjustment': regime_adj,
            'trend_consistency': trend_consistency if len(df) >= 20 else 0.5
        }
    
    @staticmethod
    def get_confidence_label(prob: float, threshold: float) -> tuple:
        """Return confidence label and score (0-100)"""
        margin = prob - threshold
        
        # Calculate confidence score (0-100)
        if margin >= 0:
            score = min(100, 50 + (margin * 200))  # Above threshold: 50-100
        else:
            score = max(0, 50 + (margin * 200))  # Below threshold: 0-50
        
        # Label based on margin
        if margin >= 0.12:
            label = "🟢 VERY HIGH"
        elif margin >= 0.07:
            label = "🟢 HIGH"
        elif margin >= 0.03:
            label = "🟡 MEDIUM"
        elif margin >= 0.00:
            label = "🟡 LOW-MED"
        elif margin >= -0.05:
            label = "🟠 LOW"
        else:
            label = "🔴 VERY LOW"
        
        return label, score

# ============================================================================
# ENHANCED MARKET REGIME DETECTION
# ============================================================================
class EnhancedMarketRegime:
    """More sophisticated regime detection"""
    
    @staticmethod
    def analyze_regime(df: pd.DataFrame) -> dict:
        """Comprehensive market regime analysis"""
        if len(df) < 50:
            return {
                'regime': "⚠️ INSUFFICIENT DATA",
                'trend_strength': 0,
                'volatility_regime': "UNKNOWN"
            }
        
        close = df['close'].iloc[-50:]
        full_close = df['close']
        
        # Moving averages
        ma_20 = close.iloc[-20:].mean()
        ma_50 = close.mean()
        ma_200 = full_close.mean() if len(full_close) >= 200 else ma_50
        
        current = close.iloc[-1]
        
        # Volatility analysis
        vol_20d = close.pct_change().iloc[-20:].std() * 100
        vol_50d = close.pct_change().std() * 100
        
        # Trend strength (linear regression slope)
        x = np.arange(len(close))
        y = close.values
        slope = np.polyfit(x, y, 1)[0]
        trend_strength = abs(slope) / current * 100  # % per day
        
        # Directional consistency
        returns = close.pct_change().dropna()
        positive_days = (returns > 0).sum()
        consistency = abs(positive_days - len(returns)/2) / (len(returns)/2)
        
        # Determine volatility regime
        if vol_20d < 1.5:
            vol_regime = "LOW VOL"
        elif vol_20d < 2.5:
            vol_regime = "NORMAL VOL"
        elif vol_20d < 4.0:
            vol_regime = "HIGH VOL"
        else:
            vol_regime = "EXTREME VOL"
        
        # Determine trend regime with more precision
        if trend_strength > 0.15 and consistency > 0.4:  # Strong persistent trend
            if current > ma_20 > ma_50 and slope > 0:
                regime = "🚀 BULL STRONG"
            elif current < ma_20 < ma_50 and slope < 0:
                regime = "📉 BEAR STRONG"
            else:
                regime = "🔄 TRANSITIONING"
        elif trend_strength > 0.08 and consistency > 0.25:  # Moderate trend
            if current > ma_50 and slope > 0:
                regime = "📈 BULL"
            elif current < ma_50 and slope < 0:
                regime = "📉 BEAR"
            else:
                regime = "🔄 MIXED"
        elif consistency < 0.15:  # Very choppy
            regime = "⚡ CHOPPY"
        else:  # Sideways
            regime = "⚖️ SIDEWAYS"
        
        return {
            'regime': regime,
            'trend_strength': trend_strength,
            'consistency': consistency,
            'volatility_regime': vol_regime,
            'vol_20d': vol_20d
        }

# ============================================================================
# IMPROVED RISK MANAGEMENT
# ============================================================================
class ImprovedRiskManagement:
    """Enhanced risk management ensuring R:R > 1.5"""
    
    @staticmethod
    def calculate_optimal_levels(current_price: float, 
                                 atr: float, 
                                 volatility: float,
                                 direction_prob: float,
                                 trend_strength: float) -> dict:
        """
        Calculate targets and stops ensuring R:R > 1.5:1
        - Adjusts stop-loss to be tighter or targets to be wider
        - Uses trend strength to scale targets
        """
        
        # Base multipliers
        if direction_prob > 0.5:  # UP signal
            # Target multipliers based on confidence and trend
            conf_multiplier = 1.0 + (direction_prob - 0.5) * 2  # 1.0 to 2.0
            trend_multiplier = 1.0 + min(trend_strength * 5, 0.5)  # Up to 1.5x
            
            target_multiplier = conf_multiplier * trend_multiplier
            
            target_high = current_price + (atr * 2.0 * target_multiplier)
            target_low = current_price + (atr * 0.8 * target_multiplier)
            
            # Initial stop
            stop_loss = current_price - (atr * 1.2)
            
            # Check R:R and adjust
            avg_target = (target_high + target_low) / 2
            upside = avg_target - current_price
            downside = current_price - stop_loss
            initial_rr = upside / downside if downside > 0 else 0
            
            # If R:R < 1.5, tighten stop or widen target
            if initial_rr < 1.5 and downside > 0:
                # Prioritize tightening stop-loss first
                required_downside = upside / 1.5
                stop_loss = current_price - required_downside
                
                # But don't make stop too tight (min 0.8 ATR)
                min_stop = current_price - (atr * 0.8)
                if stop_loss > min_stop:
                    stop_loss = min_stop
                    # If stop is at minimum, widen targets instead
                    required_upside = downside * 1.5
                    center = current_price + required_upside
                    target_high = center + (atr * 0.6)
                    target_low = center - (atr * 0.3)
        
        else:  # DOWN signal
            conf_multiplier = 1.0 + (0.5 - direction_prob) * 2
            trend_multiplier = 1.0 + min(trend_strength * 5, 0.5)
            
            target_multiplier = conf_multiplier * trend_multiplier
            
            target_low = current_price - (atr * 2.0 * target_multiplier)
            target_high = current_price - (atr * 0.8 * target_multiplier)
            
            stop_loss = current_price + (atr * 1.2)
            
            avg_target = (target_high + target_low) / 2
            upside = current_price - avg_target
            downside = stop_loss - current_price
            initial_rr = upside / downside if downside > 0 else 0
            
            if initial_rr < 1.5 and downside > 0:
                required_downside = upside / 1.5
                stop_loss = current_price + required_downside
                
                max_stop = current_price + (atr * 0.8)
                if stop_loss < max_stop:
                    stop_loss = max_stop
                    required_upside = downside * 1.5
                    center = current_price - required_upside
                    target_low = center - (atr * 0.6)
                    target_high = center + (atr * 0.3)
        
        # Final calculations
        avg_target = (target_high + target_low) / 2
        upside_risk = abs(avg_target - current_price)
        downside_risk = abs(current_price - stop_loss)
        final_rr = upside_risk / downside_risk if downside_risk > 0 else 0
        
        target_return = (avg_target - current_price) / current_price * 100
        max_loss = abs(current_price - stop_loss) / current_price * 100
        
        return {
            'target_high': target_high,
            'target_low': target_low,
            'stop_loss': stop_loss,
            'risk_reward': final_rr,
            'expected_return': target_return,
            'max_loss': max_loss,
            'atr_pct': (atr / current_price) * 100
        }

# ============================================================================
# WEIGHTED DECISION SCORING
# ============================================================================
class WeightedDecisionEngine:
    """Score-based decision making"""
    
    @staticmethod
    def calculate_signal_score(pred_data: dict) -> dict:
        """
        Calculate weighted score (0-100) for trade decision
        Factors:
        - Probability margin: 40%
        - Risk-reward: 25%
        - Market alignment: 20%
        - Volatility: 15%
        """
        score = 0
        breakdown = {}
        
        # 1. Probability margin (40 points)
        prob_margin = pred_data['week_prob_up'] - pred_data['threshold']
        if prob_margin >= 0.10:
            prob_score = 40
        elif prob_margin >= 0.05:
            prob_score = 32
        elif prob_margin >= 0.02:
            prob_score = 25
        elif prob_margin >= 0:
            prob_score = 18
        elif prob_margin >= -0.03:
            prob_score = 10
        else:
            prob_score = 0
        
        score += prob_score
        breakdown['probability'] = prob_score
        
        # 2. Risk-reward (25 points)
        rr = pred_data['risk_reward']
        if rr >= 2.5:
            rr_score = 25
        elif rr >= 2.0:
            rr_score = 20
        elif rr >= 1.5:
            rr_score = 15
        elif rr >= 1.0:
            rr_score = 8
        else:
            rr_score = 0
        
        score += rr_score
        breakdown['risk_reward'] = rr_score
        
        # 3. Market alignment (20 points)
        regime = pred_data['market_regime']
        direction = pred_data['week_direction']
        
        if ("BULL STRONG" in regime and "UP" in direction) or \
           ("BEAR STRONG" in regime and "DOWN" in direction):
            market_score = 20
        elif ("BULL" in regime and "UP" in direction) or \
             ("BEAR" in regime and "DOWN" in direction):
            market_score = 15
        elif "CHOPPY" in regime or "MIXED" in regime:
            market_score = 5
        elif "SIDEWAYS" in regime:
            market_score = 8
        else:
            market_score = 0
        
        score += market_score
        breakdown['market_alignment'] = market_score
        
        # 4. Volatility favorability (15 points)
        vol = pred_data['volatility']
        if vol < 0.02:  # Low vol - ideal
            vol_score = 15
        elif vol < 0.03:  # Moderate
            vol_score = 12
        elif vol < 0.04:  # High
            vol_score = 7
        else:  # Very high
            vol_score = 2
        
        score += vol_score
        breakdown['volatility'] = vol_score
        
        # Determine action based on score
        if score >= 75:
            action = "🟢 STRONG BUY" if "UP" in direction else "🔴 STRONG SELL"
            signal = "EXCELLENT"
        elif score >= 65:
            action = "🟢 BUY" if "UP" in direction else "🔴 SELL"
            signal = "GOOD"
        elif score >= 55:
            action = "⚡ CAUTIOUS BUY" if "UP" in direction else "⚡ CAUTIOUS SELL"
            signal = "MARGINAL"
        elif score >= 45:
            action = "⏸️ WAIT"
            signal = "WEAK"
        else:
            action = "❌ NO TRADE"
            signal = "REJECTED"
        
        return {
            'score': score,
            'action': action,
            'signal_strength': signal,
            'breakdown': breakdown
        }

# ============================================================================
# TECHNICAL INDICATORS (unchanged)
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
# MARKET DATA FETCHER (unchanged)
# ============================================================================
class MarketDataFetcher:
    @staticmethod
    def fetch_market_trend(df: pd.DataFrame, retries: int = 3) -> pd.DataFrame:
        import logging
        yf_logger = logging.getLogger('yfinance')
        yf_logger.setLevel(logging.CRITICAL)
        
        for attempt in range(retries):
            try:
                import yfinance as yf
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    market_df = yf.download('SPY', start=start_date, end=end_date, 
                                          progress=False, timeout=10, show_errors=False)
                
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
                if attempt < retries - 1:
                    continue
        
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['market_trend'] = (df['close'] > df['ema_200']).astype(int)
        df = df.drop('ema_200', axis=1)
        return df

# ============================================================================
# FEATURE ENGINEERING (unchanged)
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
# ENHANCED PREDICTION DATA CLASS
# ============================================================================
@dataclass
class EnhancedStockPrediction:
    symbol: str
    current_price: float
    price_date: str
    
    week_prob_up: float
    week_direction: str
    confidence: str
    confidence_score: float
    
    target_high: float
    target_low: float
    stop_loss: float
    risk_reward: float
    expected_return: float
    max_loss: float
    
    market_regime: str
    trend_strength: float
    volatility: float
    volatility_regime: str
    atr_pct: float
    
    adaptive_threshold: float
    threshold_breakdown: dict
    
    signal_score: float
    action: str
    signal_strength: str
    score_breakdown: dict
    
    reasoning: List[str]
    warnings: List[str]

# ============================================================================
# DATA LOADING (unchanged)
# ============================================================================
def load_and_prepare_data(symbol: str):
    csv_paths = [
        Path(f"data/stock_data/{symbol}.csv"),
        Path(f"data/{symbol}.csv"),
        Path(f"stock_data/{symbol}.csv"),
        Path(f"{symbol}.csv"),
        Path(f"data/stock_data/{symbol}_daily.csv"),
    ]
    
    df = None
    csv_found = None
    
    for csv_path in csv_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                csv_found = csv_path
                print(f" [Loading from: {csv_path}]", end="")
                break
            except:
                continue
    
    if df is None:
        sys.path.append(str(Path(__file__).parent))
        try:
            from src.data_loader import fetch_stock_data
            df = fetch_stock_data(symbol, use_cache=True)
        except:
            raise ValueError(f"Could not load data for {symbol}")
    
    if df.empty:
        raise ValueError(f"Could not fetch data for {symbol}")
    
    df.columns = df.columns.str.lower()
    
    date_columns = ['date', 'datetime', 'timestamp', 'time']
    date_col = None
    
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None and len(df.columns) > 0:
        first_col = df.columns[0]
        if df[first_col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any():
            date_col = first_col
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df.set_index(date_col, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]
        except:
            raise ValueError(f"Could not parse dates from CSV for {symbol}")
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    
    if len(df) > 0:
        print(f" [{len(df)} rows, latest: {df.index[-1].strftime('%Y-%m-%d')}]", end="")
    else:
        raise ValueError(f"No valid data found in CSV for {symbol}")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
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

# ============================================================================
# ENHANCED PREDICTION ENGINE
# ============================================================================
def predict_stock_enhanced(symbol: str) -> EnhancedStockPrediction:
    """Enhanced prediction with all improvements"""
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
    
    df, feature_cols = load_and_prepare_data(symbol)
    
    current_price = float(df['close'].iloc[-1])
    price_date = df.index[-1].strftime('%Y-%m-%d')
    current_atr = float(df['atr'].iloc[-1]) or 1.0
    current_volatility = float(df['volatility'].iloc[-1]) or 0.02
    
    # Enhanced market regime
    regime_analysis = EnhancedMarketRegime.analyze_regime(df)
    market_regime = regime_analysis['regime']
    trend_strength = regime_analysis['trend_strength']
    volatility_regime = regime_analysis['volatility_regime']
    
    # Adaptive threshold per stock
    threshold_info = AdaptiveThresholds.calculate_stock_threshold(
        df, current_volatility, market_regime
    )
    adaptive_threshold = threshold_info['threshold']
    
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
    
    # Confidence with score
    confidence, confidence_score = AdaptiveThresholds.get_confidence_label(
        week_prob_up, adaptive_threshold
    )
    
    # Improved risk management
    risk_mgmt = ImprovedRiskManagement.calculate_optimal_levels(
        current_price, current_atr, current_volatility, week_prob_up, trend_strength
    )
    
    # Weighted decision scoring
    pred_data = {
        'week_prob_up': week_prob_up,
        'threshold': adaptive_threshold,
        'risk_reward': risk_mgmt['risk_reward'],
        'market_regime': market_regime,
        'week_direction': week_direction,
        'volatility': current_volatility
    }
    decision = WeightedDecisionEngine.calculate_signal_score(pred_data)
    
    # Reasoning and warnings
    reasoning = []
    warnings = []
    
    prob_margin = week_prob_up - adaptive_threshold
    
    # Probability assessment
    if prob_margin > 0.07:
        reasoning.append(f"✅ Strong probability ({week_prob_up:.1%} >> {adaptive_threshold:.1%} threshold)")
    elif prob_margin > 0:
        reasoning.append(f"✅ Probability above threshold ({week_prob_up:.1%} > {adaptive_threshold:.1%})")
    else:
        reasoning.append(f"❌ Probability too low ({week_prob_up:.1%} < {adaptive_threshold:.1%})")
    
    # Risk-reward assessment
    if risk_mgmt['risk_reward'] >= 2.0:
        reasoning.append(f"✅ Excellent R:R ({risk_mgmt['risk_reward']:.2f}:1)")
    elif risk_mgmt['risk_reward'] >= 1.5:
        reasoning.append(f"✅ Good R:R ({risk_mgmt['risk_reward']:.2f}:1)")
    else:
        reasoning.append(f"⚠️ Poor R:R ({risk_mgmt['risk_reward']:.2f}:1)")
        warnings.append(f"Risk-reward at {risk_mgmt['risk_reward']:.2f}:1")
    
    # Market alignment
    if ("BULL" in market_regime and week_direction == "UP") or \
       ("BEAR" in market_regime and week_direction == "DOWN"):
        reasoning.append(f"✅ Aligned with {market_regime}")
    elif "CHOPPY" in market_regime:
        warnings.append("Choppy market - high risk")
        reasoning.append(f"⚠️ Choppy market detected")
    elif "MIXED" in market_regime or "SIDEWAYS" in market_regime:
        warnings.append("Market lacks clear direction")
        reasoning.append(f"⚠️ {market_regime}")
    else:
        warnings.append("Signal conflicts with market regime")
    
    # Volatility check
    if current_volatility > 0.04:
        warnings.append(f"Very high volatility ({current_volatility*100:.1f}%)")
    elif current_volatility > 0.03:
        warnings.append(f"High volatility ({current_volatility*100:.1f}%)")
    
    # Score breakdown
    reasoning.append(f"📊 Signal Score: {decision['score']:.0f}/100")
    
    return EnhancedStockPrediction(
        symbol=symbol,
        current_price=current_price,
        price_date=price_date,
        week_prob_up=week_prob_up,
        week_direction=week_direction_emoji,
        confidence=confidence,
        confidence_score=confidence_score,
        target_high=risk_mgmt['target_high'],
        target_low=risk_mgmt['target_low'],
        stop_loss=risk_mgmt['stop_loss'],
        risk_reward=risk_mgmt['risk_reward'],
        expected_return=risk_mgmt['expected_return'],
        max_loss=risk_mgmt['max_loss'],
        market_regime=market_regime,
        trend_strength=trend_strength,
        volatility=current_volatility,
        volatility_regime=volatility_regime,
        atr_pct=risk_mgmt['atr_pct'],
        adaptive_threshold=adaptive_threshold,
        threshold_breakdown=threshold_info,
        signal_score=decision['score'],
        action=decision['action'],
        signal_strength=decision['signal_strength'],
        score_breakdown=decision['breakdown'],
        reasoning=reasoning,
        warnings=warnings
    )

# ============================================================================
# CSV LOGGING
# ============================================================================
def log_to_csv(predictions: List[EnhancedStockPrediction], filename: str = "predictions_log.csv"):
    """Log predictions to CSV"""
    csv_path = Path(filename)
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'timestamp', 'symbol', 'price_date', 'current_price',
            'week_prob_up', 'week_direction', 'confidence', 'confidence_score',
            'target_high', 'target_low', 'stop_loss',
            'risk_reward', 'expected_return', 'max_loss',
            'market_regime', 'trend_strength', 'volatility', 'volatility_regime',
            'adaptive_threshold', 'signal_score', 'action', 'signal_strength', 'warnings'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for pred in predictions:
            # Clean emojis
            week_direction_clean = pred.week_direction.replace('📈', '').replace('📉', '').strip()
            market_regime_clean = pred.market_regime.replace('🚀', '').replace('📈', '').replace('📉', '').replace('⚖️', '').replace('🔄', '').replace('⚡', '').strip()
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
                'confidence_score': pred.confidence_score,
                'target_high': pred.target_high,
                'target_low': pred.target_low,
                'stop_loss': pred.stop_loss,
                'risk_reward': pred.risk_reward,
                'expected_return': pred.expected_return,
                'max_loss': pred.max_loss,
                'market_regime': market_regime_clean,
                'trend_strength': pred.trend_strength,
                'volatility': pred.volatility,
                'volatility_regime': pred.volatility_regime,
                'adaptive_threshold': pred.adaptive_threshold,
                'signal_score': pred.signal_score,
                'action': action_clean,
                'signal_strength': pred.signal_strength,
                'warnings': '; '.join(pred.warnings) if pred.warnings else ''
            }
            writer.writerow(row)
    
    print(f"\n📊 Predictions logged to: {csv_path.absolute()}")

# ============================================================================
# ENHANCED DISPLAY - COMPARATIVE TABLE
# ============================================================================
def print_comparative_table(predictions: List[EnhancedStockPrediction]):
    """Compact comparative table for multiple stocks"""
    
    print("\n" + "="*165)
    print("📊 STOCK COMPARISON TABLE - ENHANCED ANALYSIS")
    print("="*165)
    
    # Header
    header = (
        f"{'Stock':<7} "
        f"{'Date':<11} "
        f"{'Price':<10} "
        f"{'Dir':<8} "
        f"{'Prob':<7} "
        f"{'Thresh':<7} "
        f"{'Conf':<14} "
        f"{'Score':<6} "
        f"{'Targets':<18} "
        f"{'R:R':<6} "
        f"{'Regime':<17} "
        f"{'Action':<18}"
    )
    print(header)
    print("-"*165)
    
    # Rows
    for p in predictions:
        # Shorten labels for table
        dir_short = "📈UP" if "UP" in p.week_direction else "📉DN"
        conf_short = p.confidence.replace(' HIGH', '').replace(' MEDIUM', '').replace(' LOW', '')
        regime_short = p.market_regime[:15]
        
        row = (
            f"{p.symbol:<7} "
            f"{p.price_date:<11} "
            f"${p.current_price:<9.2f} "
            f"{dir_short:<8} "
            f"{p.week_prob_up:<6.1%} "
            f"{p.adaptive_threshold:<6.1%} "
            f"{conf_short:<14} "
            f"{p.signal_score:<5.0f} "
            f"${p.target_low:.0f}-${p.target_high:.0f}     "
            f"{p.risk_reward:<5.2f} "
            f"{regime_short:<17} "
            f"{p.action:<18}"
        )
        print(row)
    
    print("="*165)
    
    # Summary stats
    trades = sum(1 for p in predictions if "BUY" in p.action or "SELL" in p.action)
    strong_signals = sum(1 for p in predictions if p.signal_score >= 75)
    rejected = sum(1 for p in predictions if "NO TRADE" in p.action)
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total Analyzed: {len(predictions)}")
    print(f"   🟢 Trade Signals: {trades} ({trades/len(predictions)*100:.0f}%)")
    print(f"   ⭐ Strong Signals (Score ≥75): {strong_signals}")
    print(f"   ❌ Rejected: {rejected}")
    print(f"   Average Score: {np.mean([p.signal_score for p in predictions]):.1f}/100")
    print(f"   Average R:R: {np.mean([p.risk_reward for p in predictions]):.2f}:1")
    print(f"   Average Probability: {np.mean([p.week_prob_up for p in predictions]):.1%}")
    
    # Best opportunities
    if trades > 0:
        trade_preds = [p for p in predictions if "BUY" in p.action or "SELL" in p.action]
        best = max(trade_preds, key=lambda x: x.signal_score)
        print(f"\n🏆 BEST OPPORTUNITY: {best.symbol} (Score: {best.signal_score:.0f}, R:R: {best.risk_reward:.2f}:1)")
    
    print("="*165 + "\n")

def print_detailed_analysis(pred: EnhancedStockPrediction):
    """Detailed individual stock analysis"""
    print("\n" + "="*110)
    print(f"🔍 {pred.symbol} - DETAILED ANALYSIS (Data: {pred.price_date})")
    print("="*110)
    
    print(f"\n{'MARKET SITUATION':─^110}")
    print(f"Current Price:    ${pred.current_price:.2f}")
    print(f"Direction:        {pred.week_direction} (Probability: {pred.week_prob_up:.1%})")
    print(f"Confidence:       {pred.confidence} (Score: {pred.confidence_score:.0f}/100)")
    print(f"Adaptive Thresh:  {pred.adaptive_threshold:.1%} (vol_adj: {pred.threshold_breakdown['vol_adjustment']:+.1%}, regime_adj: {pred.threshold_breakdown['regime_adjustment']:+.1%})")
    print(f"Market Regime:    {pred.market_regime} (Trend: {pred.trend_strength:.2%}/day)")
    print(f"Volatility:       {pred.volatility*100:.2f}% ({pred.volatility_regime}) | ATR: {pred.atr_pct:.2f}%")
    
    print(f"\n{'RISK MANAGEMENT':─^110}")
    print(f"Entry Price:       ${pred.current_price:.2f}")
    print(f"Target Range:      ${pred.target_low:.2f} - ${pred.target_high:.2f}")
    print(f"Expected Return:   {pred.expected_return:+.2f}%")
    print(f"Stop Loss:         ${pred.stop_loss:.2f}")
    print(f"Max Loss:          -{pred.max_loss:.2f}%")
    print(f"Risk-Reward:       {pred.risk_reward:.2f}:1 {'✅' if pred.risk_reward >= 1.5 else '⚠️'}")
    
    print(f"\n{'SIGNAL ANALYSIS':─^110}")
    print(f"Signal Score:      {pred.signal_score:.0f}/100")
    print(f"Signal Strength:   {pred.signal_strength}")
    print(f"Action:            {pred.action}")
    
    print(f"\nScore Breakdown:")
    for component, score in pred.score_breakdown.items():
        print(f"  {component.replace('_', ' ').title():<20} {score:>3.0f} points")
    
    print(f"\n{'REASONING':─^110}")
    for reason in pred.reasoning:
        print(f"  {reason}")
    
    if pred.warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in pred.warnings:
            print(f"  • {warning}")
    
    print("="*110 + "\n")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Stock Predictor v2 - Advanced Risk & Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py -s AAPL
  python predict.py -s AAPL MSFT GOOGL --table
  python predict.py --portfolio
  python predict.py -s AAPL --detailed
  
Features:
  ✅ Improved R:R calculation (ensures > 1.5:1)
  ✅ Per-stock adaptive thresholds
  ✅ Enhanced market regime detection
  ✅ Weighted signal scoring (0-100)
  ✅ Comparative table format
        """
    )
    
    parser.add_argument("-s", "--stocks", nargs="+", help="Stock symbols")
    parser.add_argument("-p", "--portfolio", action="store_true", help="Default portfolio")
    parser.add_argument("--table", action="store_true", help="Show comparative table (auto for 2+ stocks)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis for each stock")
    parser.add_argument("--no-log", action="store_true", help="Don't log to CSV")
    parser.add_argument("--check", action="store_true", help="Check setup")
    
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
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD']
    elif args.stocks:
        symbols = args.stocks
    else:
        print("\n❌ Usage:")
        print("   python predict.py -s AAPL MSFT")
        print("   python predict.py --portfolio")
        print("   python predict.py --check\n")
        sys.exit(1)
    
    print(f"\n🚀 Analyzing {len(symbols)} stocks with Enhanced v2 Model")
    print(f"   Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    predictions = []
    for symbol in symbols:
        try:
            print(f"   {symbol}...", end=" ", flush=True)
            pred = predict_stock_enhanced(symbol)
            predictions.append(pred)
            print(f"✅ (Data: {pred.price_date}, Score: {pred.signal_score:.0f})")
        except Exception as e:
            print(f"❌ ({str(e)})")
    
    if not predictions:
        print("\n❌ No predictions generated")
        sys.exit(1)
    
    # Display results
    if len(predictions) > 1 or args.table:
        print_comparative_table(predictions)
    
    if args.detailed or len(predictions) == 1:
        for pred in predictions:
            print_detailed_analysis(pred)
    
    # Log to CSV
    if not args.no_log:
        log_to_csv(predictions)

if __name__ == "__main__":
    main()
