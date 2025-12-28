#!/usr/bin/env python3
"""
Enhanced Stock Predictor v2 - Advanced Risk Management & Analysis
✅ Real-time price fetching from yfinance
✅ Improved R:R calculation (ensures > 1.5:1)
✅ Per-stock adaptive thresholds
✅ Better market regime detection
✅ Comparative table format for multiple stocks
✅ Weighted confidence scoring
"""

# =================================================================
# ENVIRONMENT SETUP
# =================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_SUPPRESS_LOGS'] = '1'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import warnings
warnings.filterwarnings('ignore')

# =================================================================
# CORE IMPORTS
# =================================================================
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import yfinance as yf
from sklearn.preprocessing import RobustScaler

# =================================================================
# LOGGING SETUP
# =================================================================
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# =================================================================
# TENSORFLOW IMPORT (WITH ERROR HANDLING)
# =================================================================
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    print(f"⚠️  Warning: TensorFlow not available - {str(e)}")

# =================================================================
# HELPER FUNCTION FOR SAFE MODEL LOADING
# =================================================================
def load_model_safe(model_path):
    """Load Keras model with compatibility fixes for different versions"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required but not installed")
    
    try:
        # Try standard loading first
        model = load_model(str(model_path))
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"⚠️  Standard load failed, trying compatibility mode...")
        
        # Compatibility mode with custom objects
        from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Zeros, Ones
        from tensorflow.keras.regularizers import L2
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer
        
        custom_objects = {
            'GlorotUniform': GlorotUniform,
            'Orthogonal': Orthogonal,
            'Zeros': Zeros,
            'Ones': Ones,
            'L2': L2,
            'LSTM': LSTM,
            'Dense': Dense,
            'Dropout': Dropout,
            'BatchNormalization': BatchNormalization,
            'InputLayer': InputLayer,
        }
        
        try:
            model = load_model(
                str(model_path),
                custom_objects=custom_objects,
                safe_mode=False
            )
            print(f"✅ Model loaded with custom objects")
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load model: {str(e2)}")
# ============================================================================
# REAL-TIME PRICE FETCHER
# ============================================================================
class RealTimePriceFetcher:
    """Fetch current prices from yfinance"""
    
    @staticmethod
    def get_current_price(symbol: str) -> Dict:
        """
        Fetch real-time price data from yfinance
        Intelligently handles market hours and weekend/holidays
        """
        
        try:
            import yfinance as yf
            from datetime import datetime, time as dt_time
        
            ticker = yf.Ticker(symbol)
            now = datetime.now()
        
        # Define market close time (4:00 PM ET = 16:00)
        # Adjust if you're in a different timezone
            market_close = dt_time(16, 0)
            current_time = now.time()
        
        # If it's a weekend, get Friday's data
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                hist = ticker.history(period="5d")
        # If before market close + 2 hours (6 PM), use yesterday
            elif current_time < dt_time(18, 0):
                hist = ticker.history(period="5d")
        # After 6 PM, try to get today's closing price
            else:
                hist = ticker.history(period="2d")
        
            if hist.empty:
                raise ValueError(f"No price data available")
        
        # Get the most recent trading day
            latest_data = hist.iloc[-1]
            latest_date = hist.index[-1]
        
        # Check if today's data is available (after market close)
            today = now.date()
            if latest_date.date() == today and current_time >= dt_time(18, 0):
            # Use today's closing price (after 6 PM)
                print(f" [Using TODAY's close]", end="")
            else:
            # Use most recent available (likely yesterday)
                print(f" [Using {latest_date.strftime('%Y-%m-%d')} close]", end="")
        
            return {
                'current_price': float(latest_data['Close']),
                'price_date': latest_date.strftime('%Y-%m-%d'),
                'high': float(latest_data['High']),
                'low': float(latest_data['Low']),
                'open': float(latest_data['Open']),
                'volume': float(latest_data['Volume']),
                    'datetime': latest_date
        }
        
        except Exception as e:
            raise ValueError(f"Could not fetch current price for {symbol}: {str(e)}")
    
    @staticmethod
    def update_df_with_current_price(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Update DataFrame with real-time price data, with retries"""
        max_retries = 3
        retry_delay = 2
    
        for attempt in range(max_retries):
            try:
                current_data = RealTimePriceFetcher.get_current_price(symbol)
            
            # Get dates for comparison
                latest_df_date = df.index[-1]
                current_date = current_data['datetime']
            
            # Remove timezone for comparison if needed
                if hasattr(current_date, 'tz') and current_date.tz is not None:
                    current_date = current_date.tz_localize(None)
                if hasattr(latest_df_date, 'tz') and latest_df_date.tz is not None:
                    latest_df_date = latest_df_date.tz_localize(None)
            
            # If yfinance data is newer than CSV, append new row
                if current_date.date() > latest_df_date.date():
                    new_row = pd.DataFrame({
                        'open': [current_data['open']],
                        'high': [current_data['high']],
                        'low': [current_data['low']],
                        'close': [current_data['current_price']],
                        'volume': [current_data['volume']]
                    }, index=[current_date])
                
                    df = pd.concat([df, new_row])
                    print(f" [✅ Added {current_date.strftime('%Y-%m-%d')}: ${current_data['current_price']:.2f}]", end="")
            
            # If same date, update the last row with latest data
                elif current_date.date() == latest_df_date.date():
                    df.loc[df.index[-1], 'close'] = current_data['current_price']
                    df.loc[df.index[-1], 'high'] = max(df.loc[df.index[-1], 'high'], current_data['high'])
                    df.loc[df.index[-1], 'low'] = min(df.loc[df.index[-1], 'low'], current_data['low'])
                    df.loc[df.index[-1], 'volume'] = current_data['volume']
                    print(f" [✅ Updated {current_date.strftime('%Y-%m-%d')}: ${current_data['current_price']:.2f}]", end="")
            
            # If yfinance data is older, CSV is more current (unusual)
                else:
                    print(f" [⚠️ CSV newer than yfinance, using CSV: ${df['close'].iloc[-1]:.2f}]", end="")
            
                return df
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f" [Retry {attempt+1}]", end="")
                    time.sleep(retry_delay)
                    continue
                else:
                # Use CSV data if all retries fail
                    print(f" [⚠️ Using CSV data: ${df['close'].iloc[-1]:.2f}]", end="")
                    return df
    
        return df


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
        base_threshold = 0.55
        
        # Historical volatility pattern (last 60 days)
        if len(df) >= 60:
            recent_vol = df['close'].pct_change().iloc[-60:].std()
            long_vol = df['close'].pct_change().std()
            vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0
            
            # If recent volatility much higher than normal, be more conservative
            if vol_ratio > 1.5:
                vol_adj = 0.05
            elif vol_ratio > 1.2:
                vol_adj = 0.03
            elif vol_ratio < 0.8:
                vol_adj = -0.02
            else:
                vol_adj = 0.01
        else:
            # Fallback to absolute volatility
            if volatility > 0.04:
                vol_adj = 0.06
            elif volatility > 0.03:
                vol_adj = 0.04
            elif volatility > 0.02:
                vol_adj = 0.02
            else:
                vol_adj = 0.00
        
        # Trend consistency (helps differentiate MIXED from real trends)
        if len(df) >= 20:
            closes = df['close'].iloc[-20:]
            returns = closes.pct_change().dropna()
            
            # Count directional consistency
            positive_days = (returns > 0).sum()
            trend_consistency = abs(positive_days - 10) / 10
            
            if trend_consistency > 0.6:
                regime_adj = -0.02
            elif trend_consistency < 0.3:
                regime_adj = 0.04
            else:
                regime_adj = 0.01
        else:
            regime_adj = 0.01
        
        # Market regime fine-tuning
        if "BULL STRONG" in market_regime:
            regime_adj -= 0.01
        elif "BEAR STRONG" in market_regime:
            regime_adj += 0.02
        elif "MIXED" in market_regime or "SIDEWAYS" in market_regime:
            regime_adj += 0.03
        
        adjusted_threshold = base_threshold + vol_adj + regime_adj
        adjusted_threshold = max(0.50, min(0.72, adjusted_threshold))
        
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
            score = min(100, 50 + (margin * 200))
        else:
            score = max(0, 50 + (margin * 200))
        
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
# END OF PART 1/5
# ============================================================================
# Next: Part 2 will include:
# - EnhancedMarketRegime class
# - ImprovedRiskManagement class
# - WeightedDecisionEngine class
# ============================================================================
"""
Enhanced Stock Predictor v2 - PART 2/5
Market Regime Detection, Risk Management, and Decision Engine
"""

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
# ============================================================================
# WEIGHTED DECISION SCORING WITH RISK ASSESSMENT
# ============================================================================
class WeightedDecisionEngine:
    """Score-based decision making with enhanced risk evaluation"""
    
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
        risk_level = "MODERATE"
        
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
            risk_level = "HIGH"  # Choppy market = high risk
        elif "SIDEWAYS" in regime:
            market_score = 8
            risk_level = "MODERATE-HIGH"
        else:
            market_score = 0
            risk_level = "VERY HIGH"
        
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
            if risk_level == "MODERATE":
                risk_level = "MODERATE-HIGH"
        else:  # Very high
            vol_score = 2
            risk_level = "VERY HIGH"
        
        score += vol_score
        breakdown['volatility'] = vol_score
        
        # Enhanced decision logic with risk consideration
        if score >= 75 and risk_level in ["LOW", "MODERATE"]:
            action = "🟢 STRONG BUY" if "UP" in direction else "🔴 STRONG SELL"
            signal = "EXCELLENT"
            recommendation = "High confidence trade with favorable conditions"
        elif score >= 70 and risk_level == "HIGH":
            action = "⚡ CONSIDER WITH CAUTION"
            signal = "GOOD BUT RISKY"
            recommendation = "Good setup but market conditions are choppy - use tight stops"
        elif score >= 65 and risk_level in ["LOW", "MODERATE"]:
            action = "🟢 BUY" if "UP" in direction else "🔴 SELL"
            signal = "GOOD"
            recommendation = "Solid trade setup with acceptable risk"
        elif score >= 60 and risk_level in ["MODERATE-HIGH", "HIGH"]:
            action = "⏸️ WAIT FOR BETTER ENTRY"
            signal = "MARGINAL"
            recommendation = "Setup has potential but wait for clearer market conditions"
        elif score >= 55:
            action = "⚡ CAUTIOUS - SMALL POSITION"
            signal = "MARGINAL"
            recommendation = "Only for experienced traders with tight risk management"
        elif score >= 45:
            action = "⏸️ WAIT"
            signal = "WEAK"
            recommendation = "Insufficient edge - wait for better opportunity"
        else:
            action = "❌ AVOID TRADE"
            signal = "REJECTED"
            recommendation = "Poor setup - do not trade"
        
        return {
            'score': score,
            'action': action,
            'signal_strength': signal,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'breakdown': breakdown
        }
# ============================================================================
# END OF PART 2/5
# ============================================================================
# Next: Part 3 will include:
# - Technical Indicators (ATR, RSI, ADX)
# - Market Data Fetcher (SPY trend)
# - Feature Engineering
# ============================================================================
"""
Enhanced Stock Predictor v2 - PART 3/5
Technical Indicators, Market Data Fetcher, and Feature Engineering
"""

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
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
# MARKET DATA FETCHER
# ============================================================================
class MarketDataFetcher:
    """Fetch market trend data (SPY)"""
    
    @staticmethod
    def fetch_market_trend(df: pd.DataFrame, retries: int = 3) -> pd.DataFrame:
        """Fetch SPY market trend and merge with stock data"""
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
        
        # Fallback: use stock's own trend
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['market_trend'] = (df['close'] > df['ema_200']).astype(int)
        df = df.drop('ema_200', axis=1)
        return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all technical features for prediction"""
    df = df.copy()
    
    # ATR and Volatility
    df['atr'] = calculate_atr(df)
    df['atr_pct'] = df['atr'] / df['close']
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    # Moving Averages
    df['ma_50'] = df['close'].rolling(50).mean()
    df['trend_strength'] = abs(df['close'] / df['ma_50'] - 1)
    
    # Rate of Change
    df['roc_10'] = df['close'].pct_change(10)
    
    # Volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # SMA
    df['sma_7'] = df['close'].rolling(7).mean()
    df['sma_7'] = (df['close'] - df['sma_7']) / df['sma_7']
    
    # EMA
    ema_7 = df['close'].ewm(span=7, adjust=False).mean()
    df['ema_7'] = (df['close'] - ema_7) / ema_7
    
    # RSI
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['rsi_14'] = df['rsi_14'] / 100
    
    # Volume Trend
    df['volume_ma_7'] = df['volume'].rolling(7).mean()
    df['volume_ma_30'] = df['volume'].rolling(30).mean()
    df['volume_trend_week'] = df['volume_ma_7'] / df['volume_ma_30']
    
    # Weekly metrics
    df['weekly_return'] = df['close'].pct_change(5)
    df['weekly_volatility'] = df['close'].pct_change().rolling(5).std()
    
    # EMA crossover
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_diff'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
    
    # ADX
    df['adx_14'] = calculate_adx(df, period=14)
    df['adx_14'] = df['adx_14'] / 100
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Market trend
    df = MarketDataFetcher.fetch_market_trend(df)
    
    return df


# ============================================================================
# ENHANCED PREDICTION DATA CLASS
# ============================================================================
@dataclass
class EnhancedStockPrediction:
    """Data class for prediction results"""
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
# DATA LOADING WITH REAL-TIME PRICE UPDATE
# ============================================================================
def update_csv_with_latest_data(symbol: str, csv_path: Path) -> pd.DataFrame:
    """
    Auto-update CSV with latest prices from yfinance
    """
    try:
        # Load existing CSV
        df = pd.read_csv(csv_path)
        
        # Standardize columns
        df.columns = df.columns.str.lower()
        
        # Find date column
        date_col = None
        for col in ['date', 'datetime', 'timestamp', 'price']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            date_col = df.columns[0]
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        
        # Get last date in CSV
        last_csv_date = df[date_col].max()
        today = pd.Timestamp.now().normalize()
        
        # Check if update needed
        days_old = (today - last_csv_date).days
        
        if days_old <= 1:
            print(f" [CSV up-to-date: {last_csv_date.strftime('%Y-%m-%d')}]", end="", flush=True)
            # Set date as index and return
            df = df.set_index(date_col)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        
        # Need to update - fetch new data
        print(f" [CSV {days_old}d old, updating...]", end="", flush=True)
        
        import yfinance as yf
        import requests
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        ticker = yf.Ticker(symbol, session=session)
        
        # Fetch data from last CSV date to today
        start_date = last_csv_date + timedelta(days=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_data = ticker.history(start=start_date, interval='1d')
        
        if new_data.empty:
            # No new data (market closed)
            print(f" [No new data available]", end="", flush=True)
            df = df.set_index(date_col)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        
        # Process new data
        new_data = new_data.reset_index()
        new_data.columns = new_data.columns.str.lower()
        
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
        
        # Rename columns to match CSV format
        rename_map = {}
        for col in new_data.columns:
            if 'date' in col.lower() or col.lower() == 'index':
                rename_map[col] = date_col
            elif col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                rename_map[col] = col.lower()
        
        new_data = new_data.rename(columns=rename_map)
        
        # Ensure date column is properly formatted
        new_data[date_col] = pd.to_datetime(new_data[date_col])
        
        # Remove timezone
        if hasattr(new_data[date_col].dtype, 'tz') and new_data[date_col].dtype.tz is not None:
            new_data[date_col] = new_data[date_col].dt.tz_localize(None)
        
        # Keep only columns that exist in original CSV
        cols_to_keep = [col for col in df.columns if col in new_data.columns]
        new_data = new_data[cols_to_keep]
        
        # Combine old and new data
        combined = pd.concat([df, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=[date_col], keep='last')
        combined = combined.sort_values(date_col)
        
        # Save updated CSV
        combined.to_csv(csv_path, index=False)
        
        latest = combined[date_col].max()
        new_rows = len(combined) - len(df)
        print(f" [+{new_rows} rows, saved to {latest.strftime('%Y-%m-%d')}]", end="", flush=True)
        
        # Set date as index for return
        combined = combined.set_index(date_col)
        if combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)
        
        return combined
        
    except Exception as e:
        # If update fails, return original CSV data
        print(f" [Update failed, using CSV: {str(e)[:30]}]", end="", flush=True)
        df = df.set_index(date_col)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df


def load_and_prepare_data(symbol: str):
    """
    Load data from CSV and auto-update with latest prices
    """
    # Try to find CSV file
    csv_paths = [
        Path(f"data/{symbol}.csv"),
        Path(f"data/stock_data/{symbol}.csv"),
        Path(f"stock_data/{symbol}.csv"),
        Path(f"{symbol}.csv"),
    ]
    
    df = None
    csv_found = None
    
    # Find CSV
    for csv_path in csv_paths:
        if csv_path.exists():
            csv_found = csv_path
            print(f" [Found: {csv_path}]", end="", flush=True)
            break
    
    # If no CSV found, try to download from yfinance
    if csv_found is None:
        print(f" [No CSV, downloading from yfinance]", end="", flush=True)
        
        import yfinance as yf
        import requests
        
        try:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            ticker = yf.Ticker(symbol, session=session)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = ticker.history(period='2y', interval='1d')
            
            if df.empty:
                raise ValueError(f"No data from yfinance")
            
            # Save to CSV for future use
            df_to_save = df.reset_index()
            df_to_save.columns = df_to_save.columns.str.lower()
            save_path = Path(f"data/{symbol}.csv")
            save_path.parent.mkdir(exist_ok=True)
            df_to_save.to_csv(save_path, index=False)
            
            print(f" [Downloaded & saved to {save_path}]", end="", flush=True)
            
        except Exception as e:
            raise ValueError(
                f"Cannot load {symbol}:\n"
                f"   - No CSV found\n"
                f"   - yfinance failed: {str(e)}\n"
                f"   Download manually from Yahoo Finance"
            )
    else:
        # Update CSV with latest data
        df = update_csv_with_latest_data(symbol, csv_found)
    
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")
    
    # Standardize columns
    df.columns = df.columns.str.lower()
    
    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Remove timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Sort and clean
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    
    # Get info
    latest_date = df.index[-1]
    current_price = float(df['close'].iloc[-1])
    
    print(f" [✅ {len(df)} rows, {latest_date.strftime('%Y-%m-%d')}, ${current_price:.2f}]", end="")
    
    # Validate columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Convert to numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN
    df = df.dropna(subset=required_cols)
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} rows (need 100+)")
    
    # Create features
    df = create_prediction_features(df)
    
    # Feature list
    feature_cols = [
        'atr_pct', 'volatility', 'trend_strength', 'roc_10', 'volume_ratio',
        'sma_7', 'ema_7', 'rsi_14', 'volume_trend_week',
        'weekly_return', 'weekly_volatility',
        'ema_diff', 'adx_14', 'price_vwap', 'market_trend'
    ]
    
    return df, feature_cols
# END OF PART 3/5
# ============================================================================
# Next: Part 4 will include:
# - Enhanced Prediction Engine (main prediction function)
# - CSV Logging function
# - Display functions (comparative table & detailed analysis)
# ============================================================================
"""
Enhanced Stock Predictor v2 - PART 4/5
Prediction Engine, CSV Logging, and Display Functions
"""

# ============================================================================
# ENHANCED PREDICTION ENGINE
# ============================================================================
def predict_stock_enhanced(symbol: str):
    """
    Enhanced prediction with all improvements
    """
    symbol = symbol.upper()
    
    # Find model
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
    
    # Load model
    tf = get_tf()
    model = load_model_safe(model_path)
    
    # Load and prepare data (with real-time price update)
    df, feature_cols = load_and_prepare_data(symbol)
    
    # Get current values from most recent data
    current_price = float(df['close'].iloc[-1])
    price_date = df.index[-1].strftime('%Y-%m-%d')
    current_atr = float(df['atr'].iloc[-1]) if not pd.isna(df['atr'].iloc[-1]) else 1.0
    current_volatility = float(df['volatility'].iloc[-1]) if not pd.isna(df['volatility'].iloc[-1]) else 0.02
    
    # Enhanced market regime analysis
    regime_analysis = EnhancedMarketRegime.analyze_regime(df)
    market_regime = regime_analysis['regime']
    trend_strength = regime_analysis['trend_strength']
    volatility_regime = regime_analysis['volatility_regime']
    
    # Adaptive threshold per stock
    threshold_info = AdaptiveThresholds.calculate_stock_threshold(
        df, current_volatility, market_regime
    )
    adaptive_threshold = threshold_info['threshold']
    
    # Prepare features for prediction
    from sklearn.preprocessing import RobustScaler
    
    X = df[feature_cols].values.astype(float)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sequence length
    seq_len = 60
    if len(X_scaled) < seq_len:
        seq_len = min(30, len(X_scaled))
    
    if len(X_scaled) < seq_len:
        raise ValueError(f"Insufficient data (need at least {seq_len} rows)")
    
    # Create sequence for prediction
    X_seq = X_scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))
    
    # Make prediction
    predictions = model.predict(X_seq, verbose=0)
    
    # Extract week probability
    week_prob_up = float(predictions[2][0, 0])
    
    # Determine direction
    week_direction = "UP" if week_prob_up > 0.5 else "DOWN"
    week_direction_emoji = "📈 UP" if week_direction == "UP" else "📉 DOWN"
    
    # Calculate confidence with score
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
    
    # Generate reasoning and warnings
    reasoning = []
    warnings_list = []
    
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
        warnings_list.append(f"Risk-reward at {risk_mgmt['risk_reward']:.2f}:1")
    
    # Market alignment
    if ("BULL" in market_regime and week_direction == "UP") or ("BEAR" in market_regime and week_direction == "DOWN"):
        reasoning.append(f"✅ Aligned with {market_regime}")
    elif "CHOPPY" in market_regime:
        warnings_list.append("Choppy market - high risk")
        reasoning.append(f"⚠️ Choppy market detected")
    elif "MIXED" in market_regime or "SIDEWAYS" in market_regime:
        warnings_list.append("Market lacks clear direction")
        reasoning.append(f"⚠️ {market_regime}")
    else:
        warnings_list.append("Signal conflicts with market regime")
        reasoning.append(f"⚠️ Signal vs regime mismatch")
    
    # Volatility check
    if current_volatility > 0.04:
        warnings_list.append(f"Very high volatility ({current_volatility*100:.1f}%)")
    elif current_volatility > 0.03:
        warnings_list.append(f"High volatility ({current_volatility*100:.1f}%)")
    
    # Score breakdown
    reasoning.append(f"📊 Signal Score: {decision['score']:.0f}/100")
    reasoning.append(f"🎯 Risk Level: {decision['risk_level']}")
    reasoning.append(f"💡 Recommendation: {decision['recommendation']}")
    
    # Create and return prediction object
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
        warnings=warnings_list
    )

# ============================================================================
# CSV LOGGING
# ============================================================================
def log_to_csv(predictions: List, filename: str = "predictions_log.csv"):
    """Log predictions to CSV file"""
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
def print_comparative_table(predictions: List):
    """Compact comparative table for multiple stocks"""
    
    print("\n" + "="*165)
    print("📊 STOCK COMPARISON TABLE - ENHANCED ANALYSIS")
    print("="*165)
    
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
    
    for p in predictions:
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
    
    trades = sum(1 for p in predictions if "BUY" in p.action or "SELL" in p.action)
    strong_signals = sum(1 for p in predictions if p.signal_score >= 75)
    rejected = sum(1 for p in predictions if "NO TRADE" in p.action or "AVOID" in p.action)
    up_predictions = sum(1 for p in predictions if "UP" in p.week_direction)
    down_predictions = sum(1 for p in predictions if "DOWN" in p.week_direction)
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total Analyzed: {len(predictions)}")
    print(f"   🟢 Trade Signals: {trades} ({trades/len(predictions)*100:.0f}%)")
    print(f"   ⭐ Strong Signals (Score ≥75): {strong_signals}")
    print(f"   ❌ Rejected: {rejected}")
    print(f"   📊 Direction: {up_predictions} UP / {down_predictions} DOWN")
    print(f"   Average Score: {np.mean([p.signal_score for p in predictions]):.1f}/100")
    print(f"   Average R:R: {np.mean([p.risk_reward for p in predictions]):.2f}:1")
    print(f"   Average Probability: {np.mean([p.week_prob_up for p in predictions]):.1%}")
    
    if trades > 0:
        trade_preds = [p for p in predictions if "BUY" in p.action or "SELL" in p.action]
        best = max(trade_preds, key=lambda x: x.signal_score)
        print(f"\n🏆 BEST OPPORTUNITY: {best.symbol} (Score: {best.signal_score:.0f}, R:R: {best.risk_reward:.2f}:1)")
    
    print("="*165 + "\n")


def print_detailed_analysis(pred):
    """Detailed individual stock analysis with clear recommendation"""
    print("\n" + "="*110)
    print(f"🔍 {pred.symbol} - DETAILED ANALYSIS (Data: {pred.price_date})")
    print("="*110)
    
    print(f"\n{'MARKET SITUATION':─^110}")
    print(f"Current Price:    ${pred.current_price:.2f}")
    print(f"Direction:        {pred.week_direction} (Probability: {pred.week_prob_up:.1%})")
    print(f"Confidence:       {pred.confidence} (Score: {pred.confidence_score:.0f}/100)")
    print(f"Market Regime:    {pred.market_regime}")
    print(f"Volatility:       {pred.volatility*100:.2f}% ({pred.volatility_regime})")
    
    print(f"\n{'TRADE SETUP':─^110}")
    print(f"Entry Price:       ${pred.current_price:.2f}")
    print(f"Target Range:      ${pred.target_low:.2f} - ${pred.target_high:.2f} (+{pred.expected_return:.1f}%)")
    print(f"Stop Loss:         ${pred.stop_loss:.2f} (-{pred.max_loss:.1f}%)")
    print(f"Risk-Reward:       {pred.risk_reward:.2f}:1 {'✅' if pred.risk_reward >= 2.0 else '⚠️'}")
    
    print(f"\n{'DECISION':─^110}")
    print(f"Signal Score:      {pred.signal_score:.0f}/100")
    print(f"Action:            {pred.action}")
    
    recommendation = next((r for r in pred.reasoning if "Recommendation:" in r), None)
    risk_level = next((r for r in pred.reasoning if "Risk Level:" in r), None)
    
    if recommendation:
        print(f"\n{'WHY THIS DECISION?':─^110}")
        print(f"  {recommendation.replace('💡 Recommendation: ', '📌 ')}")
        if risk_level:
            print(f"  {risk_level}")
    
    print(f"\n{'KEY POINTS':─^110}")
    key_points = [r for r in pred.reasoning if not ("Recommendation:" in r or "Risk Level:" in r or "Signal Score:" in r)]
    for reason in key_points:
        print(f"  {reason}")
    
    if pred.warnings:
        print(f"\n⚠️  RISKS TO CONSIDER:")
        for warning in pred.warnings:
            print(f"  • {warning}")
    
    print(f"\n{'VERDICT':─^110}")
    if "STRONG BUY" in pred.action or "BUY" in pred.action:
        print(f"  ✅ PROCEED with this trade - Setup looks favorable")
        print(f"  💰 Potential profit target: ${pred.target_high:.2f}")
        print(f"  🛡️  Protect yourself with stop loss at: ${pred.stop_loss:.2f}")
    elif "CAUTIOUS" in pred.action or "CONSIDER" in pred.action:
        print(f"  ⚡ Trade with CAUTION - Use smaller position size")
        print(f"  📉 Tighter stop loss recommended due to market conditions")
    elif "WAIT" in pred.action:
        print(f"  ⏸️  HOLD OFF - Wait for better market conditions or clearer signals")
    else:
        print(f"  ❌ SKIP this trade - Risk outweighs potential reward")
    
    print("="*110 + "\n")
# ============================================================================
# END OF PART 4/5
# ============================================================================
# ============================================================================
# END OF PART 4/5
# ============================================================================
# Next: Part 5 (FINAL) will include:
# - Main function
# - Command-line argument parsing
# - Entry point
# ============================================================================
"""
Enhanced Stock Predictor v2 - PART 5/5 (FINAL)
Main Function, Argument Parsing, and Entry Point
"""

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main execution function"""
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
  ✅ Real-time price fetching from yfinance
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
    
    # Check setup
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
    
    # Determine stocks to analyze
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
    print(f"   📡 Fetching real-time prices from yfinance (with 3 retries per stock)...")
    print(f"   Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   {'Stock':<8} {'Status':<60} {'Score':>6}")
    
    # Run predictions
    predictions = []
    print("-" * 80)
    for symbol in symbols:
        try:
            print(f"   {symbol:<8}", end="", flush=True)
            pred = predict_stock_enhanced(symbol)
            predictions.append(pred)
            print(f" ✅ Score: {pred.signal_score:.0f}/100")
        except Exception as e:
            import traceback
            print(f" ❌ Error: {str(e)}")
            print(f"\n   DEBUG - Full traceback:")
            traceback.print_exc()
            print()
    print("-" * 80)
    
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


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()

