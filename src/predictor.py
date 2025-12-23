from typing import Dict
import numpy as np
import tensorflow as tf
from pathlib import Path

from config import Config
from src.data_loader import load_stock_data, get_current_price  # ✅ NEW IMPORT
from src.feature_engineer import (
    create_technical_indicators,
    create_targets,
    build_feature_matrix,
    make_sequences,
)
from src.decision_engine import make_trading_decision, PredictionResult, result_to_dict

# Global validation accuracies (set by training)
_VAL_ACC_TOMORROW: float = 0.597
_VAL_ACC_WEEK: float = 0.674

def set_validation_accuracies(val_tom: float, val_week: float):
    global _VAL_ACC_TOMORROW, _VAL_ACC_WEEK
    _VAL_ACC_TOMORROW = val_tom
    _VAL_ACC_WEEK = val_week

def _load_model():
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {Config.MODEL_PATH}. Run: python train.py")
    return tf.keras.models.load_model(Config.MODEL_PATH)

def _latest_sequence_for_symbol(symbol: str):
    """Build latest 60-day sequence"""
    df = load_stock_data(symbol, refresh=True)  # ✅ Force refresh to get latest data
    df = create_technical_indicators(df)
    df = create_targets(df)
    
    X_scaled, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, _ = build_feature_matrix(df)
    X_seq, _, _, _, _ = make_sequences(
        X_scaled, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, 
        seq_len=Config.SEQUENCE_LENGTH,
    )
    
    if len(X_seq) == 0:
        raise ValueError(f"Insufficient data for {symbol} (need 60+ days)")
    
    return X_seq[-1:]  # Shape: (1, 60, features)

def predict_for_symbol(symbol: str) -> Dict:
    """Main prediction with real-time price"""
    symbol = symbol.upper()
    if symbol not in Config.SUPPORTED_STOCKS:
        raise ValueError(f"{symbol} not supported. Use: {Config.SUPPORTED_STOCKS}")
    
    print(f"🔄 Loading model for {symbol}...")
    model = _load_model()
    
    print(f"📊 Building latest 60-day sequence...")
    X_last = _latest_sequence_for_symbol(symbol)
    
    print("🤖 Running prediction...")
    predictions = model.predict(X_last, verbose=0)
    
    # Extract predictions (handle both list and array outputs)
    tomorrow_cls = float(predictions[0][0, 0])  # Probability of UP
    week_cls = float(predictions[1][0, 0])      # Probability of UP
    tomorrow_ret = float(predictions[2][0, 0])  # Log return
    week_ret = float(predictions[3][0, 0])      # Log return
    
    # ✅ FIXED: Get real-time current price
    current_price = get_current_price(symbol)
    
    print(f"📊 Raw Model Probabilities:")
    print(f"   P(Tomorrow UP): {tomorrow_cls:.1%}")
    print(f"   P(Week UP):     {week_cls:.1%}")
    print(f"   Current Price:  ${current_price:.2f}")
    
    result = make_trading_decision(
        prob_tomorrow_up=tomorrow_cls,
        prob_week_up=week_cls,
        log_ret_tomorrow=tomorrow_ret,
        log_ret_week=week_ret,
        current_price=current_price,
        val_acc_tomorrow=_VAL_ACC_TOMORROW,
        val_acc_week=_VAL_ACC_WEEK,
    )
    result.symbol = symbol
    
    return result_to_dict(result)

# ✅ NEW: UI-friendly prediction function for Streamlit
def ui_predict_for_symbol(symbol: str) -> PredictionResult:
    """
    Streamlit-friendly version that returns PredictionResult object
    instead of formatted dict
    """
    symbol = symbol.upper()
    if symbol not in Config.SUPPORTED_STOCKS:
        raise ValueError(f"{symbol} not supported")
    
    model = _load_model()
    X_last = _latest_sequence_for_symbol(symbol)
    predictions = model.predict(X_last, verbose=0)
    
    tomorrow_cls = float(predictions[0][0, 0])
    week_cls = float(predictions[1][0, 0])
    tomorrow_ret = float(predictions[2][0, 0])
    week_ret = float(predictions[3][0, 0])
    
    current_price = get_current_price(symbol)
    
    result = make_trading_decision(
        prob_tomorrow_up=tomorrow_cls,
        prob_week_up=week_cls,
        log_ret_tomorrow=tomorrow_ret,
        log_ret_week=week_ret,
        current_price=current_price,
        val_acc_tomorrow=_VAL_ACC_TOMORROW,
        val_acc_week=_VAL_ACC_WEEK,
    )
    result.symbol = symbol
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = predict_for_symbol(sys.argv[1])
        print(result["prediction"])