from typing import Dict
from dataclasses import dataclass
import numpy as np
from config import Config

@dataclass
class PredictionResult:
    """Prediction result with all metrics for UI display"""
    symbol: str
    current_price: float
    
    # Tomorrow metrics
    tom_direction: str          # "UP" or "DOWN"
    p_tom_up: float            # Raw probability 0-1
    tom_confidence: float      # Edge over 50%
    tom_pct_change: float
    tom_price: float
    
    # Week metrics
    week_direction: str         # "UP" or "DOWN"
    p_week_up: float           # Raw probability 0-1
    week_confidence: float     # Edge over 50%
    week_pct_change: float
    week_price: float
    
    # Trading signal
    action: str                 # "BUY", "SELL", "HOLD"
    signal_strength: str        # "HIGH", "MEDIUM", "LOW"
    reason: str
    
    # Model performance
    model_tom_acc: float
    model_week_acc: float

class RealisticConfidence:
    """Convert raw model probabilities to realistic trading edges"""
    
    @staticmethod
    def calibrate_edge(raw_prob: float, val_accuracy: float) -> float:
        """Calibrate raw model probability to realistic edge."""
        raw_edge = abs(raw_prob - 0.5) * 2
        return raw_edge * val_accuracy
    
    @staticmethod
    def get_confidence_level(edge: float) -> str:
        """Convert edge to qualitative confidence"""
        if edge >= 0.15:
            return "HIGH"
        elif edge >= 0.08:
            return "MEDIUM"
        else:
            return "LOW"

def make_trading_decision(
    prob_tomorrow_up: float,
    prob_week_up: float,
    log_ret_tomorrow: float,
    log_ret_week: float,
    current_price: float,
    val_acc_tomorrow: float,
    val_acc_week: float,
) -> PredictionResult:
    """Professional trading decision with realistic confidence"""
    
    # Calculate realistic edges
    tom_edge = RealisticConfidence.calibrate_edge(prob_tomorrow_up, val_acc_tomorrow)
    week_edge = RealisticConfidence.calibrate_edge(prob_week_up, val_acc_week)
    
    # Convert log returns to %
    tomorrow_pct = (np.exp(log_ret_tomorrow) - 1) * 100
    week_pct = (np.exp(log_ret_week) - 1) * 100
    
    # Predicted prices
    tomorrow_price = current_price * np.exp(log_ret_tomorrow)
    week_price = current_price * np.exp(log_ret_week)
    
    # Directions (based on predicted returns, not just probabilities)
    tomorrow_dir = "UP" if log_ret_tomorrow > 0 else "DOWN"
    week_dir = "UP" if log_ret_week > 0 else "DOWN"
    
    # Trading decision (week-dominant with realistic thresholds)
    week_conf_level = RealisticConfidence.get_confidence_level(week_edge)
    
    # Decision logic: Use weekly signal with confidence thresholds
    if prob_week_up >= 0.55:  # Strong bullish
        action = "BUY"
    elif prob_week_up <= 0.45:  # Strong bearish
        action = "SELL"
    else:  # Uncertain zone
        action = "HOLD"
    
    reason = (
        f"{week_conf_level} weekly {week_dir} signal "
        f"({prob_week_up:.1%} probability). "
        f"Model accuracy: {val_acc_week:.0%}"
    )
    
    return PredictionResult(
        symbol="AAPL",  # Will be set by caller
        current_price=current_price,
        
        tom_direction=tomorrow_dir,
        p_tom_up=prob_tomorrow_up,
        tom_confidence=tom_edge,
        tom_pct_change=tomorrow_pct,
        tom_price=tomorrow_price,
        
        week_direction=week_dir,
        p_week_up=prob_week_up,
        week_confidence=week_edge,
        week_pct_change=week_pct,
        week_price=week_price,
        
        action=action,
        signal_strength=week_conf_level,
        reason=reason,
        
        model_tom_acc=val_acc_tomorrow,
        model_week_acc=val_acc_week,
    )

def result_to_dict(result: PredictionResult) -> Dict:
    """Format for command-line output"""
    tom_conf_level = RealisticConfidence.get_confidence_level(result.tom_confidence)
    week_conf_level = RealisticConfidence.get_confidence_level(result.week_confidence)
    
    return {
        "symbol": result.symbol,
        "current_price": f"${result.current_price:.2f}",
        "prediction": f"""
{'='*80}
📈 {result.symbol} PROFESSIONAL TRADING SIGNAL
📅 Tomorrow: {result.tom_direction} ({tom_conf_level}) | P(UP)={result.p_tom_up:.1%}
📈 Week:     {result.week_direction} ({week_conf_level}) | P(UP)={result.p_week_up:.1%}
🎯 ACTION:   {result.action}
📊 EDGE:     {result.week_confidence*100:.1f}% from neutral (50%)
💰 PRICE:    ${result.current_price:.2f}

📈 MODEL PERFORMANCE (Historical Validation)
   Tomorrow Direction: {result.model_tom_acc:.0%} accuracy
   Weekly Direction:   {result.model_week_acc:.0%} accuracy

📏 THRESHOLDS USED:
   UP signal:     ≥ 55%
   DOWN signal:   ≤ 45%
   HIGH strength: ≥ 15% edge
   MEDIUM:        8-15% edge
{'='*80}
✅ Prediction complete.
""",
    }