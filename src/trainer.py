"""
Multi-Task LSTM Trainer - Production Ready
15+ years → 60-day sequences → 4 outputs (tomorrow/week direction + returns)
Realistic confidence calibration
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config import Config
from src.data_loader import load_stock_data
from src.feature_engineer import (
    create_technical_indicators, 
    create_targets, 
    build_feature_matrix, 
    make_sequences
)
from src.model_builder import build_multi_task_model

# Global validation accuracies (shared with predictor)
_VAL_ACC_TOMORROW: float = 0.55
_VAL_ACC_WEEK: float = 0.77

def set_validation_accuracies(val_tom: float, val_week: float):
    """Set global validation accuracies for predictor use."""
    global _VAL_ACC_TOMORROW, _VAL_ACC_WEEK
    _VAL_ACC_TOMORROW = val_tom
    _VAL_ACC_WEEK = val_week

def build_dataset_for_symbols(symbols: list) -> tuple:
    """Build combined dataset from multiple symbols."""
    print("📊 Building dataset...")
    
    all_X, all_y_tom_dir, all_y_week_dir, all_y_tom_ret, all_y_week_ret = [], [], [], [], []
    
    for i, symbol in enumerate(symbols):
        print(f"Processing {symbol}... ({i+1}/{len(symbols)})")
        try:
            df = load_stock_data(symbol)
            df = create_technical_indicators(df)
            df = create_targets(df)
            
            X, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret, scaler = build_feature_matrix(df)
            
            all_X.append(X)
            all_y_tom_dir.append(y_tom_dir)
            all_y_week_dir.append(y_week_dir)
            all_y_tom_ret.append(y_tom_ret)
            all_y_week_ret.append(y_week_ret)
            
            print(f"  → {len(X):,} samples")
            
        except Exception as e:
            print(f"  ❌ Skipping {symbol}: {e}")
            continue
    
    if not all_X:
        raise RuntimeError("❌ No valid data for any symbol. Check EODHD API key.")
    
    # Concatenate all symbols
    X = np.concatenate(all_X, axis=0)
    y_tom_dir = np.concatenate(all_y_tom_dir, axis=0)
    y_week_dir = np.concatenate(all_y_week_dir, axis=0)
    y_tom_ret = np.concatenate(all_y_tom_ret, axis=0)
    y_week_ret = np.concatenate(all_y_week_ret, axis=0)
    
    print(f"✅ Dataset built: {len(X):,} samples, {X.shape[1]} features")
    return X, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret

def train_and_save_model() -> tuple[float, float]:
    """Complete training pipeline with proper data splitting."""
    print("🔄 Loading 15+ years of data...")
    X, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret = build_dataset_for_symbols(
        Config.SUPPORTED_STOCKS
    )
    
    print("🔄 Creating 60-day sequences...")
    X_seq, y_tom_dir_seq, y_week_dir_seq, y_tom_ret_seq, y_week_ret_seq = make_sequences(
        X, y_tom_dir, y_week_dir, y_tom_ret, y_week_ret,
        seq_len=Config.SEQUENCE_LENGTH
    )
    
    print(f"📊 Sequences: {len(X_seq):,} (shape: {X_seq.shape})")
    
    # FIXED: Proper train/validation split for ALL targets
    split_idx = int(len(X_seq) * (1 - Config.VALIDATION_SPLIT))
    
    X_train = X_seq[:split_idx]
    X_val = X_seq[split_idx:]
    
    y_tom_dir_train = y_tom_dir_seq[:split_idx]
    y_tom_dir_val = y_tom_dir_seq[split_idx:]
    
    y_week_dir_train = y_week_dir_seq[:split_idx]
    y_week_dir_val = y_week_dir_seq[split_idx:]
    
    y_tom_ret_train = y_tom_ret_seq[:split_idx]
    y_tom_ret_val = y_tom_ret_seq[split_idx:]
    
    y_week_ret_train = y_week_ret_seq[:split_idx]
    y_week_ret_val = y_week_ret_seq[split_idx:]
    
    print(f"📈 Training: {len(X_train):,} | Validation: {len(X_val):,}")
    
    print("🏗️ Building multi-task LSTM...")
    # FIXED: Pass exact input shape
    model = build_multi_task_model((Config.SEQUENCE_LENGTH, X.shape[1]))
    
    print("🚀 Starting training...")
    history = model.fit(
        X_train,
        [y_tom_dir_train, y_week_dir_train, y_tom_ret_train, y_week_ret_train],
        validation_data=(
            X_val, 
            [y_tom_dir_val, y_week_dir_val, y_tom_ret_val, y_week_ret_val]
        ),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7,
                verbose=1
            )
        ],
        verbose=1
    )
    
    print("\n📊 Calculating final validation accuracy...")
    # Predict on validation set
    val_predictions = model.predict(X_val, verbose=0)
    
    tom_dir_pred = (val_predictions[0] > 0.5).astype(int).flatten()
    week_dir_pred = (val_predictions[1] > 0.5).astype(int).flatten()
    
    val_tom_acc = accuracy_score(y_tom_dir_val, tom_dir_pred)
    val_week_acc = accuracy_score(y_week_dir_val, week_dir_pred)
    
    print(f"✅ Final Validation Accuracy:")
    print(f"   📅 Tomorrow Direction: {val_tom_acc:.1%}")
    print(f"   📈 1-Week Direction:  {val_week_acc:.1%}")
    
    # Save model
    Config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(Config.MODEL_PATH)
    
    print(f"💾 Model saved: {Config.MODEL_PATH.absolute()}")
    
    # Update global accuracies for predictor
    set_validation_accuracies(val_tom_acc, val_week_acc)
    
    return val_tom_acc, val_week_acc

if __name__ == "__main__":
    train_and_save_model()
