"""
FIXED MODEL - FOCAL LOSS + CLASS WEIGHTS
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from config import Config

# FOCAL LOSS (FIX #4) - Forces hard examples
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -alpha_t * tf.pow(1. - pt, gamma) * tf.log(pt)
    return focal_loss_fixed

def build_multi_task_model(input_shape=(30, 20)) -> Model:
    inputs = Input(shape=input_shape, name='sequence_input')
    
    # LSTM layers
    lstm1 = LSTM(Config.LSTM_UNITS_1, return_sequences=True, dropout=Config.DROPOUT_LSTM_1)(inputs)
    bn1 = BatchNormalization()(lstm1)
    lstm2 = LSTM(Config.LSTM_UNITS_2, dropout=Config.DROPOUT_LSTM_2)(bn1)
    bn2 = BatchNormalization()(lstm2)
    
    # Dense layers
    dense1 = Dense(Config.DENSE_UNITS_1, activation='relu', 
                   kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(bn2)
    dense1 = Dropout(Config.DROPOUT_DENSE_1)(dense1)
    
    dense2 = Dense(Config.DENSE_UNITS_2, activation='relu', 
                   kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION))(dense1)
    dense2 = Dropout(Config.DROPOUT_DENSE_2)(dense2)
    
    dense3 = Dense(32, activation='relu')(dense2)
    dense3 = Dropout(Config.DROPOUT_DENSE_3)(dense3)
    
    # Outputs
    tomorrow_dir = Dense(1, activation='sigmoid', name='tomorrow_output')(dense3)
    week_dir = Dense(1, activation='sigmoid', name='week_output')(dense3)
    tomorrow_ret = Dense(1, name='tomorrow_return')(dense3)
    week_ret = Dense(1, name='week_return')(dense3)
    
    model = Model(inputs=inputs, outputs=[tomorrow_dir, week_dir, tomorrow_ret, week_ret])
    
    optimizer = Adam(learning_rate=Config.LEARNING_RATE, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'tomorrow_output': focal_loss(gamma=2.0),
            'week_output': focal_loss(gamma=2.0),
            'tomorrow_return': 'mse',
            'week_return': 'mse'
        },
        loss_weights=Config.LOSS_WEIGHTS,
        metrics={'tomorrow_output': 'accuracy', 'week_output': 'accuracy'}
    )
    
    print("\n" + "="*70)
    print("✅ FIXED MODEL - FOCAL LOSS + STRONG MOVES + NO COVID")
    print("🎯 TARGET: 59-60% / 69-70% with PROPER METRICS")
    print(f"📊 Focal Loss (γ=2.0) | Class Weights: {Config.CLASS_WEIGHTS}")
    print("="*70)
    model.summary()
    print("="*70 + "\n")
    
    return model
