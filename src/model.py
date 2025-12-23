from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from config import Config


def build_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """
    Multi-task LSTM:
      - tomorrow_output: class 0/1 (DOWN/UP)
      - week_output: class 0/1
      - tomorrow_return: log-return scalar
      - week_return: log-return scalar
    """
    inputs = Input(shape=input_shape, name="input_seq")

    x = LSTM(Config.LSTM_UNITS_1, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(Config.DROPOUT_1, name="dropout_1")(x)
    x = LSTM(Config.LSTM_UNITS_2, return_sequences=False, name="lstm_2")(x)
    x = Dropout(Config.DROPOUT_2, name="dropout_2")(x)

    x = Dense(Config.DENSE_UNITS_1, activation="relu", name="dense_1")(x)
    x = Dropout(Config.DROPOUT_3, name="dropout_3")(x)
    x = Dense(Config.DENSE_UNITS_2, activation="relu", name="dense_2")(x)

    tomorrow_cls = Dense(2, activation="softmax", name="tomorrow_output")(x)
    week_cls = Dense(2, activation="softmax", name="week_output")(x)
    tomorrow_ret = Dense(1, activation="tanh", name="tomorrow_return")(x)
    week_ret = Dense(1, activation="tanh", name="week_return")(x)

    model = Model(
        inputs=inputs,
        outputs=[tomorrow_cls, week_cls, tomorrow_ret, week_ret],
        name="lstm_stock_model",
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            "tomorrow_output": "sparse_categorical_crossentropy",
            "week_output": "sparse_categorical_crossentropy",
            "tomorrow_return": "mse",
            "week_return": "mse",
        },
        metrics={
            "tomorrow_output": "accuracy",
            "week_output": "accuracy",
        },
    )

    return model
