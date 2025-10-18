# model.py
import math
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .types import ArtemisTensor, ClassificationResult

class DualBranchCNN:
    def __init__(self, hash_dim: int, embed_dim: int, lr: float = 1e-3):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        self.tf, self.keras, self.layers = tf, keras, layers
        self.model = self._build(hash_dim, embed_dim, lr)

    def _conv_block(self, x, filters: int, k: int = 5):
        layers = self.layers
        x = layers.Conv1D(filters, k, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool1D(pool_size=2)(x)
        return x

    def _build(self, hash_dim: int, embed_dim: int, lr: float):
        keras, layers = self.keras, self.layers
        in_a = keras.Input(shape=(hash_dim, 1), name="hash_branch")
        xa = self._conv_block(in_a, 32)
        xa = self._conv_block(xa, 64)
        xa = layers.Flatten()(xa)

        in_b = keras.Input(shape=(embed_dim, 1), name="embed_branch")
        xb = self._conv_block(in_b, 16)
        xb = self._conv_block(xb, 32)
        xb = layers.Flatten()(xb)

        x = layers.Concatenate()([xa, xb])
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=[in_a, in_b], outputs=out, name="DualBranchCNN")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def predict_one(self, tensor: ArtemisTensor) -> ClassificationResult:
        a = tensor.branch_a.reshape(1,-1,1); b = tensor.branch_b.reshape(1,-1,1)
        prob = float(self.model.predict([a,b], verbose=0)[0][0])
        return ClassificationResult(is_malicious=(prob>=0.5), confidence=prob, logits=math.log(prob+1e-9))

    def load_weights(self, path: str) -> None:
        self.model.load_weights(path)

    def save_weights(self, path: str) -> None:
        self.model.save_weights(path)

    def fit(self, A: np.ndarray, B: np.ndarray, y: np.ndarray,
            epochs: int = 10, batch_size: int = 16, val_data: Optional[Tuple[np.ndarray,np.ndarray,np.ndarray]]=None):
        hist = self.model.fit([A,B], y, epochs=epochs, batch_size=batch_size,
                              validation_data=([val_data[0], val_data[1]], val_data[2]) if val_data else None,
                              verbose=2)
        return hist
