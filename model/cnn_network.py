import numpy as np
import tensorflow as tf
from typing import Union


class CNNNetwork:
    def __init__(self, grid_size, action_size, learning_rate, *args, **kwargs):
        super(CNNNetwork, self).__init__()
        self.state_size = grid_size
        self.action_size = action_size
        # initialize model with 6 layers
        # input layer: (4, 4, 1) represent a (4,4) board and channel 1
        # layer1 CNN: 32 3x3 kernel, ReLU activation, stride 1, padding 1
        # layer2 CNN: 64 3x3 kernel, ReLU activation, stride 1, padding 1
        # Flatten Layer is used to turn output with shape (4, 4, 64) from layer2 into shape (1024, )
        # layer3 Full Connect: 512 neurons, ReLU activation
        # layer4 Full Connect: 128 neurons, ReLU activation
        # output layer: 4 neurons (action 0,1,2,3), linear activation
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(grid_size, grid_size, 1)),
                tf.keras.layers.Conv2D(
                    32, kernel_size=(3, 3), activation="relu", strides=1, padding="same"
                ),
                tf.keras.layers.Conv2D(
                    64, kernel_size=(3, 3), activation="relu", strides=1, padding="same"
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(action_size, activation="linear"),
            ]
        )
        self.loss = tf.keras.losses.mean_squared_error
        self.optimize = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimize,
            metrics="accuracy",
        )

    def get_weights(self) -> list:
        return self.model.get_weights()

    def set_weights(self, weights: list):
        self.model.set_weights(weights)
