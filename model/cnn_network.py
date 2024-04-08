import numpy as np
import tensorflow as tf
from typing import Union


class CNNNetwork(tf.keras.Model):
    def __init__(self, grid_size, action_size, learning_rate, *args, **kwargs):
        super(CNNNetwork, self).__init__()
        # initialize model with 6 layers
        # input layer: (4, 4, 1) represent a (4,4) board and channel 1
        # layer1 CNN: 32 3x3 kernel, ReLU activation, stride 1, padding 1
        # layer2 CNN: 64 3x3 kernel, ReLU activation, stride 1, padding 1
        # Flatten Layer is used to turn output with shape (4, 4, 64) from layer2 into shape (1024, )
        # layer3 Full Connect: 512 neurons, ReLU activation
        # layer4 Full Connect: 128 neurons, ReLU activation
        # output layer: 4 neurons (action 0,1,2,3), linear activation
        self.conv_1 = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            input_shape=(grid_size, grid_size, 1),
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", strides=1, padding="same"
        )
        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(512, activation="relu")
        self.fc_2 = tf.keras.layers.Dense(128, activation="relu")
        self.fc_3 = tf.keras.layers.Dense(action_size, activation="linear")
        self._loss = tf.keras.losses.mean_squared_error
        self._optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        self.compile(optimizer=self._optimizer, loss=self._loss)

    def call(self, inputs, training=None, mask=None):
        conv1_output = self.conv_1(inputs)
        conv2_output = self.conv_2(conv1_output)
        flat_output = self.flatten(conv2_output)
        fc1_output = self.fc_1(flat_output)
        fc2_output = self.fc_2(fc1_output)
        return self.fc_3(fc2_output)
