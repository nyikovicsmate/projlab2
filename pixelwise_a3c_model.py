from typing import Tuple, Optional, List, Union
import tensorflow as tf


# noinspection DuplicatedCode
class PixelwiseA3CModel:
    actor_model: Optional[tf.keras.models.Model]
    critic_model: Optional[tf.keras.models.Model]

    def __init__(self,
                 batch_size: int,
                 input_shape: Tuple[int, int, int]):
        self.actor_model = None
        self.critic_model = None
        self._init_actor_model(batch_size, input_shape)
        self._init_critic_model(batch_size, input_shape)

    def _init_actor_model(self,
                          batch_size: int,
                          input_shape: Tuple[int, int, int]) -> None:
        inputs = tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        conv1 = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       data_format="channels_last",
                                       use_bias=True,
                                       dilation_rate=1,  # no dilation
                                       activation="relu",
                                       kernel_initializer=None,
                                       bias_initializer=None)
        diconv2 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=2,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv3 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=3,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv4 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=4,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv5 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=3,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv6 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=2,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        conv7 = tf.keras.layers.Conv2D(filters=9,  # number of actions
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       data_format="channels_last",
                                       use_bias=True,
                                       dilation_rate=1,
                                       activation="softmax",
                                       kernel_initializer=None,
                                       bias_initializer=None)
        self.actor_model = tf.keras.models.Sequential(layers=[
            inputs,
            conv1,
            diconv2,
            diconv3,
            diconv4,
            diconv5,
            diconv6,
            conv7
        ])  # output shape (batch_size, 9, width, height)

    def _init_critic_model(self, batch_size: int,
                         input_shape: Tuple[int, int, int]) -> None:
        inputs = tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        conv1 = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       data_format="channels_last",
                                       use_bias=True,
                                       dilation_rate=1,  # no dilation
                                       activation="relu",
                                       kernel_initializer=None,
                                       bias_initializer=None)
        diconv2 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=2,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv3 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=3,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv4 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=4,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv5 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=3,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        diconv6 = tf.keras.layers.Conv2D(filters=64,
                                         kernel_size=3,
                                         strides=1,
                                         padding="same",
                                         data_format="channels_last",
                                         use_bias=True,
                                         dilation_rate=2,
                                         activation="relu",
                                         kernel_initializer=None,
                                         bias_initializer=None)
        conv7 = tf.keras.layers.Conv2D(filters=1,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       data_format="channels_last",
                                       use_bias=True,
                                       dilation_rate=1,
                                       activation="linear",
                                       kernel_initializer=None,
                                       bias_initializer=None)
        self.critic_model = tf.keras.models.Sequential(layers=[
            inputs,
            conv1,
            diconv2,
            diconv3,
            diconv4,
            diconv5,
            diconv6,
            conv7
        ])
