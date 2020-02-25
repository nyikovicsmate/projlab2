import tensorflow as tf
from typing import Tuple


# noinspection DuplicatedCode
class PixelwiseA3CModel(tf.keras.models.Model):

    def __init__(self,
                 batch_size: int,
                 input_shape: Tuple[int, int, int]):
        super().__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=3,
                                            strides=1,
                                            padding="same",
                                            data_format="channels_last",
                                            use_bias=True,
                                            dilation_rate=1,  # no dilation
                                            activation="relu",
                                            kernel_initializer=None,
                                            bias_initializer=None)
        self.diconv2 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=2,
                                              activation="relu",
                                              kernel_initializer=None,
                                              bias_initializer=None)
        self.diconv3 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=3,
                                              activation="relu",
                                              kernel_initializer=None,
                                              bias_initializer=None)
        self.diconv4 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="same",
                                              data_format="channels_last",
                                              use_bias=True,
                                              dilation_rate=4,
                                              activation="relu",
                                              kernel_initializer=None,
                                              bias_initializer=None)
        self.actor_diconv5 = tf.keras.layers.Conv2D(filters=64,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding="same",
                                                    data_format="channels_last",
                                                    use_bias=True,
                                                    dilation_rate=3,
                                                    activation="relu",
                                                    kernel_initializer=None,
                                                    bias_initializer=None)
        self.actor_diconv6 = tf.keras.layers.Conv2D(filters=64,
                                                    kernel_size=3,
                                                    strides=1,
                                                    padding="same",
                                                    data_format="channels_last",
                                                    use_bias=True,
                                                    dilation_rate=2,
                                                    activation="relu",
                                                    kernel_initializer=None,
                                                    bias_initializer=None)
        self.actor_conv7 = tf.keras.layers.Conv2D(filters=9,  # number of actions
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding="same",
                                                  data_format="channels_last",
                                                  use_bias=True,
                                                  dilation_rate=1,
                                                  activation="softmax",
                                                  kernel_initializer=None,
                                                  bias_initializer=None)
        self.critic_diconv5 = tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=3,
                                                     strides=1,
                                                     padding="same",
                                                     data_format="channels_last",
                                                     use_bias=True,
                                                     dilation_rate=3,
                                                     activation="relu",
                                                     kernel_initializer=None,
                                                     bias_initializer=None)
        self.critic_diconv6 = tf.keras.layers.Conv2D(filters=64,
                                                     kernel_size=3,
                                                     strides=1,
                                                     padding="same",
                                                     data_format="channels_last",
                                                     use_bias=True,
                                                     dilation_rate=2,
                                                     activation="relu",
                                                     kernel_initializer=None,
                                                     bias_initializer=None)
        self.critic_conv7 = tf.keras.layers.Conv2D(filters=1,
                                                   kernel_size=3,
                                                   strides=1,
                                                   padding="same",
                                                   data_format="channels_last",
                                                   use_bias=True,
                                                   dilation_rate=1,
                                                   activation="linear",
                                                   kernel_initializer=None,
                                                   bias_initializer=None)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.inputs(inputs)
        x = self.conv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        actor = self.actor_diconv5(x)
        actor = self.actor_diconv6(actor)
        actor = self.actor_conv7(actor)  # output shape (batch_size, 9, width, height)
        critic = self.critic_diconv5(x)
        critic = self.critic_diconv6(critic)
        critic = self.critic_conv7(critic)  # output shape (batch_size, 1, width, height)
        return actor, critic
