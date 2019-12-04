import copy
from typing import Tuple, Iterator, List
import tensorflow as tf
import numpy as np
from collections import defaultdict

from logging_config import logger
from state import State
from pixelwise_a3c_model import PixelwiseA3CModel


class PixelwiseA3CNetwork:
    """
    Pixelwise asyncron agent network.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int, int]):  # (batch size, width, height, channels)
        self.input_shape = input_shape
        # self.global_model = PixelwiseA3CModel(batch_size=self.input_shape[0], input_shape=self.input_shape[1:4])
        self.local_model = PixelwiseA3CModel(batch_size=self.input_shape[0], input_shape=self.input_shape[1:4])

    def train(self,
              batch_generator: Iterator,
              epochs: int,
              steps_per_episode: int = 4,
              learning_rate: float = 0.001,
              discount_factor: float = 0.95):
        logger.info("Started training.")
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                       decay_steps=epochs,
                                                                       decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-8)
        for epoch in range(epochs):
            logger.info(f"epoch {epoch}")
            # load global network variables
            # self.local_model.actor_model.set_weights(self.global_model.actor_model.get_weights())
            # self.local_model.critic_model.set_weights(self.global_model.critic_model.get_weights())
            orig_img_batch, noisy_img_batch = next(batch_generator)
            s_t0 = noisy_img_batch
            epoch_r = 0
            r = {}  # reward
            V = {}  # expected total rewards from state
            past_action_log_prob = {}
            past_action_entropy = {}
            with tf.GradientTape() as tape:
                for t in range(steps_per_episode):
                    # logger.info(f"step {t}")
                    image_batch_nchw = tf.transpose(s_t0, perm=[0, 2, 3, 1])
                    # predict the actions and values
                    a_t, V_t = self.local_model(image_batch_nchw)
                    # sample the actions
                    sampled_a_t = self._sample_tf(a_t)
                    a_t = tf.transpose(a_t, perm=[0, 3, 1, 2])
                    V_t = tf.transpose(V_t, perm=[0, 3, 1, 2])

                    past_action_log_prob[t] = self._mylog_prob(tf.math.log(a_t), sampled_a_t)
                    past_action_entropy[t] = self._myentropy(a_t, tf.math.log(a_t))
                    V[t] = V_t
                    # update the current state/image with the predicted actions
                    s_t1 = State.update(s_t0, sampled_a_t.numpy())
                    r_t = self._mse(tf.cast(orig_img_batch, tf.float32), tf.cast(s_t0, tf.float32), s_t1)
                    r[t] = tf.cast(r_t, dtype=tf.float32)
                    epoch_r += np.mean(r_t) * np.power(discount_factor, t)

                logger.info(f"epoch reward: {epoch_r}")
                R = 0
                actor_loss = 0
                critic_loss = 0
                beta = 1e-2
                for t in reversed(range(steps_per_episode)):
                    R *= discount_factor
                    R += r[t]
                    A = R - V[t]  # advantage
                    # Accumulate gradients of policy
                    log_prob = past_action_log_prob[t]
                    entropy = past_action_entropy[t]

                    # Log probability is increased proportionally to advantage
                    actor_loss -= log_prob * A
                    # Entropy is maximized
                    actor_loss -= beta * entropy
                    # Accumulate gradients of value function
                    critic_loss += (R - V[t]) ** 2 / 2

                total_loss = tf.reduce_mean(actor_loss + critic_loss)
                if tf.math.is_nan(total_loss):
                    print(total_loss)

                logger.info(f"total loss: {total_loss}")
                actor_grads = tape.gradient(total_loss, self.local_model.trainable_variables)
            optimizer.apply_gradients(zip(actor_grads, self.local_model.trainable_variables))

            if epoch > 0 and epoch % 100 == 0:
                logger.info("Saving model.")
                tf.keras.models.save_model(self.local_model, "model.tf")

    @staticmethod
    @tf.function
    def _mse(a, b, c):
        """
        Calculates the mean squared error for image batches given by the formula:
        mse = (a-b)**2 - (a-c)**2
        :param a:
        :param b:
        :param c:
        :return:
        """
        mse = tf.math.square(a - b) * 255
        mse -= tf.math.square(a - c) * 255
        return mse

    @staticmethod
    @tf.function
    def _myentropy(prob, log_prob):
        return tf.stack([- tf.math.reduce_sum(prob * log_prob, axis=1)], axis=1)

    @staticmethod
    @tf.function
    def _mylog_prob(data, indexes):
        """
        Selects elements from a multidimensional array.
        :param data: image_batch
        :param indexes: indices to select
        :return: the selected indices from data eg.: data=[[11, 2], [3, 4]] indexes=[0,1] --> [11, 4]
        """
        n_batch, n_actions, h, w = data.shape
        p_trans = tf.transpose(data, perm=[0, 2, 3, 1])
        p_trans = tf.reshape(p_trans, [-1, n_actions])
        indexes_flat = tf.reshape(indexes, [-1])
        one_hot_mask = tf.one_hot(indexes_flat, p_trans.shape[1], on_value=True, off_value=False, dtype=tf.bool)
        output = tf.boolean_mask(p_trans, one_hot_mask)
        return tf.reshape(output, (n_batch, 1, h, w))

    # @staticmethod
    # @tf.function
    # def _sample(distribution):
    #     """
    #     Samples the image action distribution returned by the last softmax activation.
    #     :param distribution: output of a softmax activated layer, an array with probability distributions,
    #     usually shaped (batch_size, channels, widht, height, number_of_actions) NCHW!
    #     :return: array shaped of (batch_size, channels, widht, height)
    #     """
    #     distribution = distribution if type(distribution) == np.ndarray else np.array(distribution)
    #     flat_dist = np.reshape(distribution, (-1, distribution.shape[-1]))
    #     flat_indexes = []
    #     for d in flat_dist:
    #         sample_value = np.random.choice(d, p=d)
    #         sample_idx = np.argmax(d == sample_value)
    #         flat_indexes.append(sample_idx)
    #     sample_idxs = np.reshape(flat_indexes, distribution.shape[0:-1])
    #     return sample_idxs

    @staticmethod
    @tf.function
    def _sample_tf(distribution):
        """
        Samples the image action distribution returned by the last softmax activation.
        :param distribution: output of a softmax activated layer, an array with probability distributions,
        usually shaped (batch_size, channels, widht, height, number_of_actions) NCHW!
        :return: array shaped of (batch_size, channels, widht, height)
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.math.log(d)
        d = tf.random.categorical(d, num_samples=1)
        d = tf.reshape(d, distribution.shape[0:-1])
        return d
