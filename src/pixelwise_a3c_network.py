import pathlib
import cv2
import tensorflow as tf
import numpy as np
from typing import Tuple, Iterator
from src import *
from pixelwise_a3c_model import PixelwiseA3CModel
from state import State


# noinspection PyPep8Naming
class PixelwiseA3CNetwork:
    """
    Pixelwise asyncron agent network.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int, int]):  # (batch size, width, height, channels)
        self.input_shape = input_shape
        self.local_model = PixelwiseA3CModel(batch_size=self.input_shape[0], input_shape=self.input_shape[1:4])

    def train(self,
              batch_generator: Iterator,
              episodes: int,
              steps_per_episode: int = 5,
              learning_rate: float = 0.001,
              discount_factor: float = 0.95,
              resume_training: bool = False):
        model_dir = pathlib.Path.joinpath(PROJECT_ROOT, "model")
        model_file = pathlib.Path.joinpath(model_dir, "checkpoint")
        model_episodes_file = pathlib.Path.joinpath(model_dir, "episodes.txt")
        episodes_elapsed = 0
        learning_rate_decay_rate = 0.9
        # resume a previously interrupted training
        if resume_training and pathlib.Path.exists(model_file):
            self.local_model.load_weights(model_file)
            # noinspection PyTypeChecker
            with open(model_episodes_file, "r") as f:
                episodes_elapsed = int(f.readline())
            logger.info(f"Resuming training.")
            logger.info(f"Using model file {model_file}")
            learning_rate *= episodes_elapsed * learning_rate_decay_rate
            episodes -= episodes_elapsed
            if episodes_elapsed >= episodes:
                logger.warning(f"Training episode count exceeds originally intended: {episodes_elapsed} vs. {episodes}")
                logger.warning("Stopping training.")
                return
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                       decay_steps=episodes,
                                                                       decay_rate=learning_rate_decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999,
                                             epsilon=1e-8)
        for episode in range(episodes_elapsed, episodes):
            logger.info(f"episode {episode}")
            orig_img_batch, noisy_img_batch = next(batch_generator)
            if len(orig_img_batch) == 0 or len(noisy_img_batch) == 0:
                logger.warning("Generator did not yield any original or noisiy (or both) image, stopping training.")
                return
            s_t0 = noisy_img_batch
            episode_r = 0
            r = {}  # reward
            V = {}  # expected total rewards from state
            past_action_log_prob = {}
            past_action_entropy = {}
            with tf.GradientTape() as tape:
                for t in range(steps_per_episode):
                    image_batch_nchw = tf.transpose(s_t0, perm=[0, 2, 3, 1])
                    # predict the actions and values
                    a_t, V_t = self.local_model(image_batch_nchw)
                    # sample the actions
                    sampled_a_t = self._sample_random(a_t)
                    # clip distribution into range to avoid 0 values, which cause problem with calculating logarithm
                    a_t = tf.clip_by_value(a_t, 1e-6, 1)
                    a_t = tf.transpose(a_t, perm=[0, 3, 1, 2])
                    V_t = tf.transpose(V_t, perm=[0, 3, 1, 2])

                    past_action_log_prob[t] = self._mylog_prob(tf.math.log(a_t), sampled_a_t)
                    past_action_entropy[t] = self._myentropy(a_t, tf.math.log(a_t))
                    V[t] = V_t
                    # update the current state/image with the predicted actions
                    s_t1 = State.update(s_t0, sampled_a_t.numpy())
                    r_t = self._mse(tf.cast(orig_img_batch, tf.float32), tf.cast(s_t0, tf.float32), s_t1)
                    r[t] = tf.cast(r_t, dtype=tf.float32)
                    s_t0 = s_t1
                    episode_r += np.mean(r_t) * np.power(discount_factor, t)

                logger.info(f"episode reward: {episode_r}")
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
                    actor_loss *= 0.5  # multiply loss by 0.5 coefficient
                    # Accumulate gradients of value function
                    critic_loss += (R - V[t]) ** 2 / 2

                total_loss = tf.reduce_mean(actor_loss + critic_loss)

                logger.info(f"total loss: {total_loss}")
                actor_grads = tape.gradient(total_loss, self.local_model.trainable_variables)
            optimizer.apply_gradients(zip(actor_grads, self.local_model.trainable_variables))

            if episode > 0 and (episode + 1) % 50 == 0:
                logger.info(f"Saving model after {episode + 1} episodes.")
                self.local_model.save_weights(model_file, overwrite=True, save_format="tf")
                # noinspection PyTypeChecker
                with open(model_episodes_file, "w") as f:
                    f.write(str(episode + 1))

    def predict(self,
                batch_generator: Iterator,
                steps_per_episode: int = 5):
        predictions_dir = pathlib.Path.joinpath(PROJECT_ROOT, "predictions")
        if not pathlib.Path.exists(predictions_dir):
            pathlib.Path.mkdir(predictions_dir)
        model_dir = pathlib.Path.joinpath(PROJECT_ROOT, "model")
        model_file = pathlib.Path.joinpath(model_dir, "checkpoint")
        self.local_model.load_weights(model_file)

        orig_img_batch, noisy_img_batch = next(batch_generator)
        s_t0 = noisy_img_batch
        for t in range(steps_per_episode):
            image_batch_nchw = tf.transpose(s_t0, perm=[0, 2, 3, 1])
            # predict the actions and values
            a_t, _ = self.local_model(image_batch_nchw)
            # sample the actions
            sampled_a_t = self._sample_most_probable(a_t)
            # update the current state/image with the predicted actions
            s_t1 = State.update(s_t0, sampled_a_t.numpy())
            s_t0 = s_t1
        # write out images
        orig_image_batch_nchw = tf.transpose(orig_img_batch, perm=[0, 2, 3, 1])
        predicted_image_batch_nchw = tf.transpose(s_t0, perm=[0, 2, 3, 1])
        for i in range(orig_image_batch_nchw.shape[0]):
            img_o = np.squeeze(orig_image_batch_nchw[i], axis=2) * 255
            img_p = np.squeeze(predicted_image_batch_nchw[i], axis=2) * 255
            orig_img_path = pathlib.Path.joinpath(predictions_dir, f"{i}_o.jpg")
            predicted_img_path = pathlib.Path.joinpath(predictions_dir, f"{i}_p.jpg")
            cv2.imwrite(orig_img_path, img_o)
            cv2.imwrite(predicted_img_path, img_p)

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
        Selectts elements from a multidimensional array.
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
    def _sample_random(distribution):
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

    @staticmethod
    @tf.function
    def _sample_most_probable(distribution):
        """
        Returns the most probable action index based on the distribution returned by the last softmax activation.
        :param distribution: output of a softmax activated layer, an array with probability distributions,
        usually shaped (batch_size, channels, widht, height, number_of_actions) NCHW!
        :return: array shaped of (batch_size, channels, widht, height)
        """
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.argmax(d, axis=1)
        d = tf.reshape(d, distribution.shape[0:-1])
        return d
