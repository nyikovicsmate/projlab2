import copy
from typing import Tuple, Iterator, List
import tensorflow as tf
import numpy as np

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
        self.global_model = PixelwiseA3CModel(batch_size=self.input_shape[0], input_shape=self.input_shape[1:4])
        self.local_model = PixelwiseA3CModel(batch_size=self.input_shape[0], input_shape=self.input_shape[1:4])
        self.global_model.actor_model.build(input_shape)
        self.global_model.critic_model.build(input_shape)
        self.local_model.actor_model.build(input_shape)
        self.local_model.critic_model.build(input_shape)
        # self.global_model.actor_model.summary()
        # self.global_model.critic_model.summary()

    # @tf.function
    def train(self,
              batch_generator: Iterator,
              epochs: int,
              steps_per_episode: int = 5,
              learning_rate: float = 0.001,
              discount_factor: float = 0.95):
        logger.info("Started training.")
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                                       decay_steps=epochs,
                                                                       decay_rate=0.9)
        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                   beta_1=0.9,
                                                   beta_2=0.999,
                                                   epsilon=1e-8)
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                    beta_1=0.9,
                                                    beta_2=0.999,
                                                    epsilon=1e-8)
        # loss = tf.keras.losses.MeanSquaredError()
        # self.global_model.compile(optimizer=optimizer)
        # self.local_model.compile(optimizer=optimizer)
        for epoch in range(epochs):
            logger.info(f"epoch {epoch}")
            # load global network variables
            self.local_model.actor_model.set_weights(self.global_model.actor_model.get_weights())
            self.local_model.critic_model.set_weights(self.global_model.critic_model.get_weights())

            orig_img_batch, noisy_img_batch = next(batch_generator)
            s_t = State(noisy_img_batch)
            epoch_r = 0
            s = {}  # state
            r = {}  # reward
            V = {}  # expected total rewards from state
            past_action_log_prob = {}
            past_action_entropy = {}
            for t in range(steps_per_episode):
                # logger.info(f"step {t}")
                s[t] = copy.deepcopy(s_t)
                image_batch_nchw = tf.transpose(s_t.image_batch, perm=[0, 2, 3, 1])
                # predict the actions
                a_t = self.local_model.actor_model(image_batch_nchw)
                sampled_a_t = self._sample_tf(a_t)
                a_t = tf.transpose(a_t, perm=[0, 3, 1, 2])
                # predict the rewards
                V_t = self.local_model.critic_model(image_batch_nchw)
                V_t = tf.transpose(V_t, perm=[0, 3, 1, 2])

                past_action_log_prob[t] = self._mylog_prob(tf.math.log(a_t), sampled_a_t)
                past_action_entropy[t] = self._myentropy(a_t, tf.math.log(a_t))
                V[t] = V_t
                # do the predicted best actions
                s_t.step(sampled_a_t)
                r_t = tf.math.square(orig_img_batch - s[t].image_batch) * 255 - tf.math.square(
                        orig_img_batch - s_t.image_batch) * 255
                r[t] = tf.cast(r_t, dtype=tf.float32)
                epoch_r += np.mean(r_t) * np.power(discount_factor, t)
            logger.info(f"epoch reward: {epoch_r}")
            # logger.info(f"updating models")
            # update global model
            # initialize accumulator variables
            R = 0
            actor_loss = 0
            critic_loss = 0
            beta = 1e-2
            # total_loss = 0
            for t in reversed(range(steps_per_episode)):
                R *= discount_factor
                R += r[t]
                A = R - V[t]  # advantage
                # Accumulate gradients of policy
                log_prob = past_action_log_prob[t]
                entropy = past_action_entropy[t]

                # Log probability is increased proportionally to advantage
                ##############################
                actor_loss -= log_prob * A
                ##############################
                # Entropy is maximized
                actor_loss -= beta * entropy
                # Accumulate gradients of value function
                critic_loss += (R - V[t]) ** 2 / 2

                ##########################
                # total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)
                # total_loss = np.mean(actor_loss + np.reshape(critic_loss, actor_loss.shape))
                # total_loss = tf.convert_to_tensor(total_loss, dtype=tf.float32)
                ##########################
            # logger.info(f"total loss: {total_loss}")

            actor_optimizer.minimize(actor_loss, self.local_model.actor_model.trainable_variables)
            # with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            #     actor_tape.watch(self.local_model.actor_model.trainable_variables)
            #     critic_tape.watch(self.local_model.critic_model.trainable_variables)
            #     actor_grads = actor_tape.gradient(actor_loss, self.local_model.actor_model.trainable_variables)
            #     critic_grads = critic_tape.gradient(critic_loss, self.local_model.critic_model.trainable_variables)
            #
            # actor_optimizer.apply_gradients(zip(actor_grads, self.local_model.actor_model.trainable_variables))
            # critic_optimizer.apply_gradients(zip(critic_grads, self.local_model.critic_model.trainable_variables))

            # with tf.GradientTape() as tape:
            #     tape.watch(self.local_model.critic_model.trainable_variables)
            #     grads_critic = tape.gradient(total_loss, self.local_model.critic_model.trainable_variables)
            #     actor_optimizer.apply_gradients(zip(grads_critic, self.local_model.critic_model.trainable_variables))
            #
            # with tf.GradientTape() as tape:
            #     tape.watch(self.local_model.actor_model.variables)
            #     grads_actor = tape.gradient(total_loss, self.local_model.actor_model.trainable_variables)
            #     actor_optimizer.apply_gradients(zip(grads_actor, self.local_model.actor_model.trainable_variables))

            # save back to global model
            self.global_model.actor_model.set_weights(self.local_model.actor_model.get_weights())
            self.global_model.critic_model.set_weights(self.local_model.critic_model.get_weights())

            if epoch % 100 == 0:
                tf.keras.models.save_model(self.global_model.actor_model, "actor_model.tf")
                tf.keras.models.save_model(self.global_model.critic_model, "critic_model.tf")

    @staticmethod
    def _myentropy(prob, log_prob):
        return tf.stack([- tf.math.reduce_sum(prob * log_prob, axis=1)], axis=1)

    @staticmethod
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

    @staticmethod
    def _sample(distribution):
        """
        Samples the image action distribution returned by the last softmax activation.
        :param distribution: output of a softmax activated layer, an array with probability distributions,
        usually shaped (batch_size, channels, widht, height, number_of_actions) NCHW!
        :return: array shaped of (batch_size, channels, widht, height)
        """
        distribution = distribution if type(distribution) == np.ndarray else np.array(distribution)
        flat_dist = np.reshape(distribution, (-1, distribution.shape[-1]))
        flat_indexes = []
        for d in flat_dist:
            sample_value = np.random.choice(d, p=d)
            sample_idx = np.argmax(d == sample_value)
            flat_indexes.append(sample_idx)
        sample_idxs = np.reshape(flat_indexes, distribution.shape[0:-1])
        return sample_idxs

    @staticmethod
    def _sample_tf(distribution):
        # tf.reshape(tf.random.categorical(tf.reshape(tf.math.log(dist), (-1, dist.shape[-1])), 1), (2, 3))
        d = tf.reshape(distribution, (-1, distribution.shape[-1]))
        d = tf.math.log(d)
        d = tf.random.categorical(d, num_samples=1)
        d = tf.reshape(d, distribution.shape[0:-1])
        return d

    # @staticmethod
    # def update_model_variables(src: tf.keras.models.Model, dst: tf.keras.models.Model):
    #     """
    #     :param src: keras model
    #     :param dst: keras model
    #     :return:
    #     """
    #     for src_var, dst_var in zip(src.trainable_variables, dst.trainable_variables):
    #         dst_var = src_var
    #         dst.set_weights(src.get_weights())
