from model.cnn_network import CNNNetwork
from model.replay_buffer import ReplayBuffer
from model.hyper_param_config import (
    epsilon_start,
    epsilon_end,
    learning_rate,
    gamma,
    batch_size,
    tau,
    target_update_freq,
)
import numpy as np
import tensorflow as tf


class DQNAgent:
    def __init__(
        self,
        _epsilon_start=epsilon_start,
        _epsilon_end=epsilon_end,
        _learning_rate=learning_rate,
        _gamma=gamma,
        _batch_size=batch_size,
        _tau=tau,
        _target_update_freq=target_update_freq,
    ):
        self.grid_size = 4
        self.action_size = 4
        self.q_network = CNNNetwork(self.grid_size, self.action_size, _learning_rate)
        self.target_q_network = CNNNetwork(
            self.grid_size, self.action_size, _learning_rate
        )
        self.target_q_network.set_weights(self.q_network.get_weights())
        self.replay_buffer = ReplayBuffer()
        self.init_epsilon = _epsilon_start
        self.end_epsilon = _epsilon_end
        self.epsilon = _epsilon_start
        self.gamma = _gamma
        self.batch_size = _batch_size
        self.tau = _tau
        self.target_update_freq = _target_update_freq
        self.target_update_count = 0

    def act(self, state: np.ndarray, random=False) -> int:
        # epsilon greedy policy, epsilon% chance random select action without using Q network.
        random |= np.random.choice(
            a=[False, True], size=1, p=[1 - self.epsilon, self.epsilon]
        )[0]

        if random:
            return np.random.choice(
                a=self.action_size, size=1, p=[1 / self.action_size] * self.action_size
            )[0]
        else:
            # reshape state to (4,4,1)
            q_sa = self.q_network.model.predict(state.reshape(1, 4, 4, 1), verbose=0)
            return np.argmax(q_sa[0])

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        # soft update
        if self.target_update_count == self.target_update_freq:
            self._soft_update()
            self.target_update_count = 0

        self.target_update_count += 1

        # sample experience
        batch_exp = self.replay_buffer.sample(self.batch_size)
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []

        for exp in batch_exp:
            batch_state.append(exp[0].reshape(4, 4, 1))
            batch_action.append(exp[1])
            batch_reward.append(exp[2])
            batch_next_state.append(exp[3].reshape(4, 4, 1))
        batch_state = tf.convert_to_tensor(batch_state)
        batch_next_state = tf.convert_to_tensor(batch_next_state)
        batch_reward = tf.convert_to_tensor(batch_reward)

        with tf.GradientTape() as tape:  # using tape for backwards propagate
            # predicting Q(s,a) for each state and action in experience by using Q network
            q_states_predict = self.q_network.model(batch_state)
            indices = tf.range(tf.shape(q_states_predict)[0])  # index of each exp in the batch [0,1,2,...,batch_size]
            q_states_predict = tf.gather_nd(q_states_predict, tf.stack([indices, batch_action], axis=1))

            # predicting max_a(Q(s')) for each next state in experience by using target network
            target_next_states_predict = self.target_q_network.model(batch_next_state)
            max_target_next_s_q = tf.reduce_max(target_next_states_predict, axis=1)  # get max_q of each Q(s', a)

            # computing predict Q(s, a) of target network by using bellman expectation equation
            q_target_next_s = batch_reward + self.gamma * max_target_next_s_q

            # computing loss
            loss = tf.keras.losses.mse(q_target_next_s, q_states_predict)

        gradients = tape.gradient(loss, self.q_network.model.trainable_variables)
        self.q_network.model.optimizer.apply_gradients(
            zip(gradients, self.q_network.model.trainable_variables)
        )

        return loss

    def _soft_update(self):
        # sof update target-network
        q_weights = self.q_network.get_weights()
        target_weights = self.target_q_network.get_weights()
        updated_weights = []

        for q_w, t_w in zip(q_weights, target_weights):
            updated_t_w = self.tau * q_w + (1 - self.tau) * t_w
            updated_weights.append(updated_t_w)

        self.target_q_network.set_weights(updated_weights)

    def set_decay_epsilon(self, episode, total_episode):
        decay_epsilon = epsilon_start * (1 - episode / total_episode)
        self.epsilon = max(self.end_epsilon, decay_epsilon)
