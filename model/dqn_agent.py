from typing import SupportsFloat

from model.cnn_network import CNNNetwork
from model.replay_buffer import ReplayBuffer
from model.hyper_param_config import (
    epsilon_start,
    epsilon_end,
    learning_rate,
    gamma,
    batch_size,
    tau,
    target_update_freq, clip_norm,
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

        # init Q network and target network
        self.q_network = CNNNetwork(self.grid_size, self.action_size, _learning_rate)
        self.target_q_network = CNNNetwork(
            self.grid_size, self.action_size, _learning_rate
        )
        self.q_network.build((None, self.grid_size, self.action_size, 1))
        self.target_q_network.build((None, self.grid_size, self.action_size, 1))
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

    def act(self, state: np.ndarray, random=False, banned_action=None) -> int:
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
            q_sa = self.q_network.call(state.reshape(1, 4, 4, 1))
            sorted_actions = np.argsort(q_sa)[0][
                ::-1
            ]  # sorting by Q value from high to low
            for action in sorted_actions:
                if action in banned_action:
                    continue
                else:
                    return action

        print("Couldn't find valid action, but the game is not terminated.")
        print(f"State: {state}")
        return np.random.choice(
            a=self.action_size, size=1, p=[1 / self.action_size] * self.action_size
        )[0]

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: SupportsFloat,
        next_state: np.ndarray,
        terminated: bool,
    ):
        self.replay_buffer.push(
            np.array(state), action, reward, np.array(next_state), terminated
        )

    def update(self):
        # soft update
        if self.target_update_count == self.target_update_freq:
            self._soft_update()
            self.target_update_count = 0

        self.target_update_count += 1

        # sample experience
        batch_exp = self.replay_buffer.sample(self.batch_size)
        batch_state = np.array([exp[0].reshape(4, 4, 1) for exp in batch_exp])
        batch_action = np.array([exp[1] for exp in batch_exp])
        batch_reward = np.array([exp[2] for exp in batch_exp])
        batch_next_state = np.array([exp[3].reshape(4, 4, 1) for exp in batch_exp])
        # if terminated(game end because board filled up), no next state and future reward should time 0
        batch_terminated = np.array([1 if exp[4] == 0 else 0 for exp in batch_exp])

        with tf.GradientTape() as tape:
            # predicting Q(s,a) for each state and action in experience by using Q network
            q_states_predict = self.q_network.call(batch_state)
            q_states_action = tf.gather_nd(
                q_states_predict,
                tf.stack([tf.range(self.batch_size), batch_action], axis=1),
            )

            # predicting max_a(Q(s')) for each next state in experience by using target network
            target_next_states_predict = self.target_q_network.call(batch_next_state)
            max_target_next_s_q = tf.reduce_max(
                target_next_states_predict, axis=1
            )  # get max_q of each Q(s', a)

            # computing predict Q(s, a) of target network by using bellman expectation equation
            q_target_next_s = (
                batch_reward + self.gamma * max_target_next_s_q * batch_terminated
            )

            # computing loss
            loss = tf.reduce_mean(tf.square(q_states_action - q_target_next_s))

        # train q network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        clipped_gradients = [
            tf.clip_by_norm(g, clip_norm) for g in gradients
        ]  # gradient clipping

        self.q_network._optimizer.apply_gradients(
            zip(clipped_gradients, self.q_network.trainable_variables)
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
