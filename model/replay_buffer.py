from model.hyper_param_config import replay_buffer_size
from typing import List, Tuple, Any
import numpy as np
import random


class ReplayBuffer:
    def __init__(self):
        self.capacity = replay_buffer_size
        self.buffer: List[Tuple[Any, Any, Any, Any, int]] = [None] * self.capacity
        self.position = 0
        self.full = False

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # save experience into buffer
        exp = (state, action, reward, next_state, int(done))
        self.buffer[self.position] = exp
        self.position = (self.position + 1) % self.capacity
        if not self.full and self.position == self.capacity - 1:
            self.full = True

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
