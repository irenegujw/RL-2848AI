from __future__ import annotations

import math
from typing import SupportsFloat, Any
import gymnasium as gym
import numpy
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from render_game import Renderer
from time import sleep


class GameEnv(gym.Env):
    action_space = None
    observation_space = None
    seed = None
    np_random = None
    score = None
    size = None
    max_illegal_move = None
    illegal_move = None
    max_move = None
    move = None
    render_mode = None
    renderer = None
    terminated = None
    illegal_move_penalty = None
    display_score = None

    def __init__(
        self,
        size=4,
        max_illegal_move=10,
        max_move=10000,
        render_mode="human",
        illegal_move_penalty=-0.5,
    ):
        self.size = size

        self.action_space = Discrete(4)  # 4 actions: up(0) down(1) left(2) right(3)
        # observation_space is more like a statement or definition about size and boundary of status
        # a 4x4 board, each cell contain value 0 - 1
        # calculate from log_2(x)/16 (in a 4x4 board, theoretical maximum is 2^16)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(self.size, self.size), dtype=np.float32
        )

        # initial a random seed，for given seed the experiment is reproducible
        self._seed()

        # set max illegal move, illegal move defined as, no changes to the board after a slide.
        self.max_illegal_move = max_illegal_move

        # set max move
        self.max_move = max_move

        # 4x4 game board
        self.board: numpy.ndarray = np.zeros((self.size, self.size), np.float32)

        # initialize Render object
        self.renderer = Renderer(self.size)

        # initial parameters
        self.score = 0
        self.illegal_move = 0
        self.move = 0
        self.render_mode = render_mode
        self.terminated = False
        self.highest_score = 0
        self.display_score = 0
        self.largest_tail = 0
        self.illegal_move_penalty = illegal_move_penalty

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.max_illegal_move = int(self.move * 0.1)

        # according to the action, move tiles on board
        reward, score, illegal = self._move(action)
        self.move += 1

        # add new random tile
        if not illegal:
            self._add_random_tile()

        # check if game end
        terminated, truncated = self._check_end()

        # construct info
        info = {
            "move": self.move,
            "score": self.score,
            "display_score": self.display_score,
            "step_reward": reward,
            "illegal_move": self.illegal_move,
            "is_illegal_move": illegal,
        }

        self.terminated = terminated | truncated
        self.score += reward
        self.display_score += score

        return self.board.copy(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.board[:] = 0
        self.move = 0
        self.illegal_move = 0
        self.score = 0
        self.display_score = 0
        self.terminated = False

        self._add_random_tile()
        self._add_random_tile()

        return self.board.copy(), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self.renderer.render(
            self.board,
            self.display_score,
            self.illegal_move,
            self.move,
            self.terminated,
        )
        sleep(0.02)
        return None

    def _move(self, action: int, check_mode=False) -> (float, int, bool):
        illegal = True
        reward = 0
        added_score = 0
        is_horizontal_move = action in [2, 3]

        for i in range(self.size):
            # take row if horizon move, column otherwise
            origin_line = self.board[i, :] if is_horizontal_move else self.board[:, i]

            if action in [1, 3]:  # reverse item in the line if swipe down or right
                origin_line = origin_line[::-1]

            combined_line, _reward = self._combine(origin_line)

            reward += _reward
            added_score += int(2 ** (_reward * 16)) if _reward else 0

            if not np.array_equal(origin_line, combined_line):
                # consider legal move if any shifted or combined happened in the row/column
                illegal = False

            if not check_mode:  # check mode is used to check if there is legal action

                if action in [1, 3]:  # reverse item in the line back to origin order
                    combined_line = combined_line[::-1]

                if is_horizontal_move:
                    self.board[i, :] = combined_line
                else:
                    self.board[:, i] = combined_line

        if illegal and not check_mode:
            reward = self.illegal_move_penalty
            self.illegal_move += 1

        return reward, added_score, illegal

    def _combine(self, tiles: np.ndarray) -> (np.ndarray, float):
        reward = 0.0

        # remove all 0 (shift tiles)
        shifted_tiles = tiles[tiles != 0]

        # combine tiles
        combined = False
        for i in range(1, shifted_tiles.shape[0]):
            if combined:
                combined = False
                continue

            if i != 0 and shifted_tiles[i] == shifted_tiles[i - 1]:
                power = int(shifted_tiles[i] * 16) + 1
                shifted_tiles[i - 1] = math.log2(2**power) / 16
                shifted_tiles[i] = 0
                reward += shifted_tiles[i - 1]
                combined = True

        # remove combined tiles
        merged_tiles = shifted_tiles[shifted_tiles != 0]

        # pad the array with zeros
        merged_tiles = np.pad(
            merged_tiles,
            (0, self.size - merged_tiles.shape[0]),
            "constant",
            constant_values=0,
        )

        return merged_tiles, reward

    def _seed(self, seed=None):
        self.np_random, self.seed = seeding.np_random(seed)

    def _add_random_tile(self) -> bool:
        # 90% select 2, 10% select 4
        selected_value = self.np_random.choice(
            a=[math.log2(2) / 16, math.log2(4) / 16], size=1, p=[0.9, 0.1]
        )[0]
        empties = self._get_empties()

        if not empties.shape[0]:
            return False

        selected_empty = self.np_random.choice(a=empties, size=1)[0]
        self.board[selected_empty[0]][selected_empty[1]] = selected_value
        return True

    def _get_empties(self) -> np.ndarray:
        # argwhere return list of coordinates of non-zero item
        # self.board == 0 turn board in into [[ True,  True,  False,  True],...]
        return np.argwhere(self.board == 0)

    def _check_end(self) -> (bool, bool):
        #  truncated indicates that the game ends due to reasons other than the board being filled up.
        truncated = False
        # game terminated because the board being filled up
        game_terminated = True

        if self.move >= self.max_move or self.illegal_move > self.max_illegal_move:
            truncated = True

        for direction in range(4):
            _, _, illegal = self._move(direction, check_mode=True)

            if not illegal:
                game_terminated = False

        return game_terminated, truncated
