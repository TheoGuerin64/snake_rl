# pyright: reportExplicitAny=false

from typing import Any, cast, final, override

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake.game import (
    Direction,
    ReverseDirectionError,
    SelfCollisionError,
    SnakeGame,
    WallCollisionError,
)
from snake.render import SnakeRenderer


def direction_from_action(action: int) -> Direction:
    match action:
        case 0:
            return Direction.UP
        case 1:
            return Direction.DOWN
        case 2:
            return Direction.LEFT
        case 3:
            return Direction.RIGHT
        case _:
            raise ValueError("Invalid action")


@final
class SnakeEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size: int = 10) -> None:
        """
        Raises:
            ValueError: If `grid_size` is less than 3.
        """

        super().__init__()
        self.game = SnakeGame(grid_size)
        self.renderer: SnakeRenderer | None = None
        self.render_mode = "human"

        self._step_without_apple = 0

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, grid_size, grid_size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def _get_observation(self):
        observation = np.zeros(
            (3, self.game.grid_size, self.game.grid_size), dtype=np.float32
        )

        for position in self.game.snake:
            x = cast(int, position[0])
            y = cast(int, position[1])
            observation[0, y, x] = 1.0

        head_x = cast(int, self.game.snake[0][0])
        head_y = cast(int, self.game.snake[0][1])
        observation[1, head_y, head_x] = 1.0
        observation[0, head_y, head_x] = 0.0

        apple_x = cast(int, self.game.apple[0])
        apple_y = cast(int, self.game.apple[1])
        observation[2, apple_y, apple_x] = 1.0

        return observation

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.game.reset()
        self._step_without_apple = 0

        return self._get_observation(), {}

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        direction = direction_from_action(action)

        try:
            won, eaten = self.game.step(direction)
        except ReverseDirectionError:
            return self._get_observation(), 0.0, False, False, {}
        except (WallCollisionError, SelfCollisionError):
            return self._get_observation(), -10.0, True, False, {}

        if eaten:
            self._step_without_apple = 0
            reward = 10.0
        else:
            self._step_without_apple += 1
            reward = -0.1

        if self._step_without_apple > self.game.grid_size * 4:
            return self._get_observation(), -10.0, True, False, {}

        return self._get_observation(), reward, won, False, {}

    @override
    def render(self) -> None:
        if self.renderer is None:
            self.renderer = SnakeRenderer(self.game)
        self.renderer.render()

    @override
    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
