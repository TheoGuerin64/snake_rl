import random
from collections import deque
from enum import Enum
from typing import NamedTuple, cast, final

import numpy as np
import numpy.typing as npt

IVector = npt.NDArray[np.int_]


class ReverseDirectionError(Exception):
    pass


class WallCollisionError(Exception):
    pass


class SelfCollisionError(Exception):
    pass


class Direction(tuple[int, int], Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def opposite(self) -> "Direction":
        match self:
            case Direction.UP:
                return Direction.DOWN
            case Direction.DOWN:
                return Direction.UP
            case Direction.LEFT:
                return Direction.RIGHT
            case Direction.RIGHT:
                return Direction.LEFT


class StepResult(NamedTuple):
    won: bool
    ate_apple: bool


@final
class SnakeGame:
    def __init__(self, grid_size: int = 10) -> None:
        """
        Raises:
            ValueError: If `grid_size` is less than 3.
        """

        if grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        self.grid_size = grid_size

        self._free_positions: set[tuple[int, int]]

        self.snake: deque[IVector]
        self.direction: Direction
        self.apple: IVector

        self._setup()

    def _setup(self) -> None:
        self._free_positions = {
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        }

        x_start = self.grid_size // 2
        y_start = self.grid_size // 2
        segments = (
            (x_start + 1, y_start),
            (x_start, y_start),
            (x_start - 1, y_start),
        )

        self.snake = deque()
        for segment in segments:
            self.snake.append(np.array(segment))
            self._free_positions.remove(segment)

        self.direction = Direction.RIGHT
        self.apple = self._random_apple_position()

    def _random_apple_position(self) -> IVector:
        position = random.choice(tuple(self._free_positions))
        return np.array(position)

    def reset(self) -> None:
        self._setup()

    def _is_out_of_bounds(self, position: IVector) -> bool:
        x = cast(int, position[0])
        y = cast(int, position[1])
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    def step(self, action: Direction) -> StepResult:
        """
        Raises:
            ValueError: If the action is invalid (reverses direction) or
                results in a collision.
        """

        if action == self.direction.opposite():
            raise ReverseDirectionError("Cannot reverse direction")
        self.direction = action

        new_head = self.snake[0] + self.direction
        if self._is_out_of_bounds(new_head):
            raise WallCollisionError("Snake collided with wall")
        if tuple(new_head) not in self._free_positions:
            raise SelfCollisionError("Snake collided with itself")

        self.snake.appendleft(new_head)
        try:
            self._free_positions.remove(tuple(new_head))
        except KeyError:
            pass

        ate_apple = np.array_equal(new_head, self.apple)

        if not self._free_positions:
            return StepResult(True, ate_apple)

        if ate_apple:
            self.apple = self._random_apple_position()
        else:
            tail = self.snake.pop()
            self._free_positions.add(tuple(tail))

        return StepResult(False, ate_apple)
