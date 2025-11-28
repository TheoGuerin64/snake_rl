from typing import cast, final

import pygame

from snake.game import SnakeGame

COLOR_CLEAR = pygame.Color(0, 0, 0, 0)


@final
class SnakeRenderer:
    _WINDOW_TITLE = "Snake RL"
    _WINDOW_SIZE = 600

    _BACKGROUND_COLOR = pygame.Color(30, 30, 30)
    _GRID_COLOR = pygame.Color(50, 50, 50)
    _GRID_LINE_COLOR = pygame.Color(80, 80, 80)
    _SNAKE_START_COLOR = pygame.Color(0, 200, 0)
    _SNAKE_END_COLOR = pygame.Color(0, 100, 0)
    _SNAKE_BORDER_COLOR = pygame.Color(0, 150, 150)
    _APPLE_COLOR = pygame.Color(200, 0, 0)

    def __init__(self, game: SnakeGame) -> None:
        self.game = game

        pygame.init()
        pygame.display.set_caption(self._WINDOW_TITLE)
        self._screen = pygame.display.set_mode((self._WINDOW_SIZE, self._WINDOW_SIZE))
        self._clock = pygame.time.Clock()

        self._tile_size = (self._WINDOW_SIZE - 1) // self.game.grid_size
        self._grid_surface, self._grid_rect = self._build_grid_surface()
        self._overlay_surface = pygame.Surface(self._grid_rect.size, pygame.SRCALPHA)

    def _build_grid_surface(self) -> tuple[pygame.Surface, pygame.Rect]:
        size = self._tile_size * self.game.grid_size + 1
        offset = (self._WINDOW_SIZE - size) // 2
        rect = pygame.Rect(offset, offset, size, size)

        surface = pygame.Surface((size, size))
        surface.fill(self._GRID_COLOR)

        for x in range(0, size, self._tile_size):
            pygame.draw.line(surface, self._GRID_LINE_COLOR, (x, 0), (x, size), 1)
        for y in range(0, size, self._tile_size):
            pygame.draw.line(surface, self._GRID_LINE_COLOR, (0, y), (size, y), 1)

        return surface, rect

    def _render_snake(self) -> None:
        color = self._SNAKE_START_COLOR
        for segment in self.game.snake:
            segment_x = cast(int, segment[0])
            segment_y = cast(int, segment[1])
            segment_rect = pygame.Rect(
                segment_x * self._tile_size,
                segment_y * self._tile_size,
                self._tile_size,
                self._tile_size,
            )

            pygame.draw.rect(self._overlay_surface, color, segment_rect)
            color = color.lerp(self._SNAKE_END_COLOR, 1 / (len(self.game.snake) - 1))

            pygame.draw.rect(
                self._overlay_surface, self._SNAKE_BORDER_COLOR, segment_rect, 2
            )

    def render(self) -> None:
        self._overlay_surface.fill(COLOR_CLEAR)

        apple_x = cast(int, self.game.apple[0])
        apple_y = cast(int, self.game.apple[1])
        apple_rect = pygame.Rect(
            apple_x * self._tile_size,
            apple_y * self._tile_size,
            self._tile_size,
            self._tile_size,
        )
        pygame.draw.rect(self._overlay_surface, self._APPLE_COLOR, apple_rect)
        self._render_snake()

        self._screen.fill(self._BACKGROUND_COLOR)
        self._screen.blit(self._grid_surface, self._grid_rect.topleft)
        self._screen.blit(self._overlay_surface, self._grid_rect.topleft)

        pygame.display.flip()
        self._clock.tick(10)

    def close(self) -> None:
        pygame.quit()
