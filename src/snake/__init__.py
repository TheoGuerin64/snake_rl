# pyright: reportUnknownMemberType=false

from typing import cast, final, override

import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
)
from torch import nn

from snake.env import SnakeEnv

GRID_SIZE = 8


@final
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = cast(
                int,
                self.cnn(
                    torch.as_tensor(observation_space.sample()[None]).float()
                ).shape[1],
            )

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    @override
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.linear(self.cnn(observations)))


def train() -> None:
    env = SnakeEnv(GRID_SIZE)
    env = Monitor(env)  # pyright: ignore[reportUnknownVariableType]
    env = DummyVecEnv([lambda: env])  # pyright: ignore[reportArgumentType, reportUnknownLambdaType]
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=100,
        gradient_steps=25,
        target_update_interval=2_000,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        verbose=1,
        policy_kwargs={
            "features_extractor_class": MinigridFeaturesExtractor,
            "normalize_images": False,
        },
    )
    model.learn(total_timesteps=2_000_000, progress_bar=True)
    model.save("snake_rl")


def play() -> None:
    env = SnakeEnv(GRID_SIZE)
    env = DummyVecEnv([lambda: env])  # pyright: ignore[reportArgumentType]
    env = VecFrameStack(env, n_stack=4)
    model = DQN.load("snake_rl", env=env)

    observation = env.reset()
    done = False
    while not done:
        action, _states = model.predict(observation, deterministic=True)  # pyright: ignore[reportArgumentType]
        observation, _rewards, dones, _infos = env.step(action)  # pyright: ignore[ reportUnknownVariableType]
        env.render()

        done = cast(bool, dones[0])
