# pyright: reportUnknownMemberType=false

from typing import cast

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from snake.env import SnakeEnv

GRID_SIZE = 8


def train() -> None:
    env = SnakeEnv(GRID_SIZE)
    env = DummyVecEnv([lambda: env])  # pyright: ignore[reportArgumentType]
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
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
