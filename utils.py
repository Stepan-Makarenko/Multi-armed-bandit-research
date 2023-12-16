import random
from typing import List
from gymnasium import Env
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set the seed for all random number generators to ensure reproducibility.

    Args:
    seed (int): The seed number.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random module


def run_experiment(env: Env, agent) -> tuple[np.ndarray[float], np.ndarray[int]]:
    # What if envs are different lengths?
    rewards = []
    is_opt_action = []
    _ = env.reset()
    agent.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act()
        opt_action = env.get_optimal_action()
        _, reward, done, _ = env.step(action)
        agent.update_state(action, reward)
        total_reward += reward
        rewards.append(total_reward)
        is_opt_action.append(int(action == opt_action))
    return np.array(rewards), np.array(is_opt_action)


def average_results(all_rewards: List[np.ndarray]) -> np.ndarray:
    summed_rewards = np.sum(np.array(all_rewards), axis=0)
    return summed_rewards / len(all_rewards)
