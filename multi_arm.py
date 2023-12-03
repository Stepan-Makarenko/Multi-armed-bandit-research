import gymnasium as gym
from gymnasium import spaces, Env, utils
import numpy as np
from typing import Optional, List, Tuple

class MultiArmedBanditEnv(Env):
    """
    A simple multi-armed bandit environment, compatible with OpenAI Gym.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, probabilities: List[float] = [0.3, 0.5], max_steps: int = 10000, seed: Optional[int] = None) -> None:
        super(MultiArmedBanditEnv, self).__init__()
        self.n_arms = len(probabilities)
        self.max_steps = max_steps
        self.curr_step = 0
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(1)  # No real observations, just a dummy space

        self.seed(seed)
        self.probabilities = probabilities

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[int, int, bool, dict]:
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        self.curr_step += 1
        
        # Reward is 1 with probability of the chosen arm, otherwise 0
        reward =  self.np_random.binomial(1, self.probabilities[action])
        done = self.max_steps < self.curr_step
        return 0, reward, done, {}

    def reset(self) -> int:
        self.curr_step = 0
        return 0  # Resetting a bandit environment does not change its state

    def render(self, mode: str = 'human', close: bool = False) -> None:
        pass  # Rendering is not needed for this simple environment

    def close(self) -> None:
        pass
