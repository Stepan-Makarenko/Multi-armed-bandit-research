import numpy as np
from gymnasium import Space


class RandomAgent:
    def __init__(self, action_space: Space) -> None:
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

    def update_state(self, action: int, reward: int) -> None:
        return

    def reset(self) -> None:
        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class ExploreAndGreed:
    def __init__(self, action_space: Space, exploration_prob: float = 0.1) -> None:
        assert (
            0 <= exploration_prob <= 1
        ), f'"eploration_prob" should be in [0, 1] range, current value = {exploration_prob}'
        self.exploration_prob = exploration_prob
        self.arm_num = action_space.n
        self.arm_count = np.zeros(self.arm_num)
        self.arm_reward = np.zeros(self.arm_num)

    def act(self) -> int:
        # act based on current internal state

        # exploitation step
        if np.random.random() > self.exploration_prob:
            return np.argmax(self.arm_reward / (self.arm_count + 1e-5))
        else:  # exploration step
            return np.random.choice(self.arm_num)

    def update_state(self, action: int, reward: int) -> None:
        # update stats based on observed reward
        self.arm_count[action] += 1
        self.arm_reward[action] += reward

    def reset(self) -> None:
        self.arm_count = np.zeros(self.arm_num)
        self.arm_reward = np.zeros(self.arm_num)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exploration_prob={self.exploration_prob})"


class UCB1:
    def __init__(self, action_space: Space, exploration_prob: float = 0.1) -> None:
        assert (
            0 <= exploration_prob <= 1
        ), f'"eploration_prob" should be in [0, 1] range, current value = {exploration_prob}'
        self.arm_num = action_space.n
        self.arm_count = np.ones(self.arm_num)
        self.arm_reward = np.zeros(self.arm_num)

    def act(self) -> int:
        # act based on current internal state

        # exploitation step
        return np.argmax(
            self.arm_reward / (self.arm_count + 1e-5)
            + np.sqrt(2 * np.log(self.arm_count.sum()) / (self.arm_count + 1e-5))
        )

    def update_state(self, action: int, reward: int) -> None:
        # update stats based on observed reward
        self.arm_count[action] += 1
        self.arm_reward[action] += reward

    def reset(self) -> None:
        self.arm_count = np.ones(self.arm_num)
        self.arm_reward = np.zeros(self.arm_num)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
