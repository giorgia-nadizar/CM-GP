import gymnasium as gym
import numpy as np
import random


class SimpleActionOnlyEnv(gym.Env):
    """ Continuous bandit: one state, always the same, 1-timestep episodes, and the reward is based on sin(action)
    """

    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=np.zeros((1,)),
            high=np.ones((1,))
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones((1,)),
            high=np.ones((1,))
        )

    def reset(self, **kwargs):
        return np.zeros((1,)), {}

    def step(self, a):
        reward = np.sin(a * 3.1516).sum()

        return np.zeros((1,)), reward, True, False, {}


class SimpleLargeActionEnv(gym.Env):
    """ Continuous bandit with high-dimensional action
    """

    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=np.zeros((1,)),
            high=np.ones((1,))
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones((16,)),
            high=np.ones((16,))
        )

    def reset(self, **kwargs):
        return np.zeros((1,)), {}

    def step(self, a):
        reward = np.sin(a * 3.1516).sum()

        return np.zeros((1,)), reward, True, False, {}


class SimpleTwoStatesEnv(SimpleActionOnlyEnv):
    """ Contextual bandit. 1-timestep episodes, each episode is in one of two possible states (selected at random). The reward function depends on the state
    """

    def reset(self, **kwargs):
        self.state = random.randrange(2)

        return np.array([self.state], dtype=np.float32), {}

    def step(self, a):
        if self.state == 0:
            reward = np.sin(a * 3.1416).sum()
        else:
            reward = -np.sin(a * 3.1416).sum()

        return np.array([self.state], dtype=np.float32), reward, True, False, {}


class SimpleSequenceEnv(SimpleTwoStatesEnv):
    """ 2 states, 10-timestep episodes, the action sometimes causes the agent to change state. Allows to check that V(s_t+1) is computed correctly by the agent
    """

    def reset(self, **kwargs):
        self.state = random.randrange(2)
        self.timestep = 0

        return self.current_state(), {}

    def step(self, a):
        reward = super().step(a)[1]

        if self.state == 1:
            reward += 0.1  # A bit better to be in state 1

        # Transition
        self.timestep += 1

        if a > 0.5:
            self.state = 1
        else:
            self.state = 0

        return self.current_state(), reward, self.timestep >= 10, False, {}

    def current_state(self):
        return np.array([self.state], dtype=np.float32)


class SimpleGoalEnv(gym.Env):
    """ Continuous navigation task: observe (x, y), produce (dx, dy), and aim at reaching (0, 0). The range of observations is [0, 1] (so we try to reach a corner)
    """

    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=np.zeros((2,)),
            high=np.ones((2,))
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones((2,)),
            high=np.ones((2,))
        )
        self.state = np.zeros((2,), dtype=np.float32)

    def reset(self, **kwargs):
        self.state[0] = random.random()
        self.state[1] = random.random()
        self._timestep = 0

        return self.state, {}

    def step(self, a):
        def distance_to_goal(s):
            # The goal is 0, 0, so the distance to goal is sqrt(self.state^2)
            return np.sqrt((s ** 2).sum())

        a = np.clip(a, -1.0, 1.0)

        old_d = distance_to_goal(self.state)
        self.state = np.clip(self.state + 0.1 * a, 0.0, 1.0)
        new_d = distance_to_goal(self.state)

        x, y = self.state
        reward = 10. * (old_d - new_d)  # Reward for getting closer to the goal

        self._timestep += 1

        done = False

        if x < 0.1 and y < 0.1:
            # Close enough to the goal
            reward = 10.0
            done = True

        if 0.4 < x < 0.6 and 0.4 < y < 0.6:
            # Obstacle in the middle, punish the agent
            reward = -10.0
            done = True

        return self.state, reward, done or (self._timestep > 50), False, {}


class SimpleGoalEnvSpeed(gym.Env):
    """ Continuous navigation task: observe (x, y, speed_x, speed_y), produce (dx, dy), and aim at reaching (0, 0). The range of observations is [0, 1] (so we try to reach a corner)
    """

    def __init__(self):
        super().__init__()

        # Position + speed
        self.observation_space = gym.spaces.Box(
            low=np.zeros((4,)),
            high=np.ones((4,))
        )

        # Acceleration
        self.action_space = gym.spaces.Box(
            low=-np.ones((2,)),
            high=np.ones((2,))
        )

        self.state = np.zeros((4,), dtype=np.float32)

    def reset(self, **kwargs):
        # position
        self.state[0] = random.random()
        self.state[1] = random.random()
        # speed
        self.state[2], self.state[3] = 0, 0

        self._timestep = 0

        return self.state, {}

    def step(self, a):
        def distance_to_goal(s):
            # The goal is 0, 0, so the distance to goal is sqrt(self.state^2)
            return np.sqrt((s ** 2).sum())

        a = np.clip(a, -1.0, 1.0)

        # Add acceleration to speed
        self.state[2] += a[0]
        self.state[3] += a[1]

        old_d = distance_to_goal(self.state)

        # Add speed to position
        self.state[0] = np.clip(self.state[0] + 0.1*self.state[2], 0.0, 1.0)
        self.state[1] = np.clip(self.state[1] + 0.1*self.state[3], 0.0, 1.0)

        new_d = distance_to_goal(self.state)

        x, y, speed_x, speed_y = self.state
        reward = 10. * (old_d - new_d)  # Reward for getting closer to the goal

        self._timestep += 1

        done = False

        if x < 0.1 and y < 0.1:
            # Close enough to the goal
            reward = 10.0
            done = True

        if 0.4 < x < 0.6 and 0.4 < y < 0.6:
            # Obstacle in the middle, punish the agent
            reward = -10.0
            done = True

        return self.state, reward, done or (self._timestep > 50), False, {}