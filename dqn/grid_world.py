# Maze environment
import gym
from gym import spaces
import numpy as np
import random

# Girdworld environment


class GridWorldEnv(gym.Env):

    def __init__(self, size=5):

        self.size = size

        self.obs_state = np.zeros((2, size, size), dtype=float)
        self.agent_location = None
        self.target_location = None

        self.observation_space = spaces.Discrete((size * size)**2)
        self.action_space = spaces.Discrete(4)

    def _action_to_direction(self, action):
        act_dir_dict = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        return act_dir_dict[action]

    def _get_obs(self):
        return self.obs_state

    # not used
    def _get_info(self):
        return None

    # random sstarting location
    def _reset_random_loc(self):
        randx = random.randint(0, self.size-1)
        randy = random.randint(0, self.size-1)
        return [randx, randy]

    # turns locations into an array
    def _reset_observation(self):

        x_ag, y_ag = self.agent_location
        x_tg, y_tg = self.target_location

        ag_array = np.zeros((self.size, self.size))
        tg_array = np.zeros((self.size, self.size))

        ag_array[x_ag, y_ag] = 1
        tg_array[x_tg, y_tg] = 1

        self.obs_state = np.array([ag_array, tg_array], dtype=float)

        return

    def reset(self, seed=None, options=None):
        # we need the following line to seed self.np_random
        super().reset(seed=seed)

        # agent and target have random starting locations
        self.agent_location = self._reset_random_loc()
        self.target_location = self._reset_random_loc()

        # the gridworld variables reset with new starting locations
        self._reset_observation()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # Take a step in the environment based off of an action
    def step(self, action):
        # map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction(action)

        # we use `np.clip` to make sure we don't leave the grid
        self.agent_location = np.clip(
            self.agent_location + direction, 0, self.size - 1
        )

        # updates the observation so that agent has moved, but target is same
        self._reset_observation()

        # an episode is done iff the agent has reached the target
        terminated = np.array_equal(
            self.agent_location, self.target_location)
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, info
