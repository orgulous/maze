import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import random
import copy


class cardinals:
    NORTH = u'\u2191'
    EAST = u'\u2192'
    SOUTH = u'\u2193'
    WEST = u'\u2190'
    TERMINATE = "terminate"


class QMaze:

    def __init__(self, qvalues, reward_grid, discount_factor, learn_rate, explore_rate, epochs, terminal_grid_ls):

        # starting q values and reward grid
        self.qvalues = qvalues
        self.reward_grid = reward_grid

        # reference list of terminal states
        self.terminal_grid_ls = terminal_grid_ls

        self.row_len = reward_grid.shape[0]
        self.col_len = reward_grid.shape[1]
        self.discount_factor = discount_factor
        self.learn_rate = learn_rate
        self.explore_rate = explore_rate

        # variable to track location of agent
        self.agent_grid = (random.choice(range(self.row_len)),
                           random.choice(range(self.col_len)))
        # start at epoch 1
        self.this_epoch = 1
        self.epoch_goal = epochs

    # append to a dictionary the value of a legal action
    # also checks for validity of action

    def get_dir_val(self, t1, direction, legal_dirs):
        row_t1, col_t1 = t1
        if (row_t1 < 0 or col_t1 < 0 or row_t1 >= self.row_len or col_t1 >= self.col_len):
            # Illegal move. nothing appended
            return legal_dirs
        else:
            # update the dict properly
            state = self.rowcol_to_state(t1)
            legal_dirs[direction] = max(self.qvalues[state])
            return legal_dirs

    # inspect all four directions for values
    # returns a dictionary of legal directions and their values
    def get_dir_vals(self, row, col):

        north_grid = (row - 1, col)
        east_grid = (row, col + 1)
        south_grid = (row + 1, col)
        west_grid = (row, col - 1)

        # track legal direction moves and return it as a dict
        # key = the direction. Value: list of sa values
        legal_dirs = dict.fromkeys(
            [cardinals.NORTH, cardinals.EAST, cardinals.SOUTH, cardinals.WEST])

        legal_dirs = self.get_dir_val(north_grid, cardinals.NORTH, legal_dirs)
        legal_dirs = self.get_dir_val(east_grid, cardinals.EAST, legal_dirs)
        legal_dirs = self.get_dir_val(south_grid, cardinals.SOUTH, legal_dirs)
        legal_dirs = self.get_dir_val(west_grid, cardinals.WEST, legal_dirs)

        # elimate the Nones
        legal_dirs = {k: v for k, v in legal_dirs.items() if v is not None}

        return legal_dirs

    # returns an int that represents the action of the system
    def direction_to_action(self, direction):
        if direction == cardinals.NORTH:
            action_val = 0
        if direction == cardinals.EAST:
            action_val = 1
        if direction == cardinals.SOUTH:
            action_val = 2
        if direction == cardinals.WEST:
            action_val = 3

        return action_val

    # returns an int that represents the state of the system
    # EX (0, 0) -> 0 and (4, 4) -> 24
    def rowcol_to_state(self, grid):
        row, col = grid
        state_val = (row) * (self.col_len) + (col)
        return state_val

    # takes in a direction and state and returns a tuple for the q table
    def sa_to_qtable(self, grid, direction_action):
        action_val = self.direction_to_action(direction_action)
        state_val = self.rowcol_to_state(grid)

        return (state_val, action_val)

    # reached the award/punishment. Terminate or continue
    def is_in_terminal_state(self):
        row, col = self.agent_grid
        if (row, col) in self.terminal_grid_ls:
            return True
        else:
            return False

    # Single iteration of moving to a new grid.
    # returns False until ending a reward/punishment is reached
    def run_iter(self):
        row_t0, col_t0 = self.agent_grid

        # dictionary that gives legal dirs and their estimate of optimal future val
        legal_dirs = self.get_dir_vals(row_t0, col_t0)

        # explore at random OR
        if self.explore_rate > random.random():
            dir_t0 = random.choice(list(legal_dirs.keys()))
        # choose direction based off of max value
        else:
            dir_t0 = max(legal_dirs, key=legal_dirs.get)

        if dir_t0 == cardinals.NORTH:
            row_t1, col_t1 = (row_t0 - 1, col_t0)
        if dir_t0 == cardinals.EAST:
            row_t1, col_t1 = (row_t0, col_t0 + 1)
        if dir_t0 == cardinals.SOUTH:
            row_t1, col_t1 = (row_t0 + 1, col_t0)
        if dir_t0 == cardinals.WEST:
            row_t1, col_t1 = (row_t0, col_t0 - 1)

        # go over the random or best choice
        est_opt_future_val_q = legal_dirs[dir_t0]

        # convert grid location to state, action pair (25x4)
        state, action = self.sa_to_qtable(self.agent_grid, dir_t0)

        # if we are in a terminal state, exit the loop
        if self.is_in_terminal_state() is True:
            return True

        # get the reward value and update it if not termainal state
        self.qvalues[state, action] = self.qvalues[state, action] + \
            self.learn_rate * (
                self.reward_grid[row_t1, col_t1] +
                (self.discount_factor * est_opt_future_val_q) -
                self.qvalues[state, action]
        )

        # the agent has moved. finally move
        self.agent_grid = (row_t1, col_t1)

        # return false because we are not in a terminal state
        return False

    # run a single epoch
    def run_epoch(self):

        terminal_bool = False
        while terminal_bool is False:
            terminal_bool = self.run_iter()

        # Increment epoch number
        self.this_epoch += + 1

        # Reset starting grid
        self.agent_grid = (random.choice(range(self.row_len)),
                           random.choice(range(self.col_len)))

    # Formats q values in north, east, south, west, orientation from 25x4 grid
    # returns a 5x5 numpy text array
    def get_qvalues(self):
        print_ls = list()
        for row in range(25):
            print_ls.append(
                '{0:.2f}'.format(self.qvalues[row, 0]) + '\n'  # North
                '{0:.2f}'.format(self.qvalues[row, 3]) + '     '  # East
                '{0:.2f}'.format(self.qvalues[row, 1]) + '\n'  # South
                '{0:.2f}'.format(self.qvalues[row, 2]) + ''  # West
            )
        print_arr = np.array(print_ls)
        print_arr = np.reshape(print_ls, (5, 5))
        return print_arr

    # main training loop of the algorithm
    def train(self):
        while self.this_epoch <= self.epoch_goal:
            self.run_epoch()
        print_arr = self.get_qvalues()
        return print_arr

    # Extract the value function from the trained q function
    def extractValue(self):
        valueGrid = np.empty((5, 5), dtype=object)
        for row in range(self.row_len):
            for col in range(self.col_len):

                # argmax of the actions based on a row, col state
                action_vals = self.qvalues[self.rowcol_to_state((row, col))]
                valueGrid[row, col] = np.max(action_vals)

        return np.array(valueGrid, dtype=float)

    # Extract the policy from the trained q function
    def extractPolicy(self):
        policyGrid = np.empty((5, 5), dtype=object)
        for row in range(self.row_len):
            for col in range(self.col_len):

                # max of q value for each row, col, and return the policy
                action_vals = self.qvalues[self.rowcol_to_state((row, col))]
                index = np.argmax(action_vals)

                if index == 0:
                    dir = cardinals.NORTH
                if index == 1:
                    dir = cardinals.EAST
                if index == 2:
                    dir = cardinals.SOUTH
                if index == 3:
                    dir = cardinals.WEST

                policyGrid[row, col] = dir

        return policyGrid


''' MAIN CODE '''
# Manual setup of the maze.
qvalues = np.zeros((25, 4))
discount_factor = 0.5
learn_rate = .2
explore_rate = .7
epochs = 200

# hardcoded rewards and punishments
reward_grid = np.zeros((5, 5))
reward_val = 1
punish_val = -1
reward_ls = ((0, 0), (4, 4))
punish_ls = ((1, 0), (3, 1), (2, 3), (3, 4), (1, 2))

# code for display of grid and hardcoded terminal states editing
def edit_array(reward_grid, reward_ls, punish_ls, reward_val, punish_val):
    for elem in reward_ls:
        reward_grid[elem] = reward_val
    for elem in punish_ls:
        reward_grid[elem] = punish_val
    return reward_grid

reward_grid = edit_array(reward_grid, reward_ls,
                         punish_ls, reward_val, punish_val)

grid = QMaze(qvalues, reward_grid, discount_factor,
             learn_rate, explore_rate, epochs, reward_ls + punish_ls)
qarr = grid.train()  # q function
policy = grid.extractPolicy()  # policy function
value = grid.extractValue()  # value function

# and then overwrite the labels (for hardcoded terminal states)
value = edit_array(value, reward_ls, punish_ls, reward_val, punish_val)
policy = edit_array(policy, reward_ls, punish_ls, '', '')

''' PLOTTING CODE '''
fig, ax = plt.subplots()
ax.matshow(value, cmap='summer')

for (i, j), z in np.ndenumerate(qarr):
    ax.text(j, i, z + '\n' + policy[i, j], ha='center', va='center',
            color='blue', fontsize='x-small')

plt.title("Q-Learning Maze")
plt.show()
