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


# Todo: Add value per grid with the policy extraction
# Disable starting on terminal states
# tracking reward cells
# Add best policy arrows
# Clean print statements
# add exploration factor

class QMaze:

    def __init__(self, qvalues, reward_grid, discount_factor, learn_rate, explore_rate, epochs):

        self.qvalues = qvalues
        self.reward_grid = reward_grid

        # Warning
        # WARNING CHANGE THIS IT WILL CAUSE BUGS EVENTUALLY due to hardcoding
        # To be deleted
        self.reward_val = 1
        self.punish_val = -1

        self.row_len = reward_grid.shape[0]
        self.col_len = reward_grid.shape[1]
        self.discount_factor = discount_factor
        self.learn_rate = learn_rate
        self.explore_rate = explore_rate

        self.this_epoch = 1
        self.epoch_goal = epochs

        self.start_grid = (random.choice(range(self.row_len)),
                           random.choice(range(self.col_len)))
        self.agent_grid = (random.choice(range(self.row_len)),
                           random.choice(range(self.col_len)))
        self.terminal_state = False

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
            # print(self.qvalues[state])
            return legal_dirs

    # inspect all four directions for values
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

        terminal_ls = list()
        terminal_ls.append((0, 0))
        terminal_ls.append((4, 4))
        terminal_ls.append((1, 0))
        terminal_ls.append((3, 1))
        terminal_ls.append((2, 3))
        terminal_ls.append((3, 4))

        if (row, col) in terminal_ls:
            return True
        else:
            return False

    # the agent moves somewhere
    # returns False until ending a reward/punishment is reached
    def run_iter(self):
        row_t0, col_t0 = self.agent_grid

        # dictionary that gives legal dirs and their estimate of optimal future val
        legal_dirs = self.get_dir_vals(row_t0, col_t0)

        # explore at random OR
        if self.explore_rate > random.random():
            # explore a random direction
            dir_t0 = random.choice(list(legal_dirs.keys()))
        else:
            # choose direction based off of max value
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

        # gets the place to put the q value in the table
        state, action = self.sa_to_qtable(self.agent_grid, dir_t0)

        '''
        print("updating cell", state, "taking action", action)

        print("before update:")
        print("current q: " + str(self.qvalues[state, action]))
        print("max a for Q: " + str(est_opt_future_val_q))
        '''

        # get the reward value and update it
        self.qvalues[state, action] = self.qvalues[state, action] + \
            self.learn_rate * (
                self.reward_grid[row_t1, col_t1] +
                (self.discount_factor * est_opt_future_val_q) -
                self.qvalues[state, action]
        )

        # the agent has moved. finally move
        # print("after update:")
        # print("new q: " + str(self.qvalues[state, action]))
        self.agent_grid = (row_t1, col_t1)

        # print("Now the agent is in state: " + str(self.agent_grid) +
        #     " having taken action " + dir_t0)

        if self.is_in_terminal_state() is True:
            return True
        else:
            return False

    # updates the values
    def run_epoch(self):

        terminal_bool = False
        my_iter = 0
        while terminal_bool is False:
            print("Iteration " + str(my_iter))
            terminal_bool = self.run_iter()
            my_iter += 1

        # Epoch done. Reset for new epoch.
        print("Epoch " + str(self.this_epoch) + " completed.")
        self.this_epoch += + 1
        self.agent_grid = (random.choice(range(self.row_len)),
                           random.choice(range(self.col_len)))
        terminal_bool = True

    def print_qvalues(self):

        print_ls = list()
        for row in range(25):
            print_ls.append(
                "Cell " + str(row) + '\n' +
                '{0:.2f}'.format(self.qvalues[row, 0]) + '\n'  # North
                '{0:.2f}'.format(self.qvalues[row, 3]) + '     '  # East
                '{0:.2f}'.format(self.qvalues[row, 1]) + '\n'  # South
                '{0:.2f}'.format(self.qvalues[row, 2]) + ''  # West
            )
        print_arr = np.array(print_ls)
        print_arr = np.reshape(print_ls, (5, 5))
        # print(print_arr)
        return print_arr

    def train(self):
        while self.this_epoch <= self.epoch_goal:
            self.run_epoch()
            print_arr = self.print_qvalues()
            # print(self.qvalues)
        return print_arr

    # extract the policy after value or policy iteration
    def extractPolicy(self):

        # look in each of the grids for best direction and return it
        self.policyGrid = np.empty((5, 5), dtype=object)
        for row in range(self.row_len):
            for col in range(self.col_len):
                dir_vals = self.get_dir_vals(row, col)
                self.policyGrid[row, col] = self.get_best_direction(dir_vals)

        return self.policyGrid


''' MAIN CODE '''
# Manual setup of the maze.
qvalues = np.zeros((25, 4))

# reward of each of the spots in the grid
reward_grid = np.zeros((5, 5))

# custom rewards
reward_val = 1
punish_val = -1
reward_grid[0, 0] = reward_val
reward_grid[4, 4] = reward_val
reward_grid[1, 0] = punish_val
reward_grid[3, 1] = punish_val
reward_grid[2, 3] = punish_val
reward_grid[3, 4] = punish_val
reward_grid[1, 2] = punish_val

discount_factor = 0.7
learn_rate = .5
explore_rate = .5
epochs = 100

grid = QMaze(qvalues, reward_grid, discount_factor,
             learn_rate, explore_rate, epochs)
arr = grid.train()

# policy = grid.extractPolicy()

''' PLOTTING CODE '''
fig, ax = plt.subplots()
ax.matshow(reward_grid, cmap='summer')

for (i, j), z in np.ndenumerate(arr):
    ax.text(j, i, z, ha='center', va='center',
            color='blue', fontsize='x-small')

plt.title("Maze")
plt.show()
