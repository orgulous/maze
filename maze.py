import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from enum import Enum


class cardinals:
    NORTH = u'\u2191'
    EAST = u'\u2192'
    SOUTH = u'\u2193'
    WEST = u'\u2190'


class Maze:

    def __init__(self, values, policy, discount_factor, reward_val, punish_val):
        # check to see if direction is a wall
        self.values = values
        self.valuesPrev = None
        self.reward_val = reward_val
        self.punish_val = punish_val

        # initialize the policy grid to default values
        self.policyGrid = policy
        self.policyGridPrev = None

        self.row_len = values.shape[0]
        self.col_len = values.shape[1]
        self.discount_factor = discount_factor
        self.iteration = 0

    # this calculates the value of an action, accounting for probability
    def calc_action_value_iter(self, direction, legal_dirs):

        main_dir_val = legal_dirs[direction]

        other_legal_dirs = dict(legal_dirs)
        other_legal_dirs.pop(direction)
        other_dir_vals = other_legal_dirs.values()
        n_other_dirs = len(other_legal_dirs)

        # probability of moving in main direction vs other directions
        main_dir_prob = .7
        secondary_dir_prob = .3 / n_other_dirs

        other_sum = 0
        for x in other_dir_vals:
            other_sum = other_sum + (x * secondary_dir_prob)

        return main_dir_val * main_dir_prob + other_sum

    # append to a dictionary the value of a legal action
    # also checks for validity of action
    def get_dir_val(self, t1, direction, legal_dirs):
        row_t1, col_t1 = t1
        if (row_t1 < 0 or col_t1 < 0 or row_t1 >= self.row_len or col_t1 >= self.col_len):
            # Illegal move. nothing appended
            return legal_dirs
        else:
            # update the dict properly
            legal_dirs[direction] = values[row_t1, col_t1]
            return legal_dirs

    # inspect all four directions for values
    def get_dir_vals(self, row, col):

        north_grid = (row - 1, col)
        east_grid = (row, col + 1)
        south_grid = (row + 1, col)
        west_grid = (row, col - 1)

        # track legal direction moves and return it as a dict
        legal_dirs = dict.fromkeys(
            [cardinals.NORTH, cardinals.EAST, cardinals.SOUTH, cardinals.WEST])

        legal_dirs = self.get_dir_val(north_grid, cardinals.NORTH, legal_dirs)
        legal_dirs = self.get_dir_val(east_grid, cardinals.EAST, legal_dirs)
        legal_dirs = self.get_dir_val(south_grid, cardinals.SOUTH, legal_dirs)
        legal_dirs = self.get_dir_val(west_grid, cardinals.WEST, legal_dirs)

        # elimate the Nones
        legal_dirs = {k: v for k, v in legal_dirs.items() if v is not None}

        return legal_dirs

    # see which direction has the max value
    def get_best_direction(self, legal_dirs):

        best_direction = max(legal_dirs, key=legal_dirs.get)
        return best_direction

    # get direction from POLICY, not values
    def update_cell_val_by_policy(self, row, col):

        if self.values[row, col] in (self.punish_val, self.reward_val):
            return

        # get values in each direction based off of iterating values
        legal_dirs = self.get_dir_vals(row, col)

        # then take the action based off iterating policy
        cell_policy_dir = self.policyGrid[row, col]

        # update values based off of iteration
        self.values[row, col] = self.calc_action_value_iter(
            cell_policy_dir, legal_dirs) * self.discount_factor

    # update cell values based off of best value
    def update_cell_val(self, row, col):
        # do not run the code on the reward/loss cells
        if self.values[row, col] in (self.punish_val, self.reward_val):
            return

        # get values of each legal direction
        legal_dirs = self.get_dir_vals(row, col)

        # return best direction
        best_direction = self.get_best_direction(legal_dirs)

        # update values based off of iteration
        self.values[row, col] = self.calc_action_value_iter(
            best_direction, legal_dirs) * self.discount_factor

    # check convergence of policy
    def check_converge_policy(self, arr1, arr2):
        if (np.array_equal(arr1, arr2)):
            print("the policy grids are the same")
            return True
        else:
            print("the policy grids are NOT the same")
            return False

    # check convergence of value
    def check_converge_values(self, arr1, arr2):
        if arr2 is not None:
            if (np.allclose(arr1, arr2, 0.001)):
                print("the value grids are the same")
                return True
            else:
                return False
        else:
            print("the value grids are NOT the same")
            return False

    # value iteration main loop
    def valueIteration(self):
        while self.check_converge_values(self.values, self.valuesPrev) is False:
            print("value iteration " + str(self.iteration))
            print(self.values)

            # store previous values
            self.valuesPrev = np.copy(self.values)

            for row in range(self.row_len):
                for col in range(self.col_len):
                    self.update_cell_val(row, col)

            self.iteration = self.iteration + 1

    # policy iteration main loop
    def policyIteration(self):

        while self.check_converge_policy(self.policyGrid, self.policyGridPrev) is False:
            print("policy iteration " + str(self.iteration))
            print(self.policyGrid)

            # copy of previous grid to check comparison
            self.policyGridPrev = np.copy(self.policyGrid)

            # edit each cell
            for row in range(self.row_len):
                for col in range(self.col_len):
                    self.update_cell_val_by_policy(row, col)
                    dir_vals = self.get_dir_vals(row, col)

                    self.policyGrid[row, col] = self.get_best_direction(
                        dir_vals)

            self.iteration = self.iteration + 1

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
values = np.zeros((5, 5))

# custom rewards
reward_val = 1
punish_val = -1
values[0, 0] = reward_val
values[4, 4] = reward_val
values[1, 0] = punish_val
values[3, 1] = punish_val
values[2, 3] = punish_val
values[3, 4] = punish_val

policy = np.ndarray((5, 5), dtype=object)
policy.fill(cardinals.NORTH)
policy[0] = [cardinals.SOUTH] * 5

discount_factor = 0.8

grid = Maze(values, policy, discount_factor, reward_val, punish_val)

# Select value or policy iteration here
# grid.valueIteration()
grid.policyIteration()

policy = grid.extractPolicy()

''' PLOTTING CODE '''
fig, ax = plt.subplots()
ax.matshow(values, cmap='binary_r')

for (i, j), z in np.ndenumerate(values):
    ax.text(j, i, '{:0.2f}'.format(z) + policy[i, j], ha='center', va='center')

plt.title("Maze")
plt.show()
