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

    def __init__(self, values, policy, discount_factor):
        # check to see if direction is a wall
        self.values = values
        self.valuesPrev = None

        # initialize the policy grid to default values
        self.policyGrid = policy
        self.policyGridPrev = None

        self.row_len = values.shape[0]
        self.col_len = values.shape[1]
        self.discount_factor = discount_factor
        self.iteration = 0

    def getDirVal(self, direction):
        row, col = direction
        if (row < 0 or col < 0 or row >= self.row_len or col >= self.col_len):
            return 0
        else:
            return values[row, col]

    # this calculates the value of an action, accounting for probability
    # the probability of moving in the same direction as an action is .7
    # the other directions will be .1

    def calc_action_value_iter(self, direction, dir_vals):
        (n_val, e_val, s_val, w_val) = dir_vals
        if direction == cardinals.NORTH:
            return n_val * 0.7 + e_val * 0.1 + s_val * 0.1 + w_val * 0.1
        if direction == cardinals.EAST:
            return n_val * 0.1 + e_val * 0.7 + s_val * 0.1 + w_val * 0.1
        if direction == cardinals.SOUTH:
            return n_val * 0.1 + e_val * 0.1 + s_val * 0.7 + w_val * 0.1
        if direction == cardinals.WEST:
            return n_val * 0.1 + e_val * 0.1 + s_val * 0.1 + w_val * 0.7

    # inspect all four directions for values
    def get_dir_vals(self, row, col):
        north_grid = (row - 1, col)
        east_grid = (row, col + 1)
        south_grid = (row + 1, col)
        west_grid = (row, col - 1)

        n_val = self.getDirVal(north_grid)
        e_val = self.getDirVal(east_grid)
        s_val = self.getDirVal(south_grid)
        w_val = self.getDirVal(west_grid)

        return (n_val, e_val, s_val, w_val)

    # see which direction has the max value
    def get_best_direction(self, dir_vals):

        (n_val, e_val, s_val, w_val) = dir_vals

        max_val = max(n_val, e_val, s_val, w_val)
        best_direction = None

        if (max_val == n_val):
            best_direction = cardinals.NORTH
        if (max_val == e_val):
            best_direction = cardinals.EAST
        if (max_val == s_val):
            best_direction = cardinals.SOUTH
        if (max_val == w_val):
            best_direction = cardinals.WEST

        return best_direction

    def update_cell_val(self, row, col, policyIterate=False):

        # do not run the code on the reward/loss cells
        if self.values[row, col] <= -.5 or self.values[row, col] == 1:
            return

        dir_vals = self.get_dir_vals(row, col)

        if policyIterate is False:
            best_direction = self.get_best_direction(dir_vals)
            # update section
            self.values[row, col] = self.calc_action_value_iter(
                best_direction, dir_vals) * self.discount_factor
        else:  # policyIterate is True
            # get direction from POLICY, not values
            # then do the action using a different cal_action-value
            cell_policy_dir = self.policyGrid[row, col]
            cell_val = self.calc_action_value_iter(cell_policy_dir, dir_vals)

            # updated the values based on the policy
            self.values[row, col] = cell_val * self.discount_factor

    def check_converge(self, arr1, arr2, isValPol=False):
        if isValPol is True:
            if (np.array_equal(arr1, arr2)):
                print("the policy grids are the same")
                return True
            else:
                print("the policy grids are NOT the same")
                return False

        elif (np.allclose(arr1, arr2, 0.01)):
            print("the value grids are the same")
            return True
        else:
            print("the value grids are NOT the same")
            return False

    def valueIteration(self):
        self.valuesPrev = np.zeros((5, 5))

        while self.check_converge(self.values, self.valuesPrev) is False:

            # store previous values
            self.valuesPrev = np.copy(self.values)

            for row in range(self.row_len):
                for col in range(self.col_len):
                    self.update_cell_val(row, col)

            print("iteration " + str(self.iteration))
            self.iteration = self.iteration + 1

        print(self.values)

    # gets the policy once the valuation has converged

    def valuePolicyIteration(self):

        print(self.policyGrid, self.policyGridPrev)

        while self.check_converge(self.policyGrid, self.policyGridPrev, isValPol=True) is False:

            self.policyGridPrev = np.copy(self.policyGrid)

            for row in range(self.row_len):
                for col in range(self.col_len):
                    # True triggers the policy/val iter
                    # This means we will take the action based off initial random policy
                    self.update_cell_val(row, col, True)
                    dir_vals = self.get_dir_vals(row, col)
                    self.policyGrid[row, col] = self.get_best_direction(
                        dir_vals)

            print("reached")
            print("iteration " + str(self.iteration))
            self.iteration = self.iteration + 1
        print(self.policyGrid)

    def extractPolicy(self):
        self.policyGrid = np.empty((5, 5), dtype=object)
        for row in range(self.row_len):
            for col in range(self.col_len):
                dir_vals = self.get_dir_vals(row, col)
                self.policyGrid[row, col] = self.get_best_direction(dir_vals)

        return self.policyGrid


# Manual setup of the maze.
values = np.zeros((5, 5))
values[0, 0] = 1  # the completion
values[1, 0] = -.67  # the termination
values[3, 1] = -.7
values[3, 2] = -.7
values[2, 3] = -.7
values[3, 4] = -.7
values[4, 4] = 1

policy = np.ndarray((5, 5), dtype=object)
policy.fill(cardinals.NORTH)
print(policy)
print("main code")
discount_factor = 0.8

grid = Maze(values, policy, discount_factor)
# grid.valueIteration()
grid.valuePolicyIteration()
policy = grid.extractPolicy()

print(policy)

fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(values, cmap='binary_r')

for (i, j), z in np.ndenumerate(values):
    ax.text(j, i, '{:0.2f}'.format(z) + policy[i, j], ha='center', va='center')

plt.title("Maze")
plt.show()

# something with the anaconda powershell prompt anaconda3, and the Z drive mounting allowed me to run the anaconda env with python in wsl
