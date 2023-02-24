import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from enum import Enum


class cardinals:

    NORTH = u'\u2191'
    EAST = u'\u2192'
    SOUTH = u'\u2193'
    WEST = u'\u2190'


class policyBlock:
    N_val = 0
    E_val = 0
    S_val = 0
    W_val = 0


values = np.zeros((5, 5))
values[0, 0] = 1  # the complation
values[1, 0] = -.67  # the termination
values[3, 1] = -1
values[3, 2] = -1
values[2, 3] = -1
values[3, 4] = -1
values[4, 4] = 1

row_len = values.shape[0]
col_len = values.shape[1]

discount_factor = 0.8


def getDirVal(direction):
    row, col = direction
    #print("checking " + str(row) + " " + str(col))
    if (row < 0 or col < 0 or row >= row_len or col >= col_len):
        return 0
    else:
        #print("it is " + str(values[row, col]))
        return values[row, col]

# this calculates the value of an action, accounting for probability
# the probability of moving in the same direction as an action is .7
# the other directions will be .1


def calc_action_value(direction, dir_vals):
    (n_val, e_val, s_val, w_val) = dir_vals
    if direction == cardinals.NORTH:
        return n_val * 0.7 + e_val * 0.1 + s_val * 0.1 + w_val * 0.1
    if direction == cardinals.EAST:
        return n_val * 0.1 + e_val * 0.7 + s_val * 0.1 + w_val * 0.1
    if direction == cardinals.SOUTH:
        return n_val * 0.1 + e_val * 0.1 + s_val * 0.7 + w_val * 0.1
    if direction == cardinals.WEST:
        return n_val * 0.1 + e_val * 0.1 + s_val * 0.1 + w_val * 0.7


def get_dir_vals(row, col):
    north_grid = (row - 1, col)
    east_grid = (row, col + 1)
    south_grid = (row + 1, col)
    west_grid = (row, col - 1)

    n_val = getDirVal(north_grid)
    e_val = getDirVal(east_grid)
    s_val = getDirVal(south_grid)
    w_val = getDirVal(west_grid)

    return (n_val, e_val, s_val, w_val)


def get_best_direction(dir_vals):

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


def update_cell_val(row, col):
    # check all four with equal probability

    # do not run the code on the reward/loss cells
    if values[row, col] <= -.5 or values[row, col] == 1:
        return

    dir_vals = get_dir_vals(row, col)
    best_direction = get_best_direction(dir_vals)

    # update section
    values[row, col] = calc_action_value(
        best_direction, dir_vals) * discount_factor


def check_converge(values, values2):
    if (np.allclose(values, values2, 0.01)):
        print("the grids are the same")
        return True
    else:
        print("the grids are NOT the same")
        return False


def run_iteration(values):

    prev_val_copy = np.zeros((5, 5))
    # while there is no convergence
    print_count = 0

    while check_converge(values, prev_val_copy) is False:

        # store previous values
        prev_val_copy = np.copy(values)

        for row in range(row_len):
            for col in range(col_len):
                update_cell_val(row, col)

        print("iteration " + str(print_count))
        print_count = print_count + 1

    print(values)
    # returns the value of all four directions


def extractPolicy(values):
    policyGrid = np.empty((5, 5), dtype=object)
    for row in range(row_len):
        for col in range(col_len):
            dir_vals = get_dir_vals(row, col)
            policyGrid[row, col] = get_best_direction(dir_vals)

    return policyGrid


run_iteration(values)
policy = extractPolicy(values)


'''PLOTTING CODE BELOW'''

print(policy)

fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(values, cmap='binary_r')

for (i, j), z in np.ndenumerate(values):
    ax.text(j, i, '{:0.2f}'.format(z) + policy[i, j], ha='center', va='center')

plt.title("Maze")
plt.show()

# something with the anaconda powershell prompt anaconda3, and the Z drive mounting allowed me to run the anaconda env with python in wsl
