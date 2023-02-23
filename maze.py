import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from enum import Enum


class policyBlock:
    N_val = 0
    E_val = 0
    S_val = 0
    W_val = 0


values = np.zeros((5, 5))
values[0, 0] = 1  # the complation
values[1, 0] = -.67  # the termination
values[3, 3] = -1
values[3, 4] = -1
values[3, 2] = -1
values[3, 1] = -1
values[4, 4] = 1

row_len = values.shape[0]
col_len = values.shape[1]

discount_factor = 0.8


def returnValue(direction):
    row, col = direction
    #print("checking " + str(row) + " " + str(col))
    if (row < 0 or col < 0 or row >= row_len or col >= col_len):
        return 0
    else:
        #print("it is " + str(values[row, col]))
        return values[row, col]


def update_cell_val(row, col):
    # check all four with equal probability

    if values[row, col] <= -.5 or values[row, col] == 1:
        return

    north_grid = (row - 1, col)
    east_grid = (row, col + 1)
    south_grid = (row + 1, col)
    west_grid = (row, col - 1)

    n_val = returnValue(north_grid)
    e_val = returnValue(east_grid)
    s_val = returnValue(south_grid)
    w_val = returnValue(west_grid)

    # update section
    values[row, col] = max(n_val, e_val, s_val, w_val) * discount_factor


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


# enters iteration loop
run_iteration(values)

#plt.imshow(values, cmap="binary_r")
# plt.xticks([])
# plt.yticks([])


fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(values, cmap='binary_r')

for (i, j), z in np.ndenumerate(values):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

plt.title("Maze")
plt.show()


'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
dx, dy = 0.016, 0.06

P = np.arange(-5.0, 5.0, dx)
print(P, "\n"*3)
Q = np.arange(-5.0, 5.0, dy)
print(Q, "\n"*3)
P, Q = np.meshgrid(P, Q)
print(P, "\n"*3, Q)
  
min_max = np.min(P), np.max(P), np.min(Q), np.max(Q)
res = np.add.outer(range(8), range(8)) % 2
plt.imshow(res, cmap="binary_r")
plt.xticks([])
plt.yticks([])
plt.title("Using Matplotlib Python to Create chessboard")
plt.show() '''
