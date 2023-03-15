''' Implementation of simple gridworld reinforcement learning problem
Adapted code from pytorch example and openai gym environment example
Some code generated with ChatGPT

Environment: GridWorldEnv in grid_world.py. This shows the 
    grid environment with nxnx2 observation space (2 variables in a nxn grid)
    The DQNAgent interacts with this environment

DQNAgent: keeps track of the agent
    Replay buffer/Experience buffer is not implemented in this example

QNetwork: This is the torch neural network
'''
import grid_world as gw
from DQNAgent import DQNAgent

'''
size: The size of the square grid
env: our environment
obs_size: number of observations in total
n_actions: possible action size
hidden size: neural net hidden layer size

gamma: the discount factor of the neural net
eps values: how epsilon for explore vs exploit will decay
num_episodes: how many episodes of learning to take into account
'''

size = 10
env = gw.GridWorldEnv(size=size)
obs_size = (size * size * 2)
n_actions = env.action_space.n
hidden_size = 50

gamma = 0.95
eps_start = 1
eps_end = 0.01
eps_decay = 0.996
num_episodes = 2000

agent = DQNAgent(env, size, obs_size, hidden_size, n_actions, gamma,
                 eps_start, eps_end, eps_decay)

agent.train(num_episodes)
agent.evaluate(num_episodes)
