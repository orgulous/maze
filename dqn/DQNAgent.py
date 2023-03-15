import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from QNetwork import QNetwork

# This agent takes actions based off of a Q network or randomness
# The agent gets the reward and trains the Q network on each episode


class DQNAgent:
    def __init__(self, env, size, obs_size, hidden_size,
                 n_actions, gamma, eps_start, eps_end, eps_decay):

        self.env = env
        self.obs_size = obs_size
        self.size = size
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # our Q network and our optimizer built on it
        self.policy_net = QNetwork(obs_size, hidden_size, n_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters())

        # a history of how many iterations to solve one episode
        # one episode = one game where agent is able to move to the target
        # one episode consists of several game steps
        self.iter_count_hist = []

    # takes action on the board

    def act(self, obs, eps):
        # random action
        if np.random.uniform() < eps:
            return self.env.action_space.sample()

        # action based off of q network
        else:
            # reshaping of the numpy array to a tensor in right shape
            obs_tensor = torch.from_numpy(obs).view(-1)

            # get 4 action values from the policy network
            q_values = self.policy_net(obs_tensor)

            # pick the best action value
            action = np.argmax(q_values.detach())

            # return as scalar
            return action.item()

    # given the step info, learn from it
    def learn(self, obs, next_obs, action, reward, done):

        # reshape our numpy types to tensors
        obs_tensor = torch.from_numpy(obs).view(-1)
        next_obs_tensor = torch.from_numpy(next_obs).view(-1)

        # Get "current" q_value to compare to new one for a State, Action pair
        q_values = self.policy_net(obs_tensor)
        q_value = q_values[action]

        with torch.no_grad():
            # Find the next q_values (actions) based off the t+1 State, Action
            next_q_values = self.policy_net(next_obs_tensor)
            # Make sure to find the max value one, in the right shape/dimension
            next_q_value = torch.max(next_q_values.unsqueeze(dim=0))

            # calculate the scalar
            expected_q_value = reward + \
                self.gamma * next_q_value * (1 - done)

        # compare the losses
        loss = F.mse_loss(q_value, expected_q_value)

        # use optimizer to train the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # evaluate the model
    def evaluate(self, num_episodes, tests=5):

        # count number of successful tests
        success_count = tests
        for test in range(tests):
            obs, _ = self.env.reset()

            # print statements visualize the maze movement
            #print("Running Test", test)
            #print(np.add(obs[0], obs[1]), '\n')

            done = False
            iter = 1

            # if the model doesn't finish in (size x 3) steps, consider it a fail
            # go through the action steps
            while not done:
                if iter > (self.size * 3):
                    success_count -= 1
                    break
                action = self.act(obs, 0)
                next_obs, reward, done, _ = self.env.step(action)
                obs = next_obs
                iter += 1
                #print(np.add(obs[0], obs[1]), '\n')

        print("The agent's success rate is ", success_count, "/", tests)

        ax = plt.axes()
        ax.plot(range(num_episodes), self.iter_count_hist)
        plt.xlabel("Game Episode")
        plt.ylabel("Number of Steps to Complete Game")
        plt.title("Deep Q Network performance in Gridworld by training episode")
        plt.show()

    # main training loop
    def train(self, num_episodes):
        eps = self.eps_start

        # going through each game episode
        for episode in range(num_episodes):

            # initial values to solve
            obs, _ = self.env.reset()

            # initial starting conditions
            done = False

            # steps to solve tracker
            iter = 1

            # loop for a single episode
            while not done:
                # extract an action
                action = self.act(obs, eps)

                # take the action in env, receive reward and new agent location
                next_obs, reward, done, _ = self.env.step(action)

                # learn from these values using q network
                self.learn(obs, next_obs, action, reward, done)

                # update agent location
                obs = next_obs

                # track steps to solve.
                iter += 1

                # ends cycle if agent gets stuck in local minimum
                if iter > 500:
                    break
            self.iter_count_hist.append(iter)

            # epsilon decay
            eps = max(self.eps_end, eps * self.eps_decay)
