import random
import numpy as np

class EpsGreedyPolicy:
    def __init__(self, qtable, n_actions, epsilon):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.qtable = qtable
        
    def __call__(self, obs, return_is_greedy=False):
        if random.random() < self.epsilon:
            if return_is_greedy:
                return random.randint(0, self.n_actions-1), False
            else:
                return random.randint(0, self.n_actions-1)
        else:
            vals = self.qtable[obs]
            maxval = np.max(vals)
            argmax = np.where(vals == maxval)[0]
            
            if return_is_greedy:
                return random.choice(argmax), True
            else:
                return random.choice(argmax)

    def proba(self, obs):
        num_actions = self.qtable.shape[1]
        proba = np.full((num_actions,), self.epsilon/num_actions)

        vals = self.qtable[obs]
        maxval = np.max(vals)
        argmax = np.where(vals == maxval)[0]

        proba[argmax] += (1.0 - self.epsilon) / len(argmax)

        return proba

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def __call__(self, obs):
        return self.action_space.sample()

    def proba(self, state):
        num_actions = self.action_space.n
        return np.full((num_actions,), 1.0/num_actions)
