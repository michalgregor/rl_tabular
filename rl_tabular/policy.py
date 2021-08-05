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

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def __call__(self, obs):
        return self.action_space.sample()
