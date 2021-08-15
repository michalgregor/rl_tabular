import numpy as np
from .value_functions import StateValueTable

class TDLearning:
    def __init__(self, vtable, alpha=0.8, gamma=0.95, lambd=0):
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.vtable = vtable
        self.etrace = StateValueTable()

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_lambda(self, lambd):
        self.lambd = lambd

    def __call__(self, obs, a, reward, obs_next, done, info):
        assert(len(obs) == 1)
        obs = obs[0]; a = a[0]; reward = reward[0]
        obs_next = obs_next[0]; done = done[0]; info = info[0]

        self.etrace *= self.gamma * self.lambd
        self.etrace[obs] += 1

        td = self.alpha * (reward + self.gamma * self.vtable[obs_next]
                 - self.vtable[obs])
        self.vtable += td * self.etrace

        if done: self.etrace *= 0

class QLearning:
    def __init__(self, qtable, alpha=0.8, gamma=0.95):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.qtable = qtable

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma

    def __call__(self, obs, a, reward, obs_next, done, info):
        next_val = self.gamma * np.max(self.qtable[obs_next], axis=1) * (np.asarray(done) == False)
        td = reward + next_val - self.qtable[obs, a]
        self.qtable[obs, a] += self.alpha * td

class SARSA:
    def __init__(self, qtable, policy, alpha=0.8, gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma
        self.qtable = qtable
        self.policy = policy

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_gamma(self, gamma):
        self.gamma = gamma

    def __call__(self, obs, a, reward, obs_next, done, info):
        next_a = np.asarray([self.policy(on) for on in obs_next])
        next_val = self.gamma * self.qtable[obs_next, next_a] * (np.asarray(done) == False)
        td = reward + next_val - self.qtable[obs, a]
        self.qtable[obs, a] += self.alpha * td