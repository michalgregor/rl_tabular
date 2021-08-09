from .utils import RingBuffer
from .policy import RandomPolicy
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = RingBuffer(max_size, dtype=object)

    def prefill(self, env, batch_size, policy=None):
        if policy is None:
            policy = RandomPolicy(env.action_space)
        
        obs = env.reset()
        done = False
        
        for i in range(batch_size):
            if done:
                obs = env.reset()

            a = policy(obs)

            obs_next, reward, done, _ = env.step(a)
            self.buffer.append((obs, a, reward, obs_next, done))
            obs = obs_next
            
    def add(self, obs, a, reward, obs_next, done):
        self.buffer.append((obs, a, reward, obs_next, done))
        
    def add_batch(self, batch):
        self.buffer.extend(batch)
        
    def sample(self, batch_size):
        rand_index = np.random.randint(0, len(self.buffer), batch_size)
        return self.get(rand_index)
    
    def get(self, index=slice(None)):
        batch = self.buffer[index]
        batch = np.asarray(list(batch), dtype=object)
        
        obs = list(batch[:, 0])
        a = batch[:, 1].astype(int)
        reward = batch[:, 2].astype(float)
        obs_next = list(batch[:, 3])
        done = batch[:, 4].astype(bool)
        
        return obs, a, reward, obs_next, done
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = RingBuffer(self.buffer.maxlen, dtype=object)