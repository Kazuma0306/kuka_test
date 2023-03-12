import numpy as np
import torch

class ReplayBuffer(): 
    def __init__(self, length = 10000):
    
        # Buffer Collection: (A, S, S',R , log_prob (if available),  D)
        # Done represents a mask of either 0 and 1
        self.length = length
        self.buffer = []

    def add(self, sample):
        if (len(self.buffer) > self.length):
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, batch_size):
        idx = np.random.permutation(len(self.buffer))[:batch_size]

        state_b = []
        action_b = []
        reward_b = []
        nextstate_b = []
        done_b = [] 
        log_prob = []

        for i in idx:
            if (len(self.buffer[0])==5):
                a, s, sp, r, d = self.buffer[i]
            else:
                a, s, sp, r, d, lp = self.buffer[i]
                log_prob.append(lp)

            state_b.append(s)
            action_b.append(a)
            reward_b.append(r)
            nextstate_b.append(sp)
            done_b.append(d)

        state_b = torch.tensor(state_b)
        action_b = torch.tensor(action_b)
        reward_b = torch.tensor(reward_b)
        nextstate_b = torch.tensor(nextstate_b)
        done_b = torch.tensor(done_b)

        if len(self.buffer[0]) == 5:
            return (action_b, state_b, nextstate_b, reward_b, done_b)
        else:
            return (action_b, state_b, nextstate_b, reward_b, done_b, np.array(log_prob))




class ReplayBuffer2:
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool)
        self.next_obs = np.zeros((capacity, *observation_shape), dtype=np.uint8)

        self.index = 0
        self.is_filled = False

        # データ数．
        self._n = 0

    def add(self, action, observation, next_obs, reward, done):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done
        self.next_obs[self.index] = next_obs

        # indexは巡回し, 最も古い経験を上書きする
        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

        self._n = min(self._n + 1, self.capacity)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        return (
            self.actions[idxes],
            self.observations[idxes],
            self.next_obs[idxes],
            self.rewards[idxes],
            self.done[idxes],
        )