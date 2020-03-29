from collections import deque
import numpy as np


class ReplayBuffer():

    def __init__(self, memory_capacity, obs_shape, act_shape):
        self.memory_capacity = memory_capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.memory = {
            'state': np.empty(shape=(self.memory_capacity, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=np.uint8),
            'action': np.empty(shape=(self.memory_capacity, 1), dtype=np.uint8),
            'reward': np.empty(shape=(self.memory_capacity, 1), dtype=np.int8),
            'next_state': np.empty(shape=(self.memory_capacity, obs_shape[0], obs_shape[1], obs_shape[2]), dtype=np.uint8),
            'done': np.empty(shape=(self.memory_capacity, 1), dtype=np.bool)
        }
        self.pointer = 0
        self.fill = 0

    def add(self, state, action, reward, next_state, done):
        def __add_to_subdata(sub_name, idx, data):
            self.memory[sub_name][idx] = data
        assert self.memory
        __add_to_subdata('state', self.pointer, state)
        __add_to_subdata('action', self.pointer, action)
        __add_to_subdata('reward', self.pointer, reward)
        __add_to_subdata('next_state', self.pointer, next_state)
        __add_to_subdata('done', self.pointer, done)

        self.pointer += 1
        self.fill = max(self.fill, self.pointer)
        self.pointer = self.pointer % self.memory_capacity

    def get_batch(self, batch_size):
        def __get_from_subdata(sub_name, idx):
            return self.memory[sub_name][idx, ...]
        assert self.memory
        sample_idx = np.random.choice(self.fill, batch_size, replace=False)
        state_batch = __get_from_subdata('state', sample_idx)
        action_batch = __get_from_subdata('action', sample_idx)
        reward_batch = __get_from_subdata('reward', sample_idx)
        next_state_batch = __get_from_subdata('next_state', sample_idx)
        done_batch = __get_from_subdata('done', sample_idx)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
