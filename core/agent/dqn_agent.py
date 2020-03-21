from collections import deque
import numpy as np
import random


# Custom implementation
class DQNAgentSolver():

    MEMORY_SIZE = 1000000
    BATCH_SIZE = 25

    EXPLORATION_RATE = 0.2

    GAMMA = 0.95
    LEARNING_RATE = 0.001

    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

    def __init__(self,
                 observation_space,
                 action_space,
                 model,
                 exploration_rate=None,
                 memory_capacity=None,
                 batch_size=None):
        self.model = model
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def make_decission(self, state):
        # Add exploration parameter
        if np.random.rand() < self.exploration_rate:
            result = random.randrange(self.action_space)
        else:
            tmp = (1, ) + state.shape
            state = np.reshape(state, tmp)
            q_values = self.model.predict(state)
            result = np.argmax(q_values[0])
        return result

    def experience_replay(self):
        # Experience replay borowed from:
        # https://github.com/gsurma/cartpole/blob/master/cartpole.py
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            tmp_1 = (1,) + state.shape
            tmp_2 = (1,) + state_next.shape
            state = np.reshape(state, tmp_1)
            state_next = np.reshape(state_next, tmp_2)
            q_update = reward
            if not terminal:
                q_update = (reward + DQNAgentSolver.GAMMA *
                            np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= DQNAgentSolver.EXPLORATION_DECAY
        self.exploration_rate = max(
            DQNAgentSolver.EXPLORATION_MIN, self.exploration_rate)

    def save_weights(self, path):
        self.model.save_weights(path + '.h5f')
