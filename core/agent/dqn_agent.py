from collections import deque
import numpy as np
import random
from agent.base_agent import BaseAgent


# Custom implementation
class DQNAgentSolver(BaseAgent):

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
                 model_wrapper,
                 exploration_rate=None,
                 memory_capacity=None,
                 batch_size=None):
        self.model_wrapper = model_wrapper
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
            state = np.expand_dims(np.asarray(
                state).astype(np.float64), axis=0)
            q_values = self.model_wrapper.model.predict(state, batch_size=1)
            result = np.argmax(q_values[0])
        return result

    def experience_replay(self):
        # Experience replay borowed from:
        # https://github.com/gsurma/cartpole/blob/master/cartpole.py
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:

            state = np.expand_dims(np.asarray(
                state).astype(np.float64), axis=0)
            state_next = np.expand_dims(np.asarray(
                state_next).astype(np.float64), axis=0)

            q_update = reward
            if not terminal:
                q_update = (reward + DQNAgentSolver.GAMMA *
                            np.amax(self.model_wrapper.model.predict(state_next)[0]))
            q_values = self.model_wrapper.model.predict(state)
            q_values[0][action] = q_update
            self.model_wrapper.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= DQNAgentSolver.EXPLORATION_DECAY
        self.exploration_rate = max(
            DQNAgentSolver.EXPLORATION_MIN, self.exploration_rate)

    def save_weights(self, path):
        self.model.save_weights(path + '.h5f')
