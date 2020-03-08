from rl_matikitli.agents.dqn import DQNAgent as DQNAgent_mk
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random


# Using keras rl... not working fine
def base_dqn_agent(model, input_shape):
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent_mk(model=model,
                      nb_actions=input_shape,
                      memory=memory,
                      nb_steps_warmup=10,
                      target_model_update=1e-2,
                      policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


# Custom implementation
class DQNAgentSolver():

    MEMORY_SIZE = 1000000
    BATCH_SIZE = 25

    EXPLORATION_RATE = 0.2

    def __init__(self,
                 observation_space,
                 action_space,
                 model,
                 memory_capacity=None,
                 batch_size=None,
                 exploration_rate=None):
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
            state = np.reshape(state, (1, 1, 4))
            q_values = self.model.predict(state)
            result = np.argmax(q_values[0])
        return result

    def experience_replay(self):
        # TODO implement experience replay
        # https://github.com/gsurma/cartpole/blob/master/cartpole.py
        pass
