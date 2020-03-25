from collections import deque
import numpy as np
import random
from agent import BaseAgent, LinearSchedule


# Custom implementation
class DDQNAgentSolver(BaseAgent):

    MEMORY_SIZE = 10000
    BATCH_SIZE = 32

    EXPLORATION_RATE = 0.1
    EPSILON = 1
    EPSILON_MIN = 0.01

    GAMMA = 0.99
    LEARNING_RATE = 0.0001
    TARGET_MODEL_UPDATE_RATE = 500

    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

    def __init__(self,
                 observation_space,
                 action_space,
                 model_wrapper,
                 target_model_wrapper,
                 exploration_rate=None,
                 memory_capacity=None,
                 batch_size=None,
                 target_model_update_freq=None,
                 train_freq=4):
        self.model_wrapper = model_wrapper
        self.target_model_wrapper = target_model_wrapper
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.epsilon = DDQNAgentSolver.EPSILON
        self.epsilon_min = DDQNAgentSolver.EPSILON_MIN
        self.exploration = LinearSchedule(
            schedule_timesteps=int(self.exploration_rate * int(1e7)),
            initial_p=self.epsilon,
            final_p=self.epsilon_min
        )

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def make_decission(self, state):
        if np.random.rand() < self.epsilon:
            result = random.randrange(self.action_space)
        else:
            state = np.expand_dims(state, axis=0).astype(float)/255
            q_values = self.model_wrapper.model.predict(state, batch_size=1)
            result = np.argmax(q_values[0])
        return result

    def experience_replay(self):
        # TODO
        pass

    def save_weights(self, path):
        assert self.model_wrapper, self.target_model_wrapper
        self.model_wrapper.save_model_weights(path + '_local.h5f')
        self.target_model_wrapper.save_model_weights(path + '_target.h5f')

    def _update_target_model(self):
        assert self.model_wrapper, self.target_model_wrapper
        local_model_weights = self.model_wrapper.get_weights()
        self.target_model_wrapper.model.set_weights(local_model_weights)

    def _update_exploration(self, t):
        self.epsilon = self.exploration.value(t)
