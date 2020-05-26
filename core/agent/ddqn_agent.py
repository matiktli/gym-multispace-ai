import numpy as np
import random
from agent.base_agent import BaseAgent, LinearSchedule
from agent.replay_buffer import ReplayBuffer


# Custom implementation
class DDQNAgentSolver(BaseAgent):

    MEMORY_SIZE = 300000
    BATCH_SIZE = 32

    EXPLORATION_RATE = 0.1
    EPSILON = 1
    EPSILON_MIN = 0.01

    GAMMA = 0.99
    LEARNING_RATE = 0.0001
    LEARN_START = MEMORY_SIZE

    TRAIN_FREQ = 4
    UPDATE_TARGET_FREQ = 5000

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
                 train_freq=4,
                 no_steps_per_game=200):
        self.model_wrapper = model_wrapper
        self.target_model_wrapper = target_model_wrapper
        self.observation_space = observation_space
        self.action_space = action_space

        self.exploration_rate = exploration_rate
        self.epsilon = DDQNAgentSolver.EPSILON
        self.epsilon_min = DDQNAgentSolver.EPSILON_MIN
        self.exploration = LinearSchedule(
            schedule_timesteps=int(self.exploration_rate * no_steps_per_game),
            initial_p=self.epsilon,
            final_p=self.epsilon_min
        )

        self.replay_buffer = ReplayBuffer(
            memory_capacity, observation_space, action_space)
        self.batch_size = batch_size

        self._update_target_model()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def make_decission(self, state):
        if np.random.rand() < self.epsilon:
            result = random.randrange(self.action_space)
        else:
            state = np.expand_dims(state, axis=0).astype(float)/255
            q_values = self.model_wrapper.model.predict(state, batch_size=1)
            result = np.argmax(q_values[0])
        return result

    # https://github.com/hridayns/Research-Project-on-Reinforcement-learning/blob/06b1de576a0820b680d8481cbae85db6fccdf804/Atari/models/DDQN.py#L109
    def experience_replay(self):
        if self.replay_buffer.fill < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batch(
            self.batch_size)
        target = self.model_wrapper.model.predict(
            state_batch.astype(float) / 255, batch_size=self.batch_size)

        done_mask = done_batch.ravel()
        undone_mask = np.invert(done_batch).ravel()

        target[done_mask, action_batch[done_mask].ravel(
        )] = reward_batch[done_mask].ravel()

        Q_target = self.target_model_wrapper.model.predict(
            next_state_batch.astype(float)/255, batch_size=self.batch_size)
        Q_future = np.max(Q_target[undone_mask], axis=1)

        target[undone_mask, action_batch[undone_mask].ravel(
        )] = reward_batch[undone_mask].ravel() + DDQNAgentSolver.GAMMA * Q_future

        hist = self.model_wrapper.model.fit(state_batch.astype(
            float)/255, target, batch_size=self.batch_size, verbose=0).history
        return hist

    def save_weights(self, path):
        assert self.model_wrapper, self.target_model_wrapper
        self.model_wrapper.save_model_weights(path + '__local.h5f')
        self.target_model_wrapper.save_model_weights(path + '__target.h5f')

    def _update_target_model(self):
        assert self.model_wrapper, self.target_model_wrapper
        local_model_weights = self.model_wrapper.model.get_weights()
        self.target_model_wrapper.model.set_weights(local_model_weights)

    def update_exploration(self, t):
        self.epsilon = self.exploration.value(t)

    def step_update(self, t):
        hist = None
        if t <= DDQNAgentSolver.LEARN_START:
            return hist
        if t % DDQNAgentSolver.TRAIN_FREQ == 0:
            hist = self.experience_replay()
        if t % DDQNAgentSolver.UPDATE_TARGET_FREQ == 0:
            self._update_target_model()
        return hist
