import numpy as np


class Runner():

    def __init__(self, env, agent_solvers: []):
        self.env = env
        self.agent_solvers = agent_solvers

    def start_learning(self, no_games, no_steps_per_game):
        game_counter = 0
        while game_counter <= no_games:
            input(f'Game: {game_counter} ...')
            state_n = self.env.reset()
            step_counter = 0
            # TODO we have bug with all games ending when ifrst ones ends
            is_game_done = False
            while step_counter <= no_steps_per_game and not is_game_done:
                self.env.render()
                action_n = []
                for i, solver in enumerate(self.agent_solvers):
                    state_i = np.reshape(
                        state_n[i], solver.observation_space[0])
                    action = solver.make_decission(state_i)
                    action_n.append(action)

                observation_n_next, reward_n, done_n, info_n = self.env.step(
                    action_n)

                for i, solver in enumerate(self.agent_solvers):
                    observation_next, reward, done, info = observation_n_next[
                        i], reward_n[i], done_n[i], info_n[i]
                    if done:
                        is_game_done = True

                    state_i = np.reshape(
                        state_n[i], solver.observation_space[0])
                    solver.add_to_memory(
                        state_i, action_n[i], reward, observation_next, done)
                    solver.experience_replay()

                state_n = observation_n_next
                step_counter += 1
            game_counter += 1

    # def fit(self, env):
    #     run = 0
    #     while True:
    #         run += 1
    #         state = env.reset()
    #         state = np.reshape(state, [1, self.observation_space[0]])
    #         step = 0
    #         input('---Next game...')
    #         is_game_done = False
    #         while not is_game_done:
    #             step += 1
    #             env.render()
    #             action = self.act(state)
    #             observation_next, reward, done, info = env.step(action)
    #             is_game_done = done
    #             observation_next = np.reshape(
    #                 observation_next, [1, self.observation_space[0]])
    #             self.remember(state, action, reward,
    #                           observation_next, done)
    #             state = observation_next
    #             self.experience_replay()
