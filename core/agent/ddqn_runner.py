import numpy as np
import cv2
from gym_multispace.renderer import Renderer
import imageio


class Runner():

    def __init__(self, env, agent_solvers: [], path_to_save_assets, save_weights_each_ep: int, save_replay_each_ep: int):
        self.env = env
        self.agent_solvers = agent_solvers
        self.meta = {
            'ep_no': 0,
            'ep_score': 0,
            'ep_steps': 0,
            'ep_loss': 0,
            'ep_acc': 0
        }

        self.path_to_save_assets = path_to_save_assets
        self.save_weights_each_ep = save_weights_each_ep
        self.save_replay_each_ep = save_replay_each_ep

    def start_learning(self, no_games, no_steps_per_game):
        while self.meta['ep_no'] <= no_games:
            is_replay_record_game = self.meta['ep_no'] % self.save_replay_each_ep == 0
            is_save_weights_game = self.meta['ep_no'] % self.save_weights_each_ep == 0

            record_data = self.play_one_game(no_steps_per_game,
                                             is_replay_record_game)
            if is_replay_record_game:
                self.save_replay_to_gif(
                    record_data, self.path_to_save_assets, self.meta['ep_no'])

            if is_save_weights_game:
                self.save_weights(self.path_to_save_assets, self.meta['ep_no'])
            self.meta['ep_no'] += 1

    def play_one_game(self, no_steps_per_game, track_game=False):
        self.meta['ep_steps'] = 0
        state_n = self.env.reset()
        is_game_done = False
        game_replay = [] if track_game else None

        def __make_decissions_for_agents(agents, state_n, step_no):
            agents_actions = []
            for i, agent in enumerate(agents):
                state_i = np.reshape(
                    state_n[i], agent.observation_space)
                action = agent.make_decission(state_i)
                agents_actions.append(action)
                agent.update_exploration(step_no)
            return agents_actions

        def __remember_for_agents(agents, state_n, action_n, reward_n, next_state_n, done_n):
            for i, agent in enumerate(agents):
                agent.remember(state_n[i], action_n[i],
                               reward_n[i], next_state_n[i], done_n[i])

        def __perform_learning_for_agents(agents, step_no):
            for i, agent in enumerate(agents):
                hist = agent.step_update(step_no)

        while self.meta['ep_steps'] <= no_steps_per_game and not is_game_done:
            agents_actions = __make_decissions_for_agents(
                self.agent_solvers, state_n, self.meta['ep_steps'])

            new_state_n, reward_n, done_n, info_n = self.env.step(
                agents_actions)

            if track_game:
                game_replay.append(state_n[0])

            print(
                f"------ (Game: {self.meta['ep_no']}, Step: {self.meta['ep_steps']})\n\tReward: {reward_n}\n\tObservation: {new_state_n if len(str(new_state_n)) < 50 else '...'}\n\tInfo: {info_n}\n\tDone: {done_n}\n------")

            __remember_for_agents(self.agent_solvers, state_n, agents_actions,
                                  reward_n, new_state_n, done_n)

            __perform_learning_for_agents(
                self.agent_solvers, self.meta['ep_steps'])
            state_n = new_state_n
            self.meta['ep_steps'] += 1

        return game_replay

    def save_weights(self, path, serial_no=0):
        for agent, agent_solver in zip(self.env.world.objects_agents_ai, self.agent_solvers):
            path_with_agent_name = path + \
                f'agent__{agent.uuid}__v{serial_no}'
            agent_solver.save_weights(path_with_agent_name)

    def save_replay_to_gif(self, replay_data, path, serial_no):
        imageio.mimsave(path + f'game_{serial_no}.gif', replay_data)
