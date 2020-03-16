import numpy as np
import cv2
from gym_multispace.renderer import Renderer
import imageio


class Runner():

    def __init__(self, env, agent_solvers: []):
        self.env = env
        self.agent_solvers = agent_solvers

    def show_image_with_info(self, image, game_counter, step_counter, reward_n, path_to_save_gif=''):
        if image is not None:
            for i, info_text in enumerate([f'Game: _{game_counter}_', f'Step: _{step_counter}_', f'Reward(s): {reward_n}']):
                image = cv2.putText(image, info_text, (2, 10+i*15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        return image

    def start_learning(self, no_games, no_steps_per_game, render_every_n_games=10, path_to_save_gif=''):
        game_counter = 0
        while game_counter <= no_games:
            state_n = self.env.reset()
            step_counter = 0
            is_game_done = False
            show_img = False
            if game_counter % render_every_n_games == 0:
                show_img = True

            replay_game_visual_storage = []
            while step_counter <= no_steps_per_game and not is_game_done:
                action_n = []
                for i, solver in enumerate(self.agent_solvers):
                    state_i = np.reshape(
                        state_n[i], solver.observation_space)
                    action = solver.make_decission(state_i)
                    action_n.append(action)

                observation_n_next, reward_n, done_n, info_n = self.env.step(
                    action_n)

                print(
                    f"------ (Game: {game_counter}, Step: {step_counter})\n\tReward: {reward_n}\n\tObservation: {observation_n_next}\n\tInfo: {info_n}\n\tDone: {done_n}\n------")

                if show_img:
                    rendered_image = self.env.render(mode='terminal')
                    image_with_additional_stats = self.show_image_with_info(rendered_image,
                                                                            game_counter,
                                                                            step_counter,
                                                                            reward_n,
                                                                            path_to_save_gif)
                    if image_with_additional_stats is not None:
                        replay_game_visual_storage.append(
                            image_with_additional_stats)

                for i, solver in enumerate(self.agent_solvers):
                    observation_next, reward, done, info = observation_n_next[
                        i], reward_n[i], done_n[i], info_n[i]
                    if done:
                        is_game_done = True

                    state_i = np.reshape(
                        state_n[i], solver.observation_space)
                    solver.add_to_memory(
                        state_i, action_n[i], reward, observation_next, done)
                    solver.experience_replay()

                state_n = observation_n_next
                step_counter += 1
            if len(replay_game_visual_storage) > 0:
                self.save_replay_to_gif(
                    replay_game_visual_storage, path_to_save_gif + '/game_' + str(game_counter))
            game_counter += 1
            cv2.destroyAllWindows()

    def save_weights(self, path):
        if len(self.agent_solvers) > 1:
            for agent, agent_solver in zip(self.env.world.objects_agents_ai, self.agent_solvers):
                path_with_agent_name = path + '_' + agent.uuid
                agent_solver.save_weights(path_with_agent_name)
        else:
            path_with_agent_name = path + '_' + \
                self.env.world.objects_agents_ai[0].uuid
            self.agent_solvers[0].save_weights(path_with_agent_name)

    def save_replay_to_gif(self, replay_data, path):
        if replay_data[0]:
            imageio.mimsave(path + '.gif', replay_data)
