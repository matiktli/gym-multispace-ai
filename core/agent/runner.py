import numpy as np
import cv2
from gym_multispace.renderer import Renderer


class Runner():

    def __init__(self, env, agent_solvers: []):
        self.env = env
        self.agent_solvers = agent_solvers

    def show_image_with_info(self, image, game_counter, step_counter, reward_n):
        if image is not None:
            for i, info_text in enumerate([f'Game: _{game_counter}_', f'Step: _{step_counter}_', f'Reward(s): {reward_n}']):
                image = cv2.putText(image, info_text, (2, 10+i*15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.imshow(Renderer.WINDOW_NAME, image)
            cv2.waitKey(1)

    def start_learning(self, no_games, no_steps_per_game, render_every_n_games=10):
        game_counter = 0
        while game_counter <= no_games:
            state_n = self.env.reset()
            step_counter = 0
            is_game_done = False
            show_img = False
            if game_counter % render_every_n_games == 0:
                show_img = True
            while step_counter <= no_steps_per_game and not is_game_done:
                action_n = []
                for i, solver in enumerate(self.agent_solvers):
                    state_i = np.reshape(
                        state_n[i], solver.observation_space[0])
                    action = solver.make_decission(state_i)
                    action_n.append(action)

                observation_n_next, reward_n, done_n, info_n = self.env.step(
                    action_n)

                if show_img:
                    rendered_image = self.env.render(mode='terminal')
                    self.show_image_with_info(
                        rendered_image, game_counter, step_counter, reward_n)
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
            cv2.destroyAllWindows()

    def save_weights(self, path):
        # TODO save weights logic
        pass

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
