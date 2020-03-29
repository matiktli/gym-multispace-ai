# Gym imports
import gym
import gym_multispace
from gym_multispace.env_util import create_env

# Local imports
import model.ddqn_model as nn
from agent.ddqn_agent import DDQNAgentSolver
from agent.ddqn_runner import Runner
from utils.argument_parser import get_arguments_for_ddqn

# Other imports
import os


args = get_arguments_for_ddqn()
scenario_path = args.scenario_path
no_games = args.no_games
no_steps_per_game = args.no_steps_per_game
save_replay_every_n_games = args.save_replay_every_n_games
save_weights_every_n_games = args.save_weights_every_n_games

path_to_save_assets = args.path_to_save_assets
if not os.path.exists(path_to_save_assets):
    os.makedirs(path_to_save_assets)
agent_exploration_rate = args.agent_exploration_rate
agent_memory_size = args.agent_memory_size
agent_batch_size = args.agent_batch_size
agent_learning_rate = args.agent_learning_rate


# Initialise gym-ai env
env = create_env(scenario_path, is_absolute=True)
# ----------------------------
initial_observation = env.reset()

# Initialise model
obs_space_shape = env.observation_space[0].shape
act_space_shape = env.action_space[0].n

ddqn_agents = []
for agent in env.agents:
    ddqn_model_wrapper = nn.DDQN(
        obs_space_shape, act_space_shape, agent_learning_rate)

    target_ddqn_model_wrapper = nn.DDQN(
        obs_space_shape, act_space_shape, agent_learning_rate)

    ddqn_agent_solver = DDQNAgentSolver(obs_space_shape,
                                        act_space_shape,
                                        ddqn_model_wrapper,
                                        target_ddqn_model_wrapper,
                                        agent_exploration_rate,
                                        agent_memory_size,
                                        agent_batch_size,
                                        target_model_update_freq=no_steps_per_game,
                                        train_freq=4,
                                        no_steps_per_game=no_steps_per_game)
    ddqn_agents.append(ddqn_agent_solver)


runner = Runner(env, ddqn_agents, path_to_save_assets,
                save_replay_every_n_games, save_weights_every_n_games)
runner.start_learning(no_games=no_games,
                      no_steps_per_game=no_steps_per_game)

# save weights after training
runner.save_weights(path_to_save_assets)
