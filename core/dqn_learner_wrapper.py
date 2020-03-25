# Gym imports
import gym
import gym_multispace
from gym_multispace.env_util import create_env

# Local imports
import model.dqn_model as nn
from agent.dqn_agent import DQNAgentSolver
from agent.dqn_runner import Runner

# Other imports
import argparse
import os


# Parse arguments
parser = argparse.ArgumentParser(description='DQN Learner')
parser.add_argument('--scenario_path', type=str,
                    help='Path to the scenario that will be learned.')
parser.add_argument('--no_games', type=int,
                    help='Number of games to play')
parser.add_argument('--no_steps_per_game', type=int,
                    help='Number of steps that one game going to last.')
parser.add_argument('--render_every_n_games', type=int,
                    default=100,
                    help='Specify how often to render environment.')
parser.add_argument('--path_to_save_assets', type=str,
                    help='Path to where assets would be saved.')

# Collect DQN solver properties
parser.add_argument('--agent_exploration_rate', type=float,
                    default=DQNAgentSolver.EXPLORATION_RATE,
                    help='Agent`s exploration rate.')
parser.add_argument('--agent_memory_size', type=int,
                    default=DQNAgentSolver.MEMORY_SIZE,
                    help='Agent`s memory size.')
parser.add_argument('--agent_batch_size', type=int,
                    default=DQNAgentSolver.BATCH_SIZE,
                    help='Agent`s batch size.')
parser.add_argument('--agent_learning_rate', type=float,
                    default=DQNAgentSolver.LEARNING_RATE,
                    help='Agent`s learning rate.')

args = parser.parse_args()
scenario_path = args.scenario_path
no_games = args.no_games
no_steps_per_game = args.no_steps_per_game
render_every_n_games = args.render_every_n_games
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

dqn_agents = []
for agent in env.agents:
    dqn_model_wrapper = nn.DQN(obs_space_shape, act_space_shape, agent_learning_rate)

    dqn_agent_solver = DQNAgentSolver(obs_space_shape,
                                      act_space_shape,
                                      dqn_model_wrapper,
                                      agent_exploration_rate,
                                      agent_memory_size,
                                      agent_batch_size)
    dqn_agents.append(dqn_agent_solver)


runner = Runner(env, dqn_agents)
runner.start_learning(no_games=no_games,
                      no_steps_per_game=no_steps_per_game,
                      render_every_n_games=render_every_n_games,
                      path_to_save_gif=path_to_save_assets)

# save weights after training
runner.save_weights(path_to_save_assets)
