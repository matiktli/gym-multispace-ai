# Gym imports
import gym
import gym_multispace
from gym_multispace.env_util import create_env, create_single_env


# Local imports
from model.dqn_model import base_dqn_model
from agent.dqn_agent import base_dqn_agent, DQNAgentSolver
from agent.runner import Runner

# Initialise gym-ai env
scenario_path = 'scenario/single_scenario.py'
env = create_env(scenario_path, is_absolute=True)
# TMP presentation env-------- (single agent)
# env = create_single_env(scenario_path, is_absolute=True)
# ----------------------------
initial_observation = env.reset()

# Initialise model
obs_space_shape = env.observation_space[0].shape
act_space_shape = env.action_space[0].n

dqn_agents = []
for agent in env.agents:
    dqn_model = base_dqn_model(obs_space_shape, act_space_shape)
    dqn_agent_solver = DQNAgentSolver(obs_space_shape,
                                      act_space_shape,
                                      dqn_model,
                                      DQNAgentSolver.MEMORY_SIZE,
                                      DQNAgentSolver.BATCH_SIZE,
                                      DQNAgentSolver.EXPLORATION_RATE)
    dqn_agents.append(dqn_agent_solver)


runner = Runner(env, dqn_agents)
runner.start_learning(no_games=100, no_steps_per_game=100)
