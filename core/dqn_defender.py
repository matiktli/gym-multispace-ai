# Gym imports
import gym
import gym_multispace
from gym_multispace.env_util import create_env, create_single_env


# Local imports
from model.dqn_model import base_dqn_model
from agent.dqn_agent import base_dqn_agent

# Initialise gym-ai env
scenario_path = 'scenario/defender_scenario.py'
env = create_env(scenario_path, is_absolute=True)
# TMP presentation env-------- (single agent)
env = create_single_env(scenario_path, is_absolute=True)
# ----------------------------
initial_observation = env.reset()

# Initialise model
obs_space_shape = env.observation_space[0].shape
print('obs_s: ', env.observation_space[0])
act_space_shape = env.action_space[0].n
dqn_model = base_dqn_model(obs_space_shape, act_space_shape)

# Build keras-rl custom agent
dqn_agent = base_dqn_agent(dqn_model, act_space_shape)

dqn_agent.fit(env, nb_steps=100, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn_agent.save_weights('dqn_{}_weights.h5f'.format(0), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn_agent.test(env, nb_episodes=5, visualize=True)
