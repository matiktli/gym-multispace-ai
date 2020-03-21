# Multi-Agent Gym Environment - Learning AI

Learning agent tool for project: [gym-multispace](https://github.com/matiktli/gym-multispace).\
(BSc Software Engineer final project)

## About

Simple reinforecement learning tool for multi agent enviornment in format of [Gym](https://gym.openai.com/) from OpenAi.

## Running

Running simple DQN Agent. The reward in this scenario i s based on the distance of the agent to the goal.
```python
# Gym imports
from gym_multispace.env_util import create_env

# Local imports
import model.dqn_model as models
from agent.dqn_agent import base_dqn_agent, DQNAgentSolver
from agent.runner import Runner

# Initialise gym-ai env
scenario_path = 'scenario/single_scenario.py'
env = create_env(scenario_path, is_absolute=True)
# ----------------------------
initial_observation = env.reset()

# Initialise model(s)
obs_space_shape = env.observation_space[0].shape
act_space_shape = env.action_space[0].n

dqn_agents = []
for agent in env.agents:
    dqn_model = models.load_model('model_name', obs_space_shape, act_space_shape, learning_rate)
    dqn_agent_solver = DQNAgentSolver(obs_space_shape,
                                      act_space_shape,
                                      dqn_model,
                                      DQNAgentSolver.EXPLORATION_RATE,
                                      DQNAgentSolver.MEMORY_SIZE,
                                      DQNAgentSolver.BATCH_SIZE)
    dqn_agents.append(dqn_agent_solver)


runner = Runner(env, dqn_agents)
runner.start_learning(no_games=700, no_steps_per_game=500,
                      render_every_n_games=50, path_to_save_gif='game_1/gifs')

# save weights after training
runner.save_weights('game_1/weights')
```

## License

Free

---

Thanks to Hailite.io for inspiration: <https://2016.halite.io/index.html>.\
Thanks to OpenAi for sharing resources: <https://github.com/openai/multiagent-particle-envs>.
