from rl_matikitli.agents.dqn import DQNAgent as DQNAgent_mk
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.optimizers import Adam


def base_dqn_agent(model, input_shape):
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent_mk(model=model,
                      nb_actions=input_shape,
                      memory=memory,
                      nb_steps_warmup=10,
                      target_model_update=1e-2,
                      policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn
