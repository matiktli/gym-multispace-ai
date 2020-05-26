#!/bin/sh

SCENARIO_PATH='../core/scenario/single_simple_push_scenario_graphic.py'
NO_GAMES=15000
NO_STEPS_PER_GAME=500
PATH_TO_SAVE='.test-results/ddqn/push__v0/'

AGENT_EXPLORATION_RATE=0.1
AGENT_MEMORY_SIZE=10000
AGENT_BATCH_SIZE=32
AGENT_LEARNING_RATE=0.0001

SAVE_REPLAY_EVERY_N_GAMES=100
SAVE_WEIGHTS_EVERY_N_GAMES=500

python ../core/ddqn_learner_wrapper.py \
    --scenario_path $SCENARIO_PATH \
    --no_games $NO_GAMES \
    --no_steps_per_game $NO_STEPS_PER_GAME \
    --path_to_save_assets $PATH_TO_SAVE \
    --save_replay_every_n_games $SAVE_REPLAY_EVERY_N_GAMES \
    --save_weights_every_n_games $SAVE_WEIGHTS_EVERY_N_GAMES \
    --agent_exploration_rate $AGENT_EXPLORATION_RATE \
    --agent_memory_size $AGENT_MEMORY_SIZE \
    --agent_batch_size $AGENT_BATCH_SIZE \
    --agent_learning_rate $AGENT_LEARNING_RATE