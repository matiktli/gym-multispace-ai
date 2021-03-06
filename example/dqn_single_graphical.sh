#!/bin/sh

SCENARIO_PATH='../core/scenario/single_scenario_graphic.py'
NO_GAMES=320
NO_STEPS_PER_GAME=80
PATH_TO_SAVE='.test-results/dqn/single__v0/'

AGENT_EXPLORATION_RATE=0.2
AGENT_MEMORY_SIZE=1000000
AGENT_BATCH_SIZE=32
AGENT_LEARNING_RATE=0.001

python ../core/dqn_learner_wrapper.py \
    --scenario_path $SCENARIO_PATH \
    --no_games $NO_GAMES \
    --no_steps_per_game $NO_STEPS_PER_GAME \
    --path_to_save_assets $PATH_TO_SAVE \
    --agent_exploration_rate $AGENT_EXPLORATION_RATE \
    --agent_memory_size $AGENT_MEMORY_SIZE \
    --agent_batch_size $AGENT_BATCH_SIZE \
    --agent_learning_rate $AGENT_LEARNING_RATE