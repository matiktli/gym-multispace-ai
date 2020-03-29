#!/bin/sh

SCENARIO_PATH='../core/scenario/single_scenario_graphic.py'
NO_GAMES=2000
NO_STEPS_PER_GAME=150
PATH_TO_SAVE='.test-results/ddqn/single__v0/'

AGENT_EXPLORATION_RATE=0.1
AGENT_MEMORY_SIZE=10000
AGENT_BATCH_SIZE=32
AGENT_LEARNING_RATE=0.0001

python ../core/ddqn_learner_wrapper.py \
    --scenario_path $SCENARIO_PATH \
    --no_games $NO_GAMES \
    --no_steps_per_game $NO_STEPS_PER_GAME \
    --path_to_save_assets $PATH_TO_SAVE \
    --agent_exploration_rate $AGENT_EXPLORATION_RATE \
    --agent_memory_size $AGENT_MEMORY_SIZE \
    --agent_batch_size $AGENT_BATCH_SIZE \
    --agent_learning_rate $AGENT_LEARNING_RATE