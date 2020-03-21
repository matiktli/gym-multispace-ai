#!/bin/sh

SCENARIO_PATH='../core/scenario/single_scenario_graphic.py'
NO_GAMES=700
NO_STEPS_PER_GAME=150
PATH_TO_SAVE='.test-results/v2/single_g_0/'

AGENT_EXPLORATION_RATE=0.2
AGENT_MEMORY_SIZE=1000000
AGENT_BATCH_SIZE=25
AGENT_LEARNING_RATE=0.00025

python ../core/dqn_learner_wrapper.py \
    --scenario_path $SCENARIO_PATH \
    --no_games $NO_GAMES \
    --no_steps_per_game $NO_STEPS_PER_GAME \
    --path_to_save_assets $PATH_TO_SAVE \
    --agent_exploration_rate $AGENT_EXPLORATION_RATE \
    --agent_memory_size $AGENT_MEMORY_SIZE \
    --agent_batch_size $AGENT_BATCH_SIZE \
    --agent_learning_rate $AGENT_LEARNING_RATE \
    --model_name 'object_vision_dqn_model'