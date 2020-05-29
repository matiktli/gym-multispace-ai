#!/bin/sh

SCENARIO_PATH='../core/scenario/single_scenario.py'
NO_GAMES=1500
NO_STEPS_PER_GAME=500
PATH_TO_SAVE='.test-results/v8/xxx/'

AGENT_EXPLORATION_RATE=0.2 
AGENT_MEMORY_SIZE=100000
AGENT_BATCH_SIZE=32
AGENT_LEARNING_RATE=0.001

RENDER_RATE=50

python ../core/dqn_learner_wrapper.py \
    --scenario_path $SCENARIO_PATH \
    --no_games $NO_GAMES \
    --no_steps_per_game $NO_STEPS_PER_GAME \
    --path_to_save_assets $PATH_TO_SAVE \
    --agent_exploration_rate $AGENT_EXPLORATION_RATE \
    --agent_memory_size $AGENT_MEMORY_SIZE \
    --agent_batch_size $AGENT_BATCH_SIZE \
    --agent_learning_rate $AGENT_LEARNING_RATE \
    --render_every_n_games $RENDER_RATE