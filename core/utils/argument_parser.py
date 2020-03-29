import argparse


def get_arguments_for_ddqn():
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

    # Collect DDQN solver properties
    parser.add_argument('--agent_exploration_rate', type=float,
                        help='Agent`s exploration rate.')
    parser.add_argument('--agent_memory_size', type=int,
                        help='Agent`s memory size.')
    parser.add_argument('--agent_batch_size', type=int,
                        help='Agent`s batch size.')
    parser.add_argument('--agent_learning_rate', type=float,
                        help='Agent`s learning rate.')
    return parser.parse_args()
