from data_svc.main import DataLoader, DataInterpreter
import argparse

parser = argparse.ArgumentParser(description='DQN Learner')
parser.add_argument('--scenario_path', type=str,
                    help='Path to the scenario.', default='NONE')
parser.add_argument('--folder_path', type=str,
                    help='Folder with game data.')
parser.add_argument('--factor', type=int, default=50,
                    help='How accurate plot would be.')
args = parser.parse_args()

data_loader = DataLoader(args.folder_path)
data_int = DataInterpreter(data_loader.games)
data_int.plot(factor=args.factor)
