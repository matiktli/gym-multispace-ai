from data_svc.main import DataLoader, DataInterpreter
import argparse
from os.path import isfile, join
import yaml


def load_experiment_data(folder_path):
    with open(folder_path + f'config.yaml') as experiment_file:
        data = yaml.load(experiment_file, Loader=yaml.FullLoader)
        return data

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
experiment_data = load_experiment_data(args.folder_path)
data_int.plot(factor=args.factor, experiment_desc=experiment_data) 
