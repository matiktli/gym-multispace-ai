from os.path import isfile, join
import csv
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class GameData():

    def __init__(self, game_no, data, columns=None):
        self.game_no = int(game_no)
        self.data = data
        self.columns = columns

    def get_reward_sum(self):
        result = 0
        for row in self.data:
            result += float(row[1])
        return result



class DataLoader():

    def __init__(self, path):
        self.folder_path = path
        self.games, self.gifs = self.__find_data_in_folder()
        print(f'Loaded: {len(self.games)} games and {len(self.gifs)} gifs')

    def __find_data_in_folder(self):
        game_datas = []
        gif_datas = []

        onlyfiles = [f for f in listdir(
            self.folder_path) if isfile(join(self.folder_path, f))]
        for f in onlyfiles:
            if f[-3:] == 'csv':
                game_no = f.split('_')[len(f.split('_'))-1][:-4]
                game_data = self.__load_game(game_no)
                game_datas.append(game_data)
            elif f[-3:] == 'gif':
                continue

        game_datas.sort(key=lambda x: x.game_no)
        return game_datas, gif_datas

    def __load_game(self, game_no) -> GameData:
        game_raw_data = []
        with open(self.folder_path + f'game_{game_no}.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                column_names = []
                if line_count == 0:
                    column_names = row
                else:
                    game_raw_data.append(row)
                line_count += 1
        return GameData(game_no, game_raw_data, column_names)

    def __load_games(self, game_nos):
        game_datas = []
        for game_no in game_nos:
            game_datas.append(self.__load_game(game_no))
        return game_datas


class DataInterpreter():

    def __init__(self, games):
        games.sort(key=lambda x: x.game_no)
        self.games = games
        self.games_len = len(games)

    def plot(self, factor=100, experiment_desc={}):
        game_nos = list()
        game_rews = list()

        tmp_rews = 0
        for i, game_data in enumerate(self.games):
            tmp_rews += game_data.get_reward_sum()
            if i % factor == 0 and i != 0:
                game_nos.append(game_data.game_no)
                game_rews.append(tmp_rews/factor)
                tmp_rews = 0

        plt.plot(game_nos, game_rews, 'r')
        plt.xlabel('Episode no.')
        plt.ylabel(f'Avg reward from {factor} games')
        plt.suptitle(
            f'Experiment with {experiment_desc["NO_STEPS_PER_GAME"]} steps per game')

        trend = np.polyfit(game_nos, game_rews, 1)
        trend_func = np.poly1d(trend)
        plt.plot(game_nos, trend_func(game_nos), "g--")

        main_func_info = mpatches.Patch(color='red',
            label=f'DQN Agent rewards')
        reward_trend_info = mpatches.Patch(color='green',
                                           label=f'Reward(progress) trend')
        agent_mem_size_info = mpatches.Patch(color='grey',
            label=f'Mem size: {experiment_desc["AGENT_MEMORY_SIZE"]}')
        agent_batch_size_info = mpatches.Patch(color='grey',
            label=f'Batch size: {experiment_desc["AGENT_BATCH_SIZE"]}')
        agent_learning_rate_info = mpatches.Patch(color='grey',
            label=f'Learning rate: {experiment_desc["AGENT_LEARNING_RATE"]}')
        activation_info = mpatches.Patch(color='grey',
                                         label=f'Activation: {experiment_desc["ACTIVATION"] if "ACTIVATION" in experiment_desc  else "RELU"}')
        network_layers = mpatches.Patch(color='grey',
                                        label=f'Layers: {experiment_desc["NETWORK"] if "NETWORK" in experiment_desc else "24x24"}')

        plt.legend(handles=[main_func_info, reward_trend_info, agent_mem_size_info,
                            agent_batch_size_info, agent_learning_rate_info, activation_info, network_layers])
        plt.show()

    def __get_avarage_reward_every_x_games(self, factor):
        result_avgs = []
        tmp_avg = 0
        for i, game_data in enumerate(self.games):
            tmp_avg += game_data.get_reward_sum()
            if i % factor == 0 and i != 0:
                result_avgs.append(tmp_avg / factor)
                tmp_avg = 0
        return result_avgs
