from collections import deque
import numpy as np
import csv


class DataService():

    COLUMN_GENERAL_NAMES = ['frame_no', 'reward']
    COLUMN_OBJ_NAMES = ['x', 'y', 'vel_x', 'vel_y', 'mass', 'size']

    def __init__(self, path):
        self.streams = {}
        self.path = path

    def open_game_stream(self, game_counter):
        if game_counter not in self.streams:
            self.streams[game_counter] = list()
        pass

    def put_to_game_stream(self, game_counter, step_counter, obs=None, rew=None):
        if game_counter not in self.streams:
            return
        game_stream = self.streams[game_counter]
        game_stream.append({'step_no': step_counter, 'obs': obs, 'rew': rew})

    def close_game_stream(self, game_counter):
        if game_counter not in self.streams:
            return
        game_stream = self.streams[game_counter]
        self.__save_stream_to_file(game_counter, game_stream)
        del self.streams[game_counter]

    def __save_stream_to_file(self, game_counter, game_stream):
        file_path = f'{self.path}game_{game_counter}.csv'
        with open(file_path, mode='w') as csv_file:
            writer = csv.writer(
                csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for frame_data in game_stream:
                formated_data = self.__format_data(frame_data)
                if formated_data and formated_data[0] == 0:
                    writer.writerow(self.__get_labels(formated_data))
                writer.writerow(formated_data)

    def __format_data(self, step_data):
        data = []
        data.append(step_data['step_no'])
        data.append(step_data['rew'][0])
        data.extend(step_data['obs'][0].flatten())
        return data

    def __get_labels(self, formated_data):
        number_of_agents = int((len(formated_data) -
                            len(DataService.COLUMN_GENERAL_NAMES)) / len(DataService.COLUMN_OBJ_NAMES))
        result = DataService.COLUMN_GENERAL_NAMES
        for agent in range(0, number_of_agents):
            result.extend(list(map(lambda x: x + '_' + str(agent),
                                  DataService.COLUMN_OBJ_NAMES)))
        return result
