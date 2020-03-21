from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam


"""
    @input_shape = observation_space_shape,
    @output_shape = action_space_shape
"""


def load_model_and_compile(model_name, input_shape, output_shape, learning_rate):
    models = Models()
    model_fnc = getattr(models, '_' + model_name)
    return model_fnc(input_shape, output_shape, learning_rate)


class Models():

    def __init__(self):
        pass

    @staticmethod
    def _simple_dqn_model(input_shape, output_shape, learning_rate):
        print(f'Input shape: {input_shape}')
        input_shape = (1,) + input_shape
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=learning_rate))
        print(f'Input shape: {input_shape}')
        return model

    @staticmethod
    def _base_dqn_model(input_shape, output_shape, learning_rate):
        print(f'Input shape: {input_shape}')
        input_shape = (1,) + input_shape
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=learning_rate))
        print(model.summary())
        print(f'Input shape: {input_shape}')
        return model

    @staticmethod
    def _object_vision_dqn_model(input_shape, output_shape, learning_rate):
        print(f'Input shape: {input_shape}')
        input_shape = (1,) + input_shape
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        # TODO define graphical model
        model.add(Dense(output_shape, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(
            lr=learning_rate))
        print(model.summary())
        print(f'Input shape: {input_shape}')
        return model
