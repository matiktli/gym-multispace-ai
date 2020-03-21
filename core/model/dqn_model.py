from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Lambda
import keras
from keras.layers.convolutional import Convolution2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop


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

    # Inspired by: https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    @staticmethod
    def _object_vision_dqn_model(input_shape, output_shape, learning_rate):
        print(f'Input shape: {input_shape}')
        obs_input = Input(input_shape, name='observations')
        # actions_mask = Input((output_shape,), name='action_masks')

        normalized = Lambda(lambda x: x / 255.0)(obs_input)

        conv_1 = Convolution2D(16, 8, 8,
                               subsample=(4, 4),
                               activation='relu')(normalized)
        conv_2 = Convolution2D(32, 4, 4,
                               subsample=(2, 2),
                               activation='relu')(conv_1)
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        hidden = Dense(256, activation='relu')(conv_flattened)
        output = Dense(output_shape)(hidden)
        # filtered_output = keras.layers.merge.Multiply()([output, actions_mask])

        model = Model(input=obs_input,
                      output=output)
        optimizer = RMSprop(learning_rate=learning_rate,
                            rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')

        print(model.summary())
        print(f'Input shape: {input_shape}')
        return model
