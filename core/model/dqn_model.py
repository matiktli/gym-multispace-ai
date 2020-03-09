from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


"""
    @input_shape = observation_space_shape,
    @output_shape = action_space_shape
"""


def base_dqn_model(input_shape, output_shape):
    input_shape = (1,) + input_shape
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    print(model.summary())
    print(f'Input shape: {input_shape}')
    return model


def base_2_dqn_model(input_shape, output_shape):
    input_shape = (1,) + input_shape
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(output_shape, activation="linear"))
    return model
