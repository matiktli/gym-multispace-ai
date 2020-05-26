from model.base_model import BaseModel
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Input, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten


# Inspired by: https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
class DQN(BaseModel):

    def __init__(self, input_shape, output_shape, learning_rate):
        BaseModel.__init__(self, input_shape, output_shape)
        self.learning_rate = learning_rate
        print(f'Input shape: {input_shape}')
        obs_input = Input(input_shape, name='observations')
        normalized = Lambda(lambda x: x / 255.0)(obs_input)

        conv_1 = Convolution2D(16, 8, 8,
                               subsample=(4, 4),
                               activation='relu')(normalized)
        conv_2 = Convolution2D(32, 4, 4,
                               subsample=(2, 2),
                               activation='relu')(conv_1)
        conv_flattened = Flatten()(conv_2)
        hidden = Dense(256, activation='linear')(conv_flattened)
        output = Dense(output_shape)(hidden)
        # filtered_output = keras.layers.merge.Multiply()([output, actions_mask])

        self.model = Model(input=obs_input,
                           output=output)

        # Compile model
        optimizer = RMSprop(learning_rate=self.learning_rate, epsilon=0.01)
        self.model.compile(optimizer=optimizer,
                           loss='mse',
                           metrics=['acc'])
        print(self.model.summary())
