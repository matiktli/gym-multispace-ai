from model.base_model import BaseModel
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten

import tensorflow as tf


# From: https://github.com/hridayns/Research-Project-on-Reinforcement-learning/blob/master/Atari/models/DDQN.py
# Additional sources:  https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f
class DDQN(BaseModel):

    def __init__(self, input_shape, output_shape, learning_rate):
        BaseModel.__init__(self, input_shape, output_shape)
        self.learning_rate = learning_rate

        obs_input = Input(input_shape, name='observations')

        conv_1 = Convolution2D(filters=32,
                               kernel_size=(8, 8),
                               strides=(4, 4),
                               padding='valid',
                               activation='relu')(obs_input)
        conv_2 = Convolution2D(filters=64,
                               kernel_size=(4, 4),
                               strides=(2, 2),
                               padding='valid',
                               activation='relu')(conv_1)
        conv_3 = Convolution2D(filters=64,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='valid',
                               activation='relu')(conv_2)

        conv_flattened = Flatten()(conv_3)

        hidden = Dense(512, activation='linear')(conv_flattened)

        output = Dense(output_shape)(hidden)

        self.model = Model(input=obs_input,
                           output=output)
        # Compile model
        optimizer = Adam(lr=self.learning_rate, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer=optimizer,
                           loss=tf.losses.huber_loss,
                           metrics=['acc'])
        print(self.model.summary())
