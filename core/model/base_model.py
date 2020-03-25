from keras.models import Model

class BaseModel():

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model: Model = None

    def save_model_weights(self, path):
        assert self.model
        self.model.save_weights(path)

    def load_model_weights(self, path):
        assert self.model
        self.model.load_weights(path)
