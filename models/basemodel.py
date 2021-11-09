import numpy as np

from utils.io_utils import save_predictions_to_file, save_model_to_file


class BaseModel:

    def __init__(self, params, args):
        self.args = args
        self.predictions = None

        # Save all hyperparameters which are optimized
        self.params = params

        # Model has to be set by the concrete model
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        self.predictions = self.model.predict(X)
        return self.predictions

    def save_model_and_predictions(self, y_true, filename_extension=""):
        y = np.concatenate((y_true.reshape(1, -1), self.predictions.reshape(1, -1)))
        save_predictions_to_file(y, self.args, filename_extension)

        save_model_to_file(self.model, self.args, filename_extension)

    def clone(self):
        return self.__class__(self.params, self.args)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")
