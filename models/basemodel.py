import numpy as np
import typing as tp

import optuna

from utils.io_utils import save_predictions_to_file, save_model_to_file


class BaseModel:
    """Basic interface for all models.

    All implemented models should inherit from this base class to provide a common interface.
    At least they have to extend the init method defining the model and the define_trial_parameters method
    specifying the hyperparameters.

    Methods
    -------
    __init__(params, args):
        Defines the model architecture, depending on the hyperparameters (params) and command line arguments (args).
    fit(X, y, X_val=None, y_val=None)
        Trains the model on the trainings dataset (X, y). Validates the training process and uses early stopping
        if a validation set (X_val, y_val) is provided. Returns the loss history and validation loss history.
    predict(X)
        Predicts the labels of the test dataset (X). Saves and returns the predictions.
    define_trial_parameters(trial, args)
        Returns a possible hyperparameter configuration. This method is necessary for the automated hyperparameter
        optimization.
    save_model_and_prediction(y_true, filename_extension="")
        Saves the current state of the model and the predictions and true labels of the test dataset.
    save_model(filename_extension="")
        Saves the current state of the model.
    save_predictions(y_true, filename_extension="")
        Saves the predictions and true labels of the test dataset.
    clone()
        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.
    """

    def __init__(self, params: tp.Dict, args):
        """Defines the model architecture.

        After calling this method, self.model has to be defined.

        :param params: possible hyperparameter configuration, model architecture depends on this
        :param args: command line arguments containing all important information about the dataset and training process
        """
        self.args = args
        self.params = params

        # Model definition has to be implemented by the concrete model
        self.model = None

        # Create a placeholder for the predictions on the test dataset
        self.predictions = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: tp.Union[None, np.ndarray] = None,
            y_val: tp.Union[None, np.ndarray] = None) -> tp.Tuple[list, list]:
        """Trains the model.

        The training is done on the trainings dataset (X, y). If a validation set (X_val, y_val) is provided,
        the model state is validated during the training, to allow early stopping.

        Returns the loss history and validation loss history if the loss and validation loss development during
        the training are logged. Otherwise empty lists are returned.

        :param X: trainings data
        :param y: labels of trainings data
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: loss history, validation loss history
        """

        self.model.fit(X, y)

        # Should return loss history and validation loss history
        return [], []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained model.

        The labels of the test dataset (X) are predicted. The outcomes are saved for later (self.predictions) and
        also returned.

        If the task is binary or multi-class classification, not the concrete class but the probability distribution
        over the classes is returned.

        :param X: test data
        :return: predicted labels of test data
        """
        if self.args.objective == "regression":
            self.predictions = self.model.predict(X)
        elif self.args.objective == "classification" or self.args.objective == "binary":
            self.predictions = self.model.predict_proba(X)

        return self.predictions

    def save_model_and_predictions(self, y_true: np.ndarray, filename_extension=""):
        """Saves the current state of the model and the predictions and true labels of the test dataset.

        :param y_true: true labels of the test data
        :param filename_extension: (optional) additions to the filenames
        """
        self.save_predictions(y_true, filename_extension)
        self.save_model(filename_extension)

    def clone(self):
        """Clone the model.

        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.

        :return: Copy of the current model without trained parameters
        """
        return self.__class__(self.params, self.args)

    @classmethod
    def define_trial_parameters(cls, trial: optuna.Trial, args) -> tp.Dict:
        """Define the ranges of the hyperparameters

        Returns a possible hyperparameter configuration. This method is necessary for the automated hyperparameter
        optimization. All hyperparameter that should be optimized and their ranges are specified here.
        For more information see: https://optuna.org/

        :param trial: Trial class instance generated by the optuna library.
        :param args: Command line arguments containing all important information about the dataset
        :return: Hyperparameter configuration
        """

        raise NotImplementedError("This method has to be implemented by the sub class")

    def save_model(self, filename_extension=""):
        """Saves the current state of the model.

        Saves the model using pickle. Override this method if model should be saved in a different format.

        :param filename_extension: true labels of the test data
        """
        save_model_to_file(self.model, self.args, filename_extension)

    def save_predictions(self, y_true: np.ndarray, filename_extension=""):
        """Saves the predictions and true labels of the test dataset.

        Saves the predictions and the truth values together in a npy file.

        :param y_true: true labels of the test data
        :param filename_extension: true labels of the test data
        """
        if self.args.objective == "regression":
            # Save array where [:,0] is the truth and [:,1] the prediction
            y = np.concatenate((y_true.reshape(-1, 1), self.predictions.reshape(-1, 1)), axis=1)
        else:
            # Save array where [:,0] is the truth and [:,1:] are the prediction probabilities
            y = np.concatenate((y_true.reshape(-1, 1), self.predictions), axis=1)

        save_predictions_to_file(y, self.args, filename_extension)
