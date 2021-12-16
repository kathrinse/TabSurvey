from models.basemodel import BaseModel

from stg import STG as STGModel

import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from utils.io_utils import get_output_path


class STG(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

        task = "classification" if self.args.objective == "binary" else self.args.objective
        out_dim = 2 if self.args.objective == "binary" else self.args.num_classes

        self.model = STGModel(task_type=task, input_dim=self.args.num_features,
                              output_dim=out_dim, activation='tanh', sigma=0.5,
                              optimizer='SGD',  feature_selection=True, random_state=1, device=self.device,
                              **self.params)  # batch_size=128, hidden_dims=[500, 50, 10],

    def fit(self, X, y, X_val=None, y_val=None):
        X, X_val = X.astype("float"), X_val.astype("float")

        if self.args.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)

        self.model.fit(X, y, nr_epochs=self.args.epochs, valid_X=X_val, valid_y=y_val,
                       print_interval=self.args.logging_period)  # early_stop=True self.args.epochs

    def predict(self, X):
        # This great implementation doesn't want the predictions to be computed on the GPU...
        self.model._model.to("cpu")
        pred = self.model.predict(X).reshape(-1, 1)

        # Predict only gives the final predictions and not the needed probabilities!
        # Making this for the scorer to work anyway. But should be changed...
        if self.args.objective == "classification":
            categories = np.arange(self.args.num_classes).reshape(1, -1)
            oe = OneHotEncoder(categories=categories)
            pred = oe.fit_transform(pred).toarray()

        self.predictions = pred
        return self.predictions

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        self.model.save_checkpoint(filename)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "lam": trial.suggest_float("lam", 1e-3, 10, log=True),
            # Change also the number and size of the hidden_dims?
            "hidden_dims": trial.suggest_categorical("hidden_dims", [[500, 50, 10], [60, 20],
                                                                     [500, 500, 10], [500, 400, 20]]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        }
        return params
