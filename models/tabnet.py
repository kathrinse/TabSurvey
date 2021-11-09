from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

import numpy as np

from models.basemodel import BaseModel


class TabNet(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        # Paper recommends to be n_d and n_a the same
        self.params["n_a"] = self.params["n_d"]

        if args.objective == "regression":
            self.model = TabNetRegressor(**self.params)
        elif args.objective == "classification":
            self.model = TabNetMultiTaskClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)
        self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_name=["eval"], eval_metric=["rmse"],
                       max_epochs=self.args.epochs, patience=self.args.early_stopping_rounds, batch_size=128)

    def predict(self, X):
        return super().predict(X)
        #self.predictions = self.model.predict_proba(X)
        #self.predictions = np.argmax(self.predictions, axis=2).transpose()
        #return self.predictions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "n_independent": trial.suggest_int("n_independent", 1, 5),
            "n_shared": trial.suggest_int("n_shared", 1, 5),
            "momentum": trial.suggest_float("momentum", 0.001, 0.4, log=True)
        }
        return params
