from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from models.basemodel import BaseModel


class TabNet(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        # Paper recommends to be n_d and n_a the same
        self.params["n_a"] = self.params["n_d"]

        # Delete batch size from params, as TabNet can not get it as an input
        self.tabnet_params = self.params.copy()
        del self.tabnet_params["batch_size"]

        if args.objective == "regression":
            self.model = TabNetRegressor(**self.tabnet_params)
        elif args.objective == "classification":
            self.model = TabNetClassifier(**self.tabnet_params)

    def fit(self, X, y, X_val=None, y_val=None):
        if self.args.objective == "regression":
            y, y_val = y.reshape(-1, 1), y_val.reshape(-1, 1)
            metric = ["rmse"]
        elif self.args.objective == "classification":
            metric = ["logloss"]

        print("Batch size", self.params["batch_size"])

        self.model.fit(X, y, eval_set=[(X_val, y_val)], eval_name=["eval"], eval_metric=metric,
                       max_epochs=self.args.epochs, patience=self.args.early_stopping_rounds,
                       batch_size=self.params["batch_size"])

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "cat_emb_dim": trial.suggest_int("cat_emb_dim", 1, 3),
            "n_independent": trial.suggest_int("n_independent", 1, 5),
            "n_shared": trial.suggest_int("n_shared", 1, 5),
            "momentum": trial.suggest_float("momentum", 0.001, 0.4, log=True),
            "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        }
        return params
