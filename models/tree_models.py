import xgboost as xgb
import catboost as cat
import lightgbm as lgb

import numpy as np

from models.basemodel import BaseModel

'''
    Define all Gradient Boosting Decision Tree Models:
    XGBoost, CatBoost, LightGBM
'''

'''
    XGBoost
'''


class XGBoost(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.params["verbosity"] = 1

        if args.objective == "regression":
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"
        elif args.objective == "classification":
            self.params["objective"] = "multi:softmax"
            self.params["num_class"] = args.num_classes
            self.params["eval_metric"] = "mlogloss"

    def fit(self, X, y, X_val=None, y_val=None):
        train = xgb.DMatrix(X, label=y)
        val = xgb.DMatrix(X_val, label=y_val)
        eval_list = [(val, "eval")]
        self.model = xgb.train(self.params, train, num_boost_round=self.args.epochs, evals=eval_list,
                               early_stopping_rounds=self.args.early_stopping_rounds,
                               verbose_eval=self.args.logging_period)

    def predict(self, X):
        test = xgb.DMatrix(X)
        self.predictions = self.model.predict(test)  # get probabilities!
        return self.predictions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True)
        }
        return params


'''
    CatBoost
'''


class CatBoost(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.params["iterations"] = self.args.epochs
        self.params["od_type"] = "Iter"
        self.params["od_wait"] = self.args.early_stopping_rounds
        self.params["verbose"] = self.args.logging_period

        if args.objective == "regression":
            self.model = cat.CatBoostRegressor(**self.params)
        elif args.objective == "classification":
            self.model = cat.CatBoostClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y, eval_set=(X_val, y_val))

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
        }
        return params


'''
    LightGBM
'''


class LightGBM(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.params["verbosity"] = -1

        if args.objective == "regression":
            self.params["objective"] = "regression"
            self.params["metric"] = "mse"
        elif args.objective == "classification":
            self.params["objective"] = "multiclass"
            self.params["num_class"] = args.num_classes
            self.params["metric"] = "multiclass"

    def fit(self, X, y, X_val=None, y_val=None):
        train = lgb.Dataset(X, label=y)
        val = lgb.Dataset(X_val, label=y_val)
        self.model = lgb.train(self.params, train, num_boost_round=self.args.epochs, valid_sets=[val],
                               valid_names=["eval"], callbacks=[lgb.early_stopping(self.args.early_stopping_rounds),
                                                                lgb.log_evaluation(self.args.logging_period)])

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
        return params
