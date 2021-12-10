from models.basemodel import BaseModel
from utils.io_utils import get_output_path

from models.deepgbm_lib.main import train, predict
from models.deepgbm_lib.preprocess.preprocessing_cat import CatEncoder
from models.deepgbm_lib.preprocess.preprocessing_num import NumEncoder
import models.deepgbm_lib.config as deepgbm_config

import torch


class DeepGBM(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "classification":
            print("DeepGBM not implemented for classification!")
            import sys
            sys.exit()

        if args.cat_idx:
            cat_col = args.cat_idx
            num_col = list(set(range(args.num_features)) - set(args.cat_idx))
        else:
            cat_col = []
            num_col = list(range(args.num_features))

        self.ce = CatEncoder(cat_col, num_col)
        self.ne = NumEncoder(cat_col, num_col)

        deepgbm_config.update({'task': args.objective,
                               "epochs": args.epochs,
                               "early-stopping": args.early_stopping_rounds})
        deepgbm_config.update(**params)

        print(deepgbm_config)

    def fit(self, X, y, X_val=None, y_val=None):
        # preprocess
        train_x_cat, feature_sizes = self.ce.fit_transform(X.copy())
        test_x_cat = self.ce.transform(X_val.copy())

        train_x = self.ne.fit_transform(X)
        test_x = self.ne.transform(X_val)

        train_num = (train_x, y.reshape(-1, self.args.num_classes))
        test_num = (test_x, y_val.reshape(-1, self.args.num_classes))

        # train
        self.model, _ = train(train_num, test_num, train_x_cat.astype('int32'), test_x_cat.astype('int32'),
                              feature_sizes)

    def predict(self, X):
        self.predictions = predict(self.model, X, self.ce, self.ne).reshape(-1, 1)
        return self.predictions

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_trees": trial.suggest_categorical("n_trees", [100, 200]),
            "maxleaf": trial.suggest_categorical("maxleaf", [64, 128]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "loss_de": trial.suggest_int("loss_de", 2, 10),
            "loss_dr": trial.suggest_categorical("loss_dr", [0.7, 0.9])
        }
        return params
