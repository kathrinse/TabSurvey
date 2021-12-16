from models.basemodel import BaseModel

import torch

import numpy as np

from .deepfm_lib.models.deepfm import DeepFM as DeepFMModel
from .deepfm_lib.inputs import SparseFeat, DenseFeat

from utils.io_utils import get_output_path


class DeepFM(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "classification":
            print("DeepFM not yet implemented for classification")
            import sys
            sys.exit()

        self.device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        # print("On Device:", self.device)

        if args.cat_idx:
            dense_features = list(set(range(args.num_features)) - set(args.cat_idx))
            fixlen_feature_columns = [SparseFeat(str(feat), args.cat_dims[idx])
                                      for idx, feat in enumerate(args.cat_idx)] + \
                                     [DenseFeat(str(feat), 1, ) for feat in dense_features]

        else:
            # Add dummy sparse feature, otherwise it will crash...
            fixlen_feature_columns = [SparseFeat("dummy", 1)] + \
                                     [DenseFeat(str(feat), 1, ) for feat in range(args.num_features)]

        self.model = DeepFMModel(linear_feature_columns=fixlen_feature_columns,
                                 dnn_feature_columns=fixlen_feature_columns,
                                 task=args.objective, device=self.device, dnn_dropout=self.params["dnn_dropout"])

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float)
        X_dict = {str(name): X[:, name] for name in range(self.args.num_features)}

        X_val = np.array(X_val, dtype=np.float)
        X_val_dict = {str(name): X_val[:, name] for name in range(self.args.num_features)}

        if self.args.objective == "binary":
            loss = "binary_crossentropy"
            metric = "binary_crossentropy"
            labels = [0, 1]
        elif self.args.objective == "regression":
            loss = "mse"
            metric = "mse"
            labels = None

        self.model.compile(optimizer=torch.optim.AdamW(self.model.parameters()),
                           loss=loss, metrics=[metric])

        # Adding dummy spare feature
        if not self.args.cat_idx:
            X_dict["dummy"] = np.zeros(X.shape[0])
            X_val_dict["dummy"] = np.zeros(X_val.shape[0])

        self.model.fit(X_dict, y, batch_size=self.params["batch_size"], epochs=self.args.epochs,
                       validation_data=(X_val_dict, y_val), labels=labels, early_stopping=True,
                       patience=self.args.early_stopping_rounds)

    def predict(self, X):
        X = np.array(X, dtype=np.float)
        X_dict = {str(name): X[:, name] for name in range(self.args.num_features)}

        # Adding dummy spare feature
        if not self.args.cat_idx:
            X_dict["dummy"] = np.zeros(X.shape[0])

        preds = self.model.predict(X_dict, batch_size=256)
        self.predictions = preds
        return self.predictions

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "dnn_dropout": trial.suggest_float("dnn_dropout", 0, 0.9),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        }
        return params

        # dnn_dropout, l2_reg_linear, l2_reg_embedding, l2_reg_dnn, dnn_hidden_units?
