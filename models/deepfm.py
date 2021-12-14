from models.basemodel import BaseModel

import torch
import tensorflow as tf

import numpy as np

from deepctr_torch.models.deepfm import DeepFM as DeepFMModel
from deepctr_torch.inputs import SparseFeat, DenseFeat

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

        task = "binary" if args.objective == "binary_classification" else args.objective

        self.model = DeepFMModel(linear_feature_columns=fixlen_feature_columns,
                                 dnn_feature_columns=fixlen_feature_columns,
                                 task=task, device=self.device, **self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float)
        X_dict = {str(name): X[:, name] for name in range(self.args.num_features)}

        X_val = np.array(X_val, dtype=np.float)
        X_val_dict = {str(name): X_val[:, name] for name in range(self.args.num_features)}

        if self.args.objective == "binary_classification":
            loss = "binary_crossentropy"
            metric = "binary_crossentropy"
        elif self.args.objective == "regression":
            loss = "mse"
            metric = "mse"

        self.model.compile(optimizer=torch.optim.AdamW(self.model.parameters()),
                           loss=loss, metrics=[metric])

        # Adding dummy spare feature
        if not self.args.cat_idx:
            X_dict["dummy"] = np.zeros(X.shape[0])
            X_val_dict["dummy"] = np.zeros(X_val.shape[0])

        self.model.fit(X_dict, y, batch_size=128, epochs=self.args.epochs, validation_data=(X_val_dict, y_val),
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_" + metric, verbose=1,
                                                                   patience=self.args.early_stopping_rounds)])

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
        }
        return params

        # dnn_dropout, l2_reg_linear, l2_reg_embedding, l2_reg_dnn, dnn_hidden_units?
