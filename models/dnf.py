from models.basemodel import BaseModel
from utils.io_utils import get_output_path

import numpy as np
import os

import tensorflow as tf
from sklearn.metrics import log_loss

from models.dnf_lib.DNFNet.ModelHandler import ModelHandler, EarlyStopping, ReduceLRonPlateau
from models.dnf_lib.config import get_config
from models.dnf_lib.Utils.NumpyGenerator import NumpyGenerator
from models.dnf_lib.Utils.experiment_utils import create_model, create_experiment_directory


class DNFNet(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            print("DNF-Net not implemented for regression tasks.")
            import sys
            sys.exit(0)

        self.config, self.score_config = get_config(args.dataset, 'DNFNet')

        self.config.update({
            'input_dim': args.num_features,
            'output_dim': args.num_classes,
            'translate_label_to_one_hot': True if args.objective == "classification" else False,
            **self.params
        })

        self.score_config.update({
            'score_metric': log_loss,
            'score_increases': False,
        })

        print(self.config)
        print(self.score_config)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['GPU']
        tf.reset_default_graph()
        tf.random.set_random_seed(seed=self.config['random_seed'])
        np.random.seed(seed=self.config['random_seed'])
        self.score_metric = self.score_config['score_metric']

        self.experiment_dir, self.weights_dir, self.logs_dir = create_experiment_directory(self.config,
                                                                                           return_sub_dirs=True)
        self.model = create_model(self.config, models_module_name=self.config['models_module_name'])

        print("Weights dir", self.weights_dir)

        print(self.model)

        self.model_handler = None

    def fit(self, X, y, X_val=None, y_val=None):
        print("Fitting...")

        train_generator = NumpyGenerator(X, y, self.config['output_dim'],
                                         self.config['batch_size'],
                                         translate_label_to_one_hot=self.config['translate_label_to_one_hot'],
                                         copy_dataset=False)
        val_generator = NumpyGenerator(X_val, y_val, self.config['output_dim'],
                                       self.config['batch_size'],
                                       translate_label_to_one_hot=self.config['translate_label_to_one_hot'],
                                       copy_dataset=False)

        early_stopping = EarlyStopping(patience=self.config['early_stopping_patience'],
                                       score_increases=self.score_config['score_increases'], monitor='val_score')
        lr_scheduler = ReduceLRonPlateau(initilal_lr=self.config['initial_lr'], factor=self.config['lr_decay_factor'],
                                         patience=self.config['lr_patience'], min_lr=self.config['min_lr'],
                                         monitor='train_loss')

        self.model_handler = ModelHandler(config=self.config, model=self.model,
                                          callbacks=[lr_scheduler, early_stopping],
                                          target_dir=self.weights_dir, logs_dir=self.logs_dir)
        self.model_handler.build_graph(phase='train')
        val_score, epoch = self.model_handler.train(train_generator, val_generator, score_metric=self.score_metric,
                                                    score_increases=self.score_config['score_increases'])

        print("Val score:", val_score)
        print("Epoch", epoch)

    def predict(self, X):
        print("Predicting...")

        assert os.path.exists(self.weights_dir + '/model_weights.ckpt.meta')

        if os.path.exists(self.weights_dir + '/model_weights.ckpt.meta'):
            print('Loading weights')
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.model_handler.sess, self.weights_dir + '/model_weights.ckpt')

        # create sorted target array for Generator to work and to sort the predictions afterwards
        y = np.array(list(range(X.shape[0])))
        if self.args.objective == "classification":
            r = np.zeros((X.shape[0], self.args.num_classes - 1))
            y = np.concatenate([y.reshape(-1, 1), r], axis=1)

        test_generator = NumpyGenerator(X, y, self.config['output_dim'], self.config['batch_size'],
                                        translate_label_to_one_hot=False, #self.config['translate_label_to_one_hot']
                                        copy_dataset=False)

        y, y_pred = self.model_handler.test(test_generator)

        # Sort the predictions!
        y_pred_sorted = [y_pred for _, y_pred in sorted(zip(y[:, 0], y_pred))]

        self.predictions = np.concatenate(y_pred_sorted, axis=0).reshape(-1, self.args.num_classes)
        return self.predictions

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="")

        # Model already saved, only coping to the right location
        import shutil

        if os.path.exists(filename):
            shutil.rmtree(filename)

        shutil.copytree(self.weights_dir, filename)  # , dirs_exist_ok=True

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            'initial_lr': trial.suggest_float('inital_lr', 5e-3, 5e-1, log=True),
            'lr_decay_factor': trial.suggest_float('lr_decay_factor', 0.3, 0.7),
            'lr_patience': trial.suggest_int('lr_patience', 5, 15),
            'min_lr': trial.suggest_float('min_lr', 1e-7, 1e-5)
        }
        return params
