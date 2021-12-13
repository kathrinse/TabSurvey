import numpy as np
import torch.utils.data

from models.basemodel import BaseModel

from utils.io_utils import get_output_path

from nam.config import defaults
from nam.models import NAM as NAMModel
from nam.trainer import LitNAM

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


class NAM(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if self.args.objective == "classification":
            print("NAM not implemented for multi-class classification yet.")
            import sys
            sys.exit(0)

        regression = True if self.args.objective == "regression" else False
        device = "gpu" if self.args.use_gpu and torch.cuda.is_available() else 'cpu'

        self.config = defaults()
        self.config.update(regression=regression, num_epochs=self.args.epochs,
                           early_stopping_patience=self.args.early_stopping_rounds,
                           device=device, **self.params)

        print(self.config)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float)
        X_val = np.array(X_val, dtype=np.float)

        dataset = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        num_unique_vals = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
        num_units = [min(self.config.num_basis_functions, i * self.config.units_multiplier) for i in num_unique_vals]

        self.model = NAMModel(
            config=self.config,
            name="NAM",
            num_inputs=len(dataset[0][0]),
            num_units=num_units,
        )

        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False,
                                                num_workers=2)

        # Folder hack
        tb_logger = TensorBoardLogger(save_dir=self.config.logdir, name=f'{self.model.name}', version=f'0')

        checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                              monitor='val_loss',
                                              save_top_k=self.config.save_top_k,
                                              mode='min')

        litmodel = LitNAM(self.config, self.model)
        trainer = pl.Trainer(logger=tb_logger,  max_epochs=1,  # self.config.num_epochs
                             enable_checkpointing=checkpoint_callback,  # checkpoint_callback
                             callbacks=[EarlyStopping(monitor='val_loss', patience=self.args.early_stopping_rounds)])
        trainer.fit(litmodel, train_dataloaders=trainloader, val_dataloaders=valloader)

    def predict(self, X):
        X = np.array(X, dtype=np.float)
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X).float())
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()

        self.predictions = []
        with torch.no_grad():
            for batch_X in testloader:
                preds = self.model(batch_X[0])[0]  # .to(self.device)

                if self.args.objective == "binary_classification":
                    preds = torch.sigmoid(preds)

                self.predictions.append(preds)  # .detach().cpu().numpy()

        self.predictions = np.concatenate(self.predictions).reshape(-1, 1)
        return self.predictions

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        torch.save(self.model.state_dict(), filename)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            'lr': trial.suggest_float('lr', 0.001, 0.1, log=True),
            'output_regularization': trial.suggest_float('output_regularization', 0.001, 0.1, log=True),
            # 'l2_regularization': trial.suggest_float('l2_regularization', 0.000001, 0.0001, log=True),
            'dropout': trial.suggest_float('dropout', 0, 0.9),
            'feature_dropout': trial.suggest_float('feature_dropout', 0, 0.2),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        }
        return params

# l2_regularization=0.1,  hidden_sizes=[], activation='exu'
