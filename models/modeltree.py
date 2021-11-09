from modeltrees import ModelTreeRegressor, ModelTreeClassifier
from models.basemodel import BaseModel

from sklearn import linear_model

'''
    Define the Model Trees Model from Schufa. https://github.com/schufa-innovationlab/model-trees
'''


class ModelTree(BaseModel):
    # Define all hyperparameters which are optimized
    CRITERION = "criterion"
    DEPTH = "max_depth"
    SPLIT = "min_sample_split"

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = ModelTreeRegressor(criterion=self.params[self.CRITERION], max_depth=self.params[self.DEPTH],
                                            min_samples_split=self.params[self.SPLIT])
        elif args.objective == "classification":
            raise NotImplementedError("ModelTree is not implemented for multi-class classification yet")

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            cls.CRITERION: trial.suggest_categorical(cls.CRITERION, ['gradient', 'gradient-renorm-z']),
            cls.DEPTH: trial.suggest_int(cls.DEPTH, 2, 32, log=True),
            cls.SPLIT: trial.suggest_int(cls.SPLIT, 2, 32, log=True),
        }
        return params
