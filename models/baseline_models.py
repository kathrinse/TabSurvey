from sklearn import linear_model, neighbors, svm, tree, ensemble

from models.basemodel import BaseModel

'''
    Define all Models implemented by the Sklearn library: 
    Linear Model, KNN, SVM, Decision Tree, Random Forest
'''

'''
    Linear Model - Ordinary least squares Linear Regression / Logistic Regression
    
    Takes no hyperparameters
'''


class LinearModel(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = linear_model.LinearRegression(max_iter=args.epochs, n_jobs=-1)
        elif args.objective == "classification":
            self.model = linear_model.LogisticRegression(max_iter=args.epochs, multi_class="multinomial", n_jobs=-1)
        elif args.objective == "binary_classification":
            self.model = linear_model.LogisticRegression(max_iter=args.epochs, n_jobs=-1)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params


'''
    K-Neighbors Regressor - Regression/Classification based on k-nearest neighbors
    
    Takes number of neighbors as hyperparameters
'''


class KNN(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = neighbors.KNeighborsRegressor(n_neighbors=params["n_neighbors"], n_jobs=-1)
        elif args.objective == "classification" or args.objective == "binary_classification":
            self.model = neighbors.KNeighborsClassifier(n_neighbors=params["n_neighbors"], n_jobs=-1)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_neighbors": trial.suggest_categorical("n_neighbors", list(range(3, 42, 2)))
        }
        return params


'''
    Support Vector Machines - Epsilon-Support Vector Regression / C-Support Vector Classification
    
    Takes the regularization parameter as hyperparameter
'''


class SVM(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = svm.SVR(C=params["C"])
        elif args.objective == "classification" or args.objective == "binary_classification":
            self.model = svm.SVC(C=params["C"], probability=True)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "C": trial.suggest_float("C", 1e-10, 1e10, log=True)
        }
        return params


'''
    Decision Tree - Decision Tree Regressor/Classifier
    
    Takes the maximum depth of the tree as hyperparameter
'''


class DecisionTree(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = tree.DecisionTreeRegressor(max_depth=params["max_depth"])
        elif args.objective == "classification" or args.objective == "binary_classification":
            self.model = tree.DecisionTreeClassifier(max_depth=params["max_depth"])

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True)
        }
        return params


'''
    Random Forest - Random Forest Regressor/Classifier
    
    Takes the maximum depth of the trees and the number of estimators as hyperparameter
'''


class RandomForest(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = ensemble.RandomForestRegressor(n_estimators=params["n_estimators"],
                                                        max_depth=params["max_depth"], n_jobs=-1)
        elif args.objective == "classification" or args.objective == "binary_classification":
            self.model = ensemble.RandomForestClassifier(n_estimators=params["n_estimators"],
                                                         max_depth=params["max_depth"], n_jobs=-1)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 5, 100, log=True)
        }
        return params
