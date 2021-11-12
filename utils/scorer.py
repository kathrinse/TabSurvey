from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score
import numpy as np


def get_scorer(args):
    if args.objective == "regression":
        return RegScorer()
    elif args.objective == "classification":
        return ClassScorer()
    else:
        raise NotImplementedError("Not Scorer for \"" + args.objective + "\" implemented")


class Scorer:
    def eval(self, y_true, y_pred):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_results(self):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_objective_result(self):
        raise NotImplementedError("Has be implemented in the sub class")


class RegScorer(Scorer):

    def __init__(self):
        self.mses = []
        self.r2s = []

    def eval(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.mses.append(mse)
        self.r2s.append(r2)

        return {"MSE": mse, "R2": r2}

    def get_results(self):
        mse_mean = np.mean(self.mses)
        mse_std = np.std(self.mses)

        r2_mean = np.mean(self.r2s)
        r2_std = np.std(self.r2s)

        return {"MSE - mean": mse_mean,
                "MSE - std": mse_std,
                "R2 - mean": r2_mean,
                "R2 - std": r2_std}

    def get_objective_result(self):
        return np.mean(self.mses)


class ClassScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []

    '''
        y_true: (n_samples,)
        y_pred: (n_samples, n_classes) - probabilities of the classes (summing to 1)
    '''
    def eval(self, y_true, y_pred):
        logloss = log_loss(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average="micro")

        # Accuracy and F1 score need the final label predictions
        pred_label = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true, pred_label)
        f1 = f1_score(y_true, pred_label, average="micro")  # use here macro or weighted?

        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        return {"Log Loss - mean": logloss_mean,
                "Log Loss - std": logloss_std,
                "AUC - mean": auc_mean,
                "AUC - std": auc_std,
                "Accuracy - mean": acc_mean,
                "Accuracy - std": acc_std,
                "F1 score - mean": f1_mean,
                "F1 score - std": f1_std}

    def get_objective_result(self):
        return np.mean(self.loglosses)
