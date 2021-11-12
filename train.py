import optuna

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file
from utils.parser import get_parser

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder


def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        curr_model.fit(X_train, y_train, X_val, y_val)
        train_timer.end()

        # Test model
        test_timer.start()
        predictions = curr_model.predict(X_test)
        test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
        if save_model:
            curr_model.save_model_and_predictions(y_test, i)

        # Compute scores on the output
        sc.eval(y_test, predictions)

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    # print("Finished cross validation")
    return sc


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc = cross_validation(model, self.X, self.y, self.args)

        return sc.get_objective_result()


def main(args):
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    study = optuna.create_study(direction=args.direction)
    study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
    print("Best parameters:", study.best_trial.params)

    # Run best trial again and save it!
    model = model_name(study.best_trial.params, args)
    cross_validation(model, X, y, args, save_model=True)


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print(arguments)

    main(arguments)
