import configargparse


def get_parser():
    # Use parser that can read YML files
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # Put all parameters into config_california_housing.yml file!
    parser.add('-config', '--config', required=False, is_config_file_arg=True, help='config file path',
               default="config/config_covertype.yml")  # config_kddcup99    config_california_housing

    parser.add('--model_name', required=True, help="Name of the model that should be trained")
    parser.add('--dataset', required=True, help="Name of the dataset that will be used")
    parser.add('--objective', type=str, default="regression", choices=["regression", "classification"],
               help="Regression or Classification task")

    parser.add('--use_gpu', action="store_true", help="Set to true if GPU is available")
    parser.add('--gpu_id', type=int, help="")

    parser.add('--n_trials', type=int, default=100, help="Number of trials of the hyperparameter optimization")
    parser.add('--direction', type=str, default="minimize", choices=['minimize', 'maximize'],
               help="Direction of optimization.")

    parser.add('--num_splits', type=int, default=5, help="Number of splits done for cross validation")
    parser.add('--shuffle', action="store_true", help="Direction of optimization.")
    parser.add('--seed', type=int, help="Seed for KFold initialization.")

    parser.add('--scale', action="store_true", help="Normalize input data.")
    parser.add('--target_encode', action="store_true", help="Encode the targets that they start at 0. (0, 1, 2,...)")

    parser.add('--early_stopping_rounds', type=int, help="Number of rounds before early stopping applies.")
    parser.add('--epochs', type=int, help="Max number of epochs to train.")
    parser.add('--logging_period', type=int, help="Number of iteration after which validation is printed.")

    parser.add('--num_features', type=int, help="Set the total number of features.")
    parser.add('--num_classes', type=int, help="Set the number of classes in a classification task.")
    parser.add('--cat_idx', type=int, action="append", help="Indices of the categorical features")

    # Todo: Validate the arguments

    return parser
