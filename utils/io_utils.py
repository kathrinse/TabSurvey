import numpy as np
import os
import pickle

output_dir = "output/"


def save_predictions_to_file(arr, args, extension=""):
    filename = get_output_path(args, directory="predictions", filename="p", extension=extension, file_type="npy")
    np.save(filename, arr)


def save_model_to_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    pickle.dump(model, open(filename, 'wb'))


def save_results_to_file(args, results, train_time=None, test_time=None, best_params=None):
    filename = get_output_path(args, filename="results", type="txt")

    with open(filename, "w") as text_file:
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        text_file.write("\nTrain time: %f\n" % train_time)
        text_file.write("Test time: %f\n" % test_time)

        text_file.write("\nBest Parameters: %s" % best_params)


def get_output_path(args, filename, file_type, directory=None, extension=None):
    # For example: output/LinearModel/Covertype
    dir_path = output_dir + args.model_name + "/" + args.dataset

    if directory:
        # For example: .../models
        dir_path = dir_path + "/" + directory

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + "/" + filename

    if extension:
        file_path += "_" + str(extension)

    file_path += "." + file_type

    # For example: .../m_3.pkl

    return file_path
