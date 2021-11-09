import numpy as np
import os
import pickle

output_dir = "output/"


def save_predictions_to_file(arr, args, extension=""):
    path = output_dir + args.model_name + "/" + args.dataset + "/predictions"

    if not os.path.isdir(path):
        os.makedirs(path)

    filename = path + "/p_" + str(extension) + ".npy"
    np.save(filename, arr)


def save_model_to_file(model, args, extension=""):
    path = output_dir + args.model_name + "/" + args.dataset + "/models"

    if not os.path.isdir(path):
        os.makedirs(path)

    filename = path + "/m_" + str(extension) + ".pkl"
    pickle.dump(model, open(filename, 'wb'))


def save_results_to_file(args, results, train_time=None, test_time=None, best_params=None):
    path = output_dir + args.model_name + "/" + args.dataset

    if not os.path.isdir(path):
        os.makedirs(path)

    filename = path + "/results.txt"

    with open(filename, "w") as text_file:
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        text_file.write("\nTrain time: %f\n" % train_time)
        text_file.write("Test time: %f\n" % test_time)

        text_file.write("\nBest Parameters: %s" % best_params)
