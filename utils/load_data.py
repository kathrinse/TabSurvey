import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np


def load_data(args):
    print("Loading dataset " + args.dataset + "...")

    if args.dataset == "CaliforniaHousing":  # Regression dataset
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    elif args.dataset == "Covertype":  # Multi-class classification dataset
        X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
        # X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

    elif args.dataset == "KddCup99":
        X, y = sklearn.datasets.fetch_kddcup99(return_X_y=True)
        X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

        # filter out all target classes, that occur less than 1%
        target_counts = np.unique(y, return_counts=True)
        smaller1 = int(X.shape[0] * 0.01)
        small_idx = np.where(target_counts[1] < smaller1)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)

        '''
        # filter out all target classes, that occur less than 100
        target_counts = np.unique(y, return_counts=True)
        small_idx = np.where(target_counts[1] < 100)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)
        '''

    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")
    print(X.shape)

    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Setting this?
        args.num_classes = len(le.classes_)

    scale_idx = []

    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])
        elif args.scale:
            scale_idx.append(i)

    if scale_idx:
        scaler = StandardScaler()
        X[:, scale_idx] = scaler.fit_transform(X[:, scale_idx])

    return X, y
