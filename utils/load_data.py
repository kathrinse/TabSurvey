import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


def load_data(args):
    print("Loading dataset " + args.dataset + "...")

    if args.dataset == "CaliforniaHousing":  # Regression dataset
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    elif args.dataset == "Covertype":  # Multi-class classification dataset
        X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
        X, y = X[:10000, :], y[:10000] # only take 10000 samples from dataset
    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")

    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

    #if args.target_one_hot_encode:
     #   enc = OneHotEncoder()
     #   y = enc.fit_transform(y.reshape(-1, 1)).toarray()

    # Preprocess data
    if args.scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X, y)

    return X, y
