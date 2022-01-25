# Run a model to compute attributions and compare them to a baseline.
import numpy as np
import matplotlib.pyplot as plt
from utils.parser import get_attribution_parser
from models import str2model
from utils.load_data import load_data
from utils.io_utils import save_results_to_json_file
from sklearn.model_selection import train_test_split
from utils.baseline_attributions import get_shap_attributions
from models.basemodel import BaseModel

def train_model(args, model: BaseModel,  X_train: np.ndarray, X_val: np.ndarray,
                                         y_train: np.ndarray, y_val: np.ndarray):
    """ Train model using parameters args. 
        X_train, y_train: Training data and labels
        X_val and y_val: Test data and labels
    """
    loss_history, val_loss_history = model.fit(X_train, y_train, X_val, y_val)
    val_model(model, X_val, y_val)
    return model 


def global_removal_benchmark(args, model: BaseModel,  X_train, X_val, y_train, y_val, 
                feature_importances: np.ndarray, order_morf = True) -> np.ndarray:
    """ Perform a feature removal benchmark for the attributions. 
        The features that are attributed the highest overall attribution scores are successivly removed from the 
        dataset. The model is then retrained.
        features_importances: A vector of N (number of features in X) values that contain the importance score for each feature.
            The features will be orderes by the absolute value of the importance.
        order_morf: Feature removal order. Either remove most important (morf=True) or least important (morf=False) features first
        Return: array with the obtained accuracies.
    """
    print(X_train.shape, feature_importances.shape)
    if X_train.shape[1] != len(feature_importances):
        raise ValueError("Number of Features in Trainset must be equal to number of importances passed.")

    ranking = np.argsort((1 if order_morf else -1)*np.abs(feature_importances))
    results = np.zeros(len(feature_importances))
    old_cat_index = args.cat_idx
    old_cat_dims = args.cat_dims
    for i in range(len(feature_importances)):
        remaining_features = len(feature_importances)-i
        use_idx = ranking[:remaining_features].copy()
        np.random.shuffle(use_idx) # make sure the neighborhood relation is not important.

        print(f"Using {len(use_idx)} features ...")
        # Retrain the model and report acc.
        X_train_bench = X_train[:, use_idx]
        X_val_bench = X_val[:, use_idx]

        # modify feature args accordingly
        # args.num_features: points to the new number of features
        # args.cat_idx: Indices of categorical features
        # args.cat_dims: Number of categorical feature values
        # These values have to be recomputed for the modified dataset
        new_cat_idx = []
        new_cat_dims = []
        for j in range(len(use_idx)):
            if use_idx[j] in old_cat_index:
                old_index = old_cat_index.index(use_idx[j])
                new_cat_idx.append(j)
                new_cat_dims.append(old_cat_dims[old_index])

        args.cat_idx = new_cat_idx
        args.cat_dims = new_cat_dims
        #print(new_cat_idx, args.cat_dims)
        args.num_features = remaining_features
        del model
        model_name = str2model(args.model_name)
        model = model_name(arguments.parameters[args.model_name], args)
        model = train_model(args, model, X_train_bench, X_val_bench, y_train, y_val)
        acc_obtained = val_model(model, X_val_bench, y_val)
        results[i] = acc_obtained

        res_dict = {}
        res_dict["model"] = args.model_name
        res_dict["order"] = "MoRF" if order_morf else "LeRF"
        res_dict["accuracies"] = results.tolist()
        res_dict["attributions"] = feature_importances.tolist()
    save_results_to_json_file(args, res_dict, f"global_benchmark{args.strategy}", append=True)
    return results

def compute_spearman_corr(attr1, attr2):
    """ Compute the spearman correlations between two attributions.
        Return a vector with the spearman correlation between all rows in the matrix.
    """
    num_inputs = attr1.shape[0]
    resmat = np.zeros(num_inputs)
    ranks1 = np.argsort(np.argsort(attr1, axis=0), axis=0)
    ranks2 = np.argsort(np.argsort(attr2, axis=0), axis=0)

    cov = np.mean(ranks1*ranks2, axis=0)-np.mean(ranks1, axis = 0)*np.mean(ranks2, axis=0) # E[XY]-E[Y]E[X]
    corr = cov/(np.std(ranks1, axis=0)*np.std(ranks2, axis=0))
    return corr

def compare_to_shap(args, attrs, model, X_val, y_val, sample_size=100):
    """ 
        Compare feature attributions by the model to shap values on a random set of validation points.
        Compute correlation and save raw output.
    """
    use_samples = np.arange(len(X_val))
    np.random.shuffle(use_samples) 
    use_samples = use_samples[:sample_size]
    attrs = attrs[use_samples]

    res_dict = {}
    res_dict["model"] = args.model_name
    res_dict["model_attributions"] = attrs.tolist()
 

    shap_attrs = get_shap_attributions(model, X_val[use_samples], y_val[use_samples])
    #save_attributions_image(attrs, feature_names, args.model_name+"_shap")
    res_dict["shap_attributions"] = shap_attrs.tolist()
    
    #print(attrs, shap_attrs)
    rank_corrs = compute_spearman_corr(np.abs(attrs), np.abs(shap_attrs))
    res_dict["rank_corr_mean"] = np.mean(rank_corrs)
    res_dict["rank_corr_std"] = np.std(rank_corrs)
    save_results_to_json_file(args, res_dict, f"shap_compare{args.strategy}", append=False)

def val_model(model, X_val, y_val) -> float:
    """ 
        Validation of a trained model on the test set (X_val, y_val). 
    """
    ypred = model.predict(X_val)
    if len(ypred.shape)==2:
        ypred = ypred[:,-1]
    acc = np.sum((ypred.flatten() > 0.5) == y_val)/len(y_val)
    print("Accuracy: ", acc)
    return acc

def save_attributions_image(attrs, namelist, file_name):
    """ Save attributions in a plot. """
    print(np.max(attrs), np.min(attrs))
    attrs_abs = np.abs(attrs)
    attrs_abs -= np.min(attrs_abs)
    attrs_abs /= np.max(attrs_abs)
    plt.ioff()
    plt.matshow(attrs_abs)
    plt.xticks(np.arange(len(namelist)), namelist, rotation=90)
    plt.tight_layout()
    plt.gcf().savefig(f"output/attributions_{file_name}.png")


def main(args):
    if args.model_name == "TabTransformer": # Use discretized version of adult dataset for TabNet attributions.
        args.scale = False

    X, y = load_data(args)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=args.seed)
    args.epochs=1
    model_name = str2model(args.model_name)
    model = model_name(arguments.parameters[args.model_name], args)
    #val_model(model, X_val, y_val)
    model = train_model(args, model, X_train, X_val, y_train, y_val)
    attrs = model.attribute(X_val, y_val, args.strategy)
    feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    save_attributions_image(attrs[:20,:], feature_names, args.model_name)

    if args.globalbenchmark:
        for run in range(args.numruns):
            global_removal_benchmark(args, model, X_train, X_val, y_train, y_val, attrs.mean(axis=0).flatten(), order_morf = True)

    if args.compareshap:
        compare_to_shap(args, attrs, model, X_val, y_val, sample_size=10)

if __name__ == "__main__":
    parser = get_attribution_parser()
    arguments = parser.parse_args()
    print(arguments.parameters)
    main(arguments)
