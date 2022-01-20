# Run a model to compute attributions and compare them to a baseline.
import numpy as np
import matplotlib.pyplot as plt
from utils.parser import get_attribution_parser
from models import str2model
from utils.load_data import load_data
from sklearn.model_selection import train_test_split
from utils.baseline_attributions import get_shap_attributions


def train_model(args, model,  X_train, X_val, y_train, y_val):
    loss_history, val_loss_history = model.fit(X_train, y_train, X_val, y_val)
    val_model(model, X_val, y_val)
    return model 


def val_model(model, X_val, y_val):
    ypred = model.predict(X_val)
    print(ypred.shape, y_val.shape)
    if len(ypred.shape)==2:
        ypred = ypred[:,-1]
    acc = np.sum((ypred.flatten() > 0.5) == y_val)/len(y_val)
    print("Accuracy: ", acc)


def save_attributions(attrs, namelist, file_name):
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
    
    print(args.scale)
    model_name = str2model(args.model_name)
    model = model_name(arguments.parameters[args.model_name], args)
    args.epochs = 25
    val_model(model, X_val, y_val)
    #model = train_model(args, model, X_train, X_val, y_train, y_val)
    attrs = model.attribute(X_val, y_val)
    print("Shape of attributions:", attrs.shape)
    feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    save_attributions(attrs[:20,:], feature_names, args.model_name)
    attrs = get_shap_attributions(model, X_val[:20], y_val[:20])
    save_attributions(attrs, feature_names, args.model_name+"_shap")

if __name__ == "__main__":
    parser = get_attribution_parser()
    arguments = parser.parse_args()
    print(arguments.parameters)
    main(arguments)
