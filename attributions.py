# Run a model to compute attributions and compare them to a baseline.
import numpy as np
from utils.parser import get_attribution_parser
from models import str2model
from utils.load_data import load_data
from sklearn.model_selection import train_test_split

def train_model(args, model,  X_train, X_val, y_train, y_val):
    #loss_history, val_loss_history = model.fit(X_train, y_train, X_val, y_val)
    val_model(model, X_val, y_val)
    return model 

def val_model(model, X_val, y_val):
    ypred = model.predict(X_val)
    print(ypred.shape, y_val.shape)
    if len(ypred.shape)==2:
        ypred = ypred[:,0]
    acc = np.sum((ypred.flatten() > 0.5) == y_val)/len(y_val)
    print("Accuracy: ", acc)

def main(args):
    X, y = load_data(args)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=args.seed)
    args.epochs = 1
    model_name = str2model(args.model_name)
    model = model_name(arguments.parameters[args.model_name], args)
    model = train_model(args, model, X_train, X_val, y_train, y_val)
    attrs = model.attribute(X_val, y_val)
    print("Shape of attributions:", attrs.shape)
    
if __name__ == "__main__":
    parser = get_attribution_parser()
    arguments = parser.parse_args()
    print(arguments.parameters)
    main(arguments)
