# Open Performance Benchmark on Tabular Data

Basis for various experiments on deep learning models for tabular data.
See the [Deep Neural Networks and Tabular Data: A Survey](https://ieeexplore.ieee.org/abstract/document/9998482/) paper.

## Results 
Open performance benchmark results based on (stratified) 5-fold cross-validation. We use the same fold splitting strategy for every data set. The top results for each data set are in bold. The mean and standard deviation values are reported for each baseline model. Missing results indicate that the corresponding model could not be applied to the task type (regression or multi-class classification)

| Method         | HELOC         |           | Adult         |           | HIGGS         |           | Covertype     |           | Cal. Housing  |
|----------------|---------------|-----------|---------------|-----------|---------------|-----------|---------------|-----------|---------------|
|                | Acc↑          | AUC↑      | Acc↑          | AUC↑      | Acc↑          | AUC↑      | Acc↑          | AUC↑      | MSE↓          |
| Linear Model   | 73.0±0.0      | 80.1±0.1  | 82.5±0.2      | 85.4±0.2  | 64.1±0.0      | 68.4±0.0  | 72.4±0.0      | 92.8±0.0  | 0.528±0.008   |
| KNN            | 72.2±0.0      | 79.0±0.1  | 83.2±0.2      | 87.5±0.2  | 62.3±0.1      | 67.1±0.0  | 70.2±0.1      | 90.1±0.2  | 0.421±0.009   |
| Decision Tree  | 80.3±0.0      | 89.3±0.1  | 85.3±0.2      | 89.8±0.1  | 71.3±0.0      | 78.7±0.0  | 79.1±0.0      | 95.0±0.0  | 0.404±0.007   |
| Random Forest  | 82.1±0.3      | 90.0±0.2  | 86.1±0.2      | 91.7±0.2  | 71.9±0.0      | 79.7±0.0  | 78.1±0.1      | 96.1±0.0  | 0.272±0.006   |
| XGBoost        | 83.5±0.2      | 92.2±0.0  | 87.3±0.2      | 92.8±0.1  | 77.6±0.0      | 85.9±0.0  | **97.3±0.0**      | **99.9±0.0**  | 0.206±0.005   |
| LightGBM       | 83.5±0.1      | 92.3±0.0  | **87.4±0.2**      | **92.9±0.1**  | 77.1±0.0      | 85.5±0.0  | 93.5±0.0      | 99.7±0.0  | **0.195±0.005**   |
| CatBoost       | **83.6±0.3**  | **92.4±0.1**| 87.2±0.2      | 92.8±0.1  | 77.5±0.0      | 85.8±0.0  | 96.4±0.0      | 99.8±0.0  | 0.196±0.004   |
| Model Trees    | 82.6±0.2      | 91.5±0.0  | 85.0±0.2      | 90.4±0.1  | 69.8±0.0      | 76.7±0.0  | -              | -          | 0.385±0.019   |
| MLP            | 73.2±0.3      | 80.3±0.1  | 84.8±0.1      | 90.3±0.2  | 77.1±0.0      | 85.6±0.0  | 91.0±0.4      | 76.1±3.0  | 0.263±0.008   |
| VIME           | 72.7±0.0      | 79.2±0.0  | 84.8±0.2      | 90.5±0.2  | 76.9±0.2      | 85.5±0.1  | 90.9±0.1      | 82.9±0.7  | 0.275±0.007   |
| DeepFM         | 73.6±0.2      | 80.4±0.1  | 86.1±0.2      | 91.7±0.1  | 76.9±0.0      | 83.4±0.0  | -              | -          | 0.260±0.006   |
| DeepGBM        | 78.0±0.4      | 84.1±0.1  | 84.6±0.3      | 90.8±0.1  | 74.5±0.0      | 83.0±0.0  | -              | -          | 0.856±0.065   |
| NODE           | 79.8±0.2      | 87.5±0.2  | 85.6±0.3      | 91.1±0.2  | 76.9±0.1      | 85.4±0.1  | 89.9±0.1      | 98.7±0.0  | 0.276±0.005   |
| NAM            | 73.3±0.1      | 80.7±0.3  | 83.4±0.1      | 86.6±0.1  | 53.9±0.6      | 55.0±1.2  | -              | -          | 0.725±0.022   |
| Net-DNF        | 82.6±0.4      | 91.5±0.2  | 85.7±0.2      | 91.3±0.1  | 76.6±0.1      | 85.1±0.1  | 94.2±0.1      | 99.1±0.0  | -              |
| TabNet         | 81.0±0.1      | 90.0±0.1  | 85.4±0.2      | 91.1±0.1  | 76.5±1.3      | 84.9±1.4  | 93.1±0.2      | 99.4±0.0  | 0.346±0.007   |
| TabTransformer | 73.3±0.1      | 80.1±0.2  | 85.2±0.2      | 90.6±0.2  | 73.8±0.0      | 81.9±0.0  | 76.5±0.3      | 72.9±2.3  | 0.451±0.014   |
| SAINT          | 82.1±0.3      | 90.7±0.2  | 86.1±0.3      | 91.6±0.2  | **79.8±0.0**  | **88.3±0.0**  | 96.3±0.1      | 99.8±0.0  | 0.226±0.004   |
| RLN            | 73.2±0.4      | 80.1±0.4  | 81.0±1.6      | 75.9±8.2  | 71.8±0.2      | 79.4±0.2  | 77.2±1.5      | 92.0±0.9  | 0.348±0.013   |
| STG            | 73.1±0.1      | 80.0±0.1  | 85.4±0.1      | 90.9±0.1  | 73.9±0.1      | 81.9±0.1  | 81.8±0.3      | 96.2±0.0  | 0.285±0.006   |



## How to use

### Using the docker container

The code is designed to run inside a docker container. See the `Dockerfile`.
In the docker file, different conda environments are specified for the various 
requirements of the models. Therefore, building the container for the first time takes a
while.

Just build it as usual via `docker build -t <image name> <path to Dockerfile>`.

To start the docker container then run:

``docker run -v ~/output:/opt/notebooks/output -p 3123:3123 --rm -it --gpus all <image name>``

- The `-v ~/output:/opt/notebooks/output` option is recommended to have access to the 
outputs of the experiments on your local machine.

- The `docker run` command starts a jupyter notebook (to have a nice editor for small changes or experiments).
To have access to the notebook from outside the docker container, `-p 3123:3123` connects the notebook to your local 
machine. You can change the port number in the `Dockerfile`.

- If you have GPUs available, add also the `--gpus all` option to have access to them from
inside the docker container.

To enter the running docker container via the command do the following:
- Call `docker ps` to find the ID of the running container.
- Call `docker exec -it <container id> bash` to enter the container. 
Now you can navigate to the right directory with `cd opt/notebooks/`.

----------------------------

### Run a single model on a single dataset

To run a single model on a single dataset call:

``python train.py --config/<config-file of the dataset>.yml --model_name <Name of the Model>``

All parameters set in the config file, can be overwritten by command line arguments, for example:

- ``--optimize_hyperparameters`` Uses [Optuna](https://optuna.org/) to run a hyperparameter optimization. If not set, the parameters listed in the `best_params.yml` file are used.

- ``--n_trails <number trials>`` Number of trials to run for the hyperparameter search

- ``--epochs <number epochs>`` Max number of epochs

- ``--use_gpu`` If set, available GPUs are used (specified by `gpu_ids`)

- ... and so on. All possible parameters can be found in the config files or calling: 
``python train.y -h``

If you are using the docker container, first enter the right conda environment using `conda activate <env name>` to 
have all required packages. The `train.py` file is in the `opt/notebooks/` directory.

--------------------------------------

### Run multiple models on multiple datasets

To run multiple models on multiple datasets, there is the bash script `testall.sh` provided.
In the bash script the models and datasets can be specified. Every model needs to know in 
which conda environment in has to be executed.

If you run inside our docker container, just comment out all models and datasets you don't
want to run and then call:

`./testall.sh`

-------------------------------------
### Computing model attributions (currently supported for SAINT, TabTransformer, TabNet)

The framework provides implementations to compute feature attribution explanations for several models.
Additionally, the feature attributions can be automatically compared to SHAP values and a global ablation 
test which successively perturbs the most important features, can be run. The same parameters as before can be passed, but
with some additions:

`attribute.py --model_name <Name of the Model> [--globalbenchmark] [--compareshap] [--numruns <int>] [--strategy diag]`

- `--globalbenchmark` Additionally run the global perturbation benchmark

- `--compareshap` Compare attributions to shapley values

- `--numruns <number run>` Number of repetitions for the global benchmark

- ``--strategy diag`` SAINT and TabTransformer support another attribution strategy, where the diagonal of the attention map is used. Pass this argument to use it.


-------------------------------------

## Add new models

Every new model should inherit from the base class `BaseModel`. Implement the following methods:

- `def __init__(self, params, args)`: Define your model here.
- `def fit(self, X, y, X_val=None, y_val=None)`: Implement the training process. (Return the loss and validation history)
- `def predict(self, X)`: Save and return the predictions on the test data - the regression values or the concrete classes for classification tasks
- `def predict_proba(self, X)`: Only for classification tasks. Save and return the probability distribution over the classes.
- `def define_trial_parameters(cls, trial, args)`: Define the hyperparameters that should be optimized.
- (optional) `def save_model`: If you want to save your model in a specific manner, override this function to.

Add your `<model>.py` file to the `models` directory and do not forget to update the `models/__init__.py` file.

----------------------------------------------

## Add new datasets

Every dataset needs a config file specifying its features. Add the config file to the `config` directory.

Necessary information are:
- *dataset*: Name of the dataset
- *objective*: Binary, classification or regression task
- *direction*: Direction of optimization. In the current implementation the binary scorer returns the AUC-score,
hence, should be maximized. The classification scorer uses the log loss and the regression scorer mse, therefore
both should be minimized.
- *num_features*: Total number of features in the dataset
- *num_classes*: Number of classes in classification task. Set to 1 for binary or regression task.
- *cat_idx*: List the indices of the categorical features in your dataset (if there are any).

It is recommended to specify the remaining hyperparameters here as well.

----------------------------

<!-- ![Architecture of the docker container](Docker_architecture.png) -->




## Citation  
If you use this codebase, please cite our work:
```bib
@article{borisov2022deep,
 author={Borisov, Vadim and Leemann, Tobias and Seßler, Kathrin and Haug, Johannes and Pawelczyk, Martin and Kasneci, Gjergji},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Deep Neural Networks and Tabular Data: A Survey}, 
  year={2022},
  volume={},
  number={},
  pages={1-21},
  doi={10.1109/TNNLS.2022.3229161}
}
```
