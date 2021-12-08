import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset

import models.deepgbm_lib.config as config

from models.deepgbm_lib.utils.helper import AdamW, eval_metrics, printMetric

'''

    Train given model (used for Embedding Model and DeepGBM)
    
    Returns:
        - model: trained model
        - optimizer: optimizer used during training

'''

model_path = "deepgbm.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trainModel(model, train_x, train_y, tree_outputs, test_x, test_y, optimizer,
               train_x_cat=None, test_x_cat=None, epochs=20, early_stopping_rounds=5, save_model=False):
    task = config.config['task']

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    tree_outputs = torch.tensor(tree_outputs)

    if train_x_cat is not None:
        train_x_cat = torch.tensor(train_x_cat)
        trainset = TensorDataset(train_x, train_y, tree_outputs, train_x_cat)
    else:
        trainset = TensorDataset(train_x, train_y, tree_outputs)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.config['batch_size'])
    # , shuffle=True, num_workers=2

    min_test_loss = float("inf")
    min_test_loss_idx = 0

    for epoch in range(epochs):

        running_loss = 0.0
        num_it = 0

        for i, data in enumerate(trainloader, 0):

            # Get data and target from trainloader

            if train_x_cat is not None:
                inputs, target, tree_targets, inputs_cat = data
            else:
                inputs, target, tree_targets = data

            # Put model in training mode
            model.train()

            # Zero the gradients
            optimizer.zero_grad()

            # forward
            if train_x_cat is not None:
                outputs = model(inputs, inputs_cat)
            else:
                outputs = model(inputs)

            # Compute the loss using the tree outputs
            loss_ratio = max(0.3, config.config['loss_dr'] ** (epoch // config.config['loss_de']))
            loss_val = model.joint_loss(outputs[0], target.float(), outputs[1], tree_targets.float(), loss_ratio)

            # Update gradients
            loss_val.backward()

            # Compute loss for documentation
            loss = model.true_loss(outputs[0], target.float())

            # optimize the parameters
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            num_it += 1

        print("Epoch %d: training loss %.3f" % (epoch + 1, running_loss / num_it))
        running_loss = 0.0

        # Eval Testset
        test_loss, preds = evaluateModel(model, test_x, test_y, test_x_cat)
        metric = eval_metrics(task, test_y, preds)
        printMetric(task, metric, test_loss)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            min_test_loss_idx = epoch

        if min_test_loss_idx + early_stopping_rounds < epoch:
            print("Early stopping applies!")
            return

    print('Finished Training')

    if save_model:
        torch.save(model.state_dict(), model_path)

    return model, optimizer


'''

    Evaluate given model (used for Embedding Model and DeepGBM)
    
    Returns:
        - test loss
        - predictions on test data

'''


def evaluateModel(model, test_x, test_y, test_x_cat=None):
    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    if test_x_cat is not None:
        test_x_cat = torch.tensor(test_x_cat)
        testset = TensorDataset(test_x, test_y, test_x_cat)
    else:
        testset = TensorDataset(test_x, test_y)

    testloader = torch.utils.data.DataLoader(testset, batch_size=config.config['test_batch_size'])

    # Put model in evaluation mode
    model.eval()

    y_preds = []
    sum_loss = 0

    with torch.no_grad():
        for data in testloader:

            # Get data and target from dataloader
            if test_x_cat is not None:
                inputs, target, inputs_cat = data
            else:
                inputs, target = data

            # Calculate outputs 
            if test_x_cat is not None:
                outputs = model(inputs, inputs_cat)[0]
            else:
                outputs = model(inputs)[0]

            y_preds.append(outputs.cpu())

            # Compute loss
            loss = model.true_loss(outputs, target.float()).cpu()
            sum_loss += loss * target.shape[0]

    return sum_loss / test_x.shape[0], np.concatenate(y_preds)


'''
    Make predictions on new data
    
    Returns:
        - predictions for input data

'''


def makePredictions(model, test_x, test_cat):
    testset = TensorDataset(torch.tensor(test_x), torch.tensor(test_cat))
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.config['test_batch_size'])

    # Put model in evaluation mode
    model.eval()

    y_preds = []

    with torch.no_grad():
        for data in testloader:
            inputs, inputs_cat = data

            outputs = model(inputs, inputs_cat)[0]
            y_preds.append(outputs.cpu())

    return np.concatenate(y_preds, axis=0)
