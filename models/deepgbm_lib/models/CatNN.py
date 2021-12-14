import numpy as np
import math

import torch
import torch.nn as nn

import models.deepgbm_lib.config as config

'''
    CatNN:
    
    Neural Network specialized for categorical data.

'''

class CatNN(nn.Module):

    def __init__(self, field_size, feature_sizes, deep_layers_activation = 'relu'):
        super(CatNN, self).__init__()
        
        # Set all class variables
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = config.config['embedding_size']
        self.task = config.config['task']
        self.deep_layers_activation = deep_layers_activation
        self.bias = 0
        self.deep_layers = config.config['cate_layers']

        stdv = math.sqrt(1.0 /len(feature_sizes))
        
        # A simple NN based on the DeepFM
        # TODO: Change this to something better?
        
        # 1. Embedding of the categorical features
        self.input_emb = nn.Embedding(sum(self.feature_sizes), 1)
        self.input_emb.weight.data.normal_(0, stdv)

        # 2. Dropout layer
        self.dropout_1 = nn.Dropout(0.5)
        
        # 3. Embedding of the categorical features to embedding size
        self.second_emb = nn.Embedding(sum(self.feature_sizes), self.embedding_size)
        self.second_emb.weight.data.normal_(0, stdv)
            
        # 4. Dropout layer
        self.dropout_2 = nn.Dropout(0.5)
        
        # Deep layers:
        
        # 5. Linear layer with BN
        self.linear_1 = nn.Linear(self.field_size * self.embedding_size, self.deep_layers[0])
        self.batch_norm_1 = nn.BatchNorm1d(self.deep_layers[0])
        
        # 6. Linear layer with BN
        self.linear_2 = nn.Linear(self.deep_layers[0], self.deep_layers[1])
        self.batch_norm_2 = nn.BatchNorm1d(self.deep_layers[1])
        
        # Set correct loss
        if self.task == 'binary':
            self.criterion = nn.BCELoss()
        elif self.task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            print ("Task not yet implemented.")
            # TODO: Implement classification
        
        print("Init CatNN succeed!")

    def forward(self, X):
        
        # Convert input tensor to datatype long
        Xi = X.long()
        
        batch_size = X.size(0)
        
        # Embedding of the input
        emb = self.input_emb(Xi.reshape(batch_size * self.field_size))
        emb = emb.view(batch_size, -1)
        
        dropout_1 = self.dropout_1(emb)
        
        next_emb = self.second_emb(Xi.reshape(batch_size * self.field_size))
        next_emb = next_emb.view(batch_size, self.field_size, -1)
        
        # Make three-dim array to two-dim array by multiplying the rows
        
        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        emb_sum = torch.sum(next_emb, 1)            
        emb_sum_square = emb_sum * emb_sum # (x+y)^2            
        emb_square = next_emb * next_emb            
        emb_square_sum = torch.sum(emb_square, 1) #x^2+y^2
        mult_emb = (emb_sum_square - emb_square_sum) * 0.5
        
        dropout_2 = self.dropout_2(mult_emb)
        
        # Deep layers:
        
        deep_emb = next_emb.reshape(batch_size, -1)
        
        if self.deep_layers_activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif self.deep_layers_activation == 'tanh':
            activation = nn.Tanh()
        elif self.deep_layers_activation == 'relu':
            activation = nn.ReLU()
        else: 
            print("Activation layer not yet implemented.")
                        
        x_deep = self.linear_1(deep_emb)
        x_deep = self.batch_norm_1(x_deep)
        x_deep = activation(x_deep)
        
        x_deep = self.linear_2(x_deep)
        x_deep = self.batch_norm_2(x_deep)
        x_deep = activation(x_deep)
        
        # Why this?
        total_sum = torch.sum(dropout_1, 1) + torch.sum(dropout_2, 1) + torch.sum(x_deep, 1) + self.bias
        
        if self.task == 'binary':
            return nn.Sigmoid()(total_sum)
        
        # TODO: Classification missing
        
        return total_sum
    
    
    def true_loss(self, pred, target):
        return self.criterion(pred.view(-1), target.view(-1))