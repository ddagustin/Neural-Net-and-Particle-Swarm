# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:43:17 2021

COIT29224 Evolutionary Computation
Assessment 1 - Neural Network Weight Optimisation using
                Particle Swarms

@author: Daniel Roi Agustin
"""

# ---- import dependencies
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# import assessment related modules
from src import NeuralNetwork, PSO

import warnings
warnings.filterwarnings("ignore")

# for reproducibility
random.seed(0)


# ---- data preprocessing

# load dataset and split into training and testing sets
dataset = load_iris()
x_ = dataset.data
y_ = dataset.target

x_train, x_test, y_train, y_test = train_test_split( x_, y_, test_size = 0.3, random_state = 123, stratify = y_ )

# scale x_ values to optimise model training
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform( x_train )

x_test_scaled = scaler.transform( x_test )


# ---- application

# initialise neural network
nn = NeuralNetwork( 4, 30, 3, activation_func="tanh" ) # include x and y here

# neural network is also trained INSIDE this PSO instance
pso = PSO( 100, 30 ) # only swarms here
pso.optimise_swarm( nn, x_train_scaled, y_train, 500 ) # nn and iters here

# apply trained nn to test data
y_pred = nn.predict( x_test_scaled )
test_acc = ( y_pred == y_test ).mean()

print( test_acc )