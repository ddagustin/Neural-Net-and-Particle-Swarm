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
random.seed(100)


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

# neural network self optimisation
print( "Starting Neural Network Optimisation" )
nn.self_optimise(x_train_scaled, y_train, 0.05, 500) 
nn_y_pred = nn.predict( x_test_scaled )
nn_test_acc = ( nn_y_pred == y_test ).mean()

print( "Neural network done" )
print( "\n" )

# neural network is also trained INSIDE this PSO instance
print( "Starting Particle Swarm Optimisation" )
pso = PSO( 100, 30 ) # only swarms here
pso.optimise_swarm( nn, x_train_scaled, y_train, 500 ) # nn and iters here

# apply trained nn to test data
pso_y_pred = nn.predict( x_test_scaled )
pso_test_acc = ( pso_y_pred == y_test ).mean()

print( "PSO done" )
print("\n")

print("Test Parameters\n" 
      f"Neural Network Accuracy = {nn_test_acc:.4f}\n"
      f"Particle Swarm Opsimisation Accuracy = {pso_test_acc:.4f}")

