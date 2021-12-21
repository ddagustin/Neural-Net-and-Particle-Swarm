# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:43:17 2021

COIT29224 Evolutionary Computation
Assessment 1 - Neural Network Weight Optimisation using
                Particle Swarms

@author: Daniel Roi Agustin
"""

# ---- import dependencies and assessment related module
import random
import numpy as np
from src import ACTIVATIONS, DERIVATIVES

from sklearn.datasets import load_iris

dataset = load_iris()
x_ = dataset.data
y_ = dataset.target


# ---- neural network
class NeuralNetwork():

    def __init__( self, n_inputs, n_hidden_nodes, n_outputs, activation_func = "linear" ):
        """
        Neural Network class stores the architecture and functions
        
            Takes in the neural network parameters of inputs, hidden layer sizes, and outputs; 
            resulting to an array of weights and biases in the following shape:
                input nodes * hidden layer nodes + hidden layer nodes * output nodes +
                input bias nodes * hidden layer nodes + hidden bias nodes * output nodes
            increasing hidden layers will just add corresponding number of weights connecting the layers
            
            Activation function is set for all layers except output, which uses softmax and 
                negative log likelihood for probabilities.
                Use of negative log likelihood instead of sigmoid for multi-class classification problem
        
        Attributes
        -------
        n_inputs : int
            number of input nodes
            
        n_hidden_nodes : int
            number of hidden layer nodes
        
        n_outputs : int
            number of output nodes
        
        activation_func : string, optional
            { "linear", "sigmoid", "tanh", "softmax", "relu" }; default = "linear"
            string representation of activation function to be used in
            hidden layer calculations.
        
        """
        
        # architecture
        self.n_inputs = n_inputs
        self.n_hidden_nodes = n_hidden_nodes
        self.n_outputs = n_outputs
        
        # activation functions
        self.activation_func = ACTIVATIONS[ activation_func ]
        self.output_act_func = ACTIVATIONS[ "softmax" ]
        self.back_activation_func = DERIVATIVES[ activation_func ]
        
        # initialise flat weights and biases array
        self.weight_dims = np.cumsum([ n_inputs * n_hidden_nodes, n_hidden_nodes, n_outputs * n_hidden_nodes, n_outputs ])
        self.init_weights()
        
        # accuracy and loss values
        self.loss = None
        self.accuracy = None
        
    
    def update_weights( self, updated_weights ):
        """
        Function to update weight parameters

        Parameters
        ----------
        updated_weights : ndarray
            number of dimensions equal to 
            n_hidden_nodes * ( n_inputs * 1 ) + n_outputs * ( n_hidden_nodes + 1 )

        Returns
        -------
        None.

        """
        self.weights = updated_weights
        
        # roll weights and biases into multidimensional arrays
        self.layer_1 = self.weights[ 0 : self.weight_dims[0] ].reshape(( self.n_inputs, self.n_hidden_nodes ))        
        self.bias_1 = self.weights[ self.weight_dims[0] : self.weight_dims[1] ].reshape(( self.n_hidden_nodes, ))
        self.layer_2 = self.weights[ self.weight_dims[1] : self.weight_dims[2]  ].reshape(( self.n_hidden_nodes, self.n_outputs ))
        self.bias_2 = self.weights[ self.weight_dims[2] : ].reshape(( self.n_outputs, ))
        
    
    def update_accuracy( self, X, y ):
        """
        Calculate network accuracy with current parameters

        Parameters
        ----------
        X : ndarray
            array of input data
        
        y : ndarray
            array of class labels of X

        Returns
        -------
        None.

        """
        # calculate predictions
        predictions = self.predict( X )
        
        # calculate accuracy
        self.accuracy = ( predictions == y ).mean() 
        
    def train( self, X, y ):
        """
        Represents calculation of fitness to calculate the loss of the neural network.

        Parameters
        ----------
        X : ndarray
            array of input data
        
        y : ndarray
            array of class labels of X

        Returns
        -------
        loss : double
            loss value of neural network

        """
        # calculate probabilities
        self.probabilities = self.forward_prop( X )
                                                
        # negative log likelihood for multiclass classification and loss calculation
        neg_log_likelihood = -np.log( self.probabilities[ range( X.shape[0] ), y ])
        
        self.loss = np.sum( neg_log_likelihood ) / X.shape[0]
                
        # returns loss
        return self.loss 
    
    def predict( self, X ):
        """
        Calculating the prediction matrix using the weights

        Parameters
        ----------
        X : ndarray
            array of input data for prediction

        Returns
        -------
        y_prediction : ndarray
            array of predictions

        """
        # calculate probabilities
        probabilities = self.forward_prop( X )
        
        # get predictions
        y_prediction = np.argmax( probabilities, axis = 1 )
        
        return y_prediction
        
    def forward_prop( self, X ):
        """
        Perform forward propagation; multiply input arrays and layers

        Parameters
        ----------
        X : ndarray
            array of input data

        Returns
        -------
        probabilities : ndarray
            logits of the neural network 

        """
        # perform calculations
        self.output_1 = X.dot( self.layer_1 ) + self.bias_1
        self.activation_1 = self.activation_func( self.output_1 )
        self.output_2 = self.activation_1.dot( self.layer_2 ) + self.bias_2
        
        # calculate probabilities for output layer; softmax function
        probabilities = self.output_act_func( self.output_2 )
        return probabilities
    
    def init_weights( self, method = True ):
        if method:
            self.update_weights( np.array( [ random.uniform( -1, 1 ) for x in range( self.weight_dims[-1] )] ) ) # for randomized weights
        else:
            self.update_weights( np.zeros( self.weight_dims[-1] ) ) # for 0 weights
    
    
    # ---- Functions below are for testing the neural network on its own which include:
    def backward_prop( self, X, y, learning_rate ):
        """
        Calculates the gradient descent of each layer
        Overrides the update_weights function to update the neural network weights and biases
        
        Parameters
        ----------
        X : ndarray
            array of input data
        
        y : ndarray
            array of class labels of X
        learning_rate : float
            factor for gradient descent

        Returns
        -------
        None.

        """
        # calculating layer errors
        self.error = ( self.probabilities - np.eye( np.unique(y).shape[0] )[y] ) / self.probabilities.shape[0]
        self.hidden_error = np.dot( self.error, self.layer_2.T ) * self.back_activation_func( self.output_1 )
        
        self.layer_2 -= learning_rate * np.dot( self.activation_1.T, self.error )
        self.bias_2 -= learning_rate * np.sum( self.error, axis = 0 ) # error per column
        
        self.layer_1 -= learning_rate * np.dot( X.T, self.hidden_error )
        self.bias_1 -= learning_rate * np.sum( self.hidden_error, axis = 0 ) # error per column
        
    
    def self_optimise( self, X, y, learning_rate, num_iter ):
        """
        Function to iterate and train the neural network

        Parameters
        ----------
        X : ndarray
            array of input data
        
        y : ndarray
            array of class labels of X
        learning_rate : float
            factor for gradient descent
        num_iter : int
            number of iterations

        Returns
        -------
        None.

        """
        
        temp_loss = np.Inf
        temp_acc = -1
        for i in range(num_iter):
            self.train(X, y)
            self.backward_prop( X, y, learning_rate )
            self.update_accuracy(X, y)
            
            if temp_loss > self.loss and temp_acc < self.accuracy:
                temp_loss, temp_acc = self.loss, self.accuracy
                print( f"neural network metrics; iter {i+1}: acc = {self.accuracy:.4f}, loss = {self.loss:.4f}" )
            
            
    
# ---- for testing purposes, execute when running this file
if __name__ == "__main__":
    nn = NeuralNetwork( 4, 30, 3, activation_func="tanh" )
    
    nn.self_optimise(x_, y_, 0.006, 500)