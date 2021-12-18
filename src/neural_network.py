# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:43:17 2021

COIT29224 Evolutionary Computation
Assessment 1 - Neural Network Weight Optimisation using
                Particle Swarms

@author: Daniel Roi Agustin
"""

# ---- import dependencies and assessment related module
import numpy as np
from src import ACTIVATIONS

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
        
        # initialise flat weights and biases array
        self.weight_dims = np.cumsum([ n_inputs * n_hidden_nodes, n_hidden_nodes, n_outputs * n_hidden_nodes, n_outputs ])    
        self.update_weights( np.zeros( self.weight_dims[-1] ) )
        
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
        
    # ---- include update and predict here to update loss and acc?
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
        output_1 = X.dot( self.layer_1 ) + self.bias_1
        activation_1 = self.activation_func( output_1 )
        output_2 = activation_1.dot( self.layer_2 ) + self.bias_2
        
        # calculate probabilities for output layer; softmax function
        probabilities = self.output_act_func( output_2 )
        return probabilities