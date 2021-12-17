# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:43:17 2021

COIT29224 Evolutionary Computation
Assessment 1 - Neural Network Weight Optimisation using
                Particle Swarm Optimisation

@author: Daniel Roi Agustin
"""

# ---- import dependencies
import random
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit, softmax

import warnings
warnings.filterwarnings("ignore")

random.seed(0)

# ---- class activation function
class activationFunction():
    """
    Storage of activation functions for use in Neural Network

    Parameters
    ----------
    x : numpy ndarray
        input values to be processed by the activation function

    Returns
    -------
    activated array depending on function called

    """
    def Linear( X ):
        return X
    
    def Sigmoid( X ):
        return expit( X )
    
    def Tanh( X ):
        return np.tanh( X )
    
    def Softmax( X ):
        return softmax( X, axis = 1 )
        # exp = np.exp( X - np.max( X ) )
        # return exp / np.sum( exp, axis = 1, keepdims = True )
    
    def ReLu( X ):
        return X * ( X > 0 )

ACTIVATIONS = {
    "linear" : activationFunction.Linear,
    "sigmoid" : activationFunction.Sigmoid,
    "tanh" : activationFunction.Tanh,
    "softmax" : activationFunction.Softmax,
    "relu" : activationFunction.ReLu,
    }
   
    
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
        

# ---- particle class
class Particle():
    
    def __init__( self, dimensions, n_informants ):
        """
        Particle class to store particle parameters
        
        Takes in the dimensions and number of informants to initialise the particle
        min and max boundaries are to be determined

        Parameters
        ----------
        dimensions : int
            number of dimensions of the particle
        n_informants : int
            number of informants, should be less than swarm size of PSO

        Returns
        -------
        None.

        """
        self.dimensions = dimensions
        self.n_informants = n_informants
        
        # positional values
        self.positions = np.array( [ random.uniform( -1, 1 ) for x in range( dimensions )] )
        self.velocities = np.array( [ 0.0 for x in range( dimensions )] )
        self.informantIndices = np.array( [ 0 for i in range( n_informants ) ] )
        
        self.personal_best = 1
        self.best_position = np.array( [ 0 for i in range( dimensions )] )
        self.fitness = self.personal_best
        
        self.group_best = 1
        self.group_best_position = np.array( [ 0 for i in range( dimensions )] )
        
    def calculate_fitness( self, func ):
        """
        Calculate fitness using a custom function

        Parameters
        ----------
        func : object
            Call NeuralNetwork.train during PSO iterations

        Returns
        -------
        None.

        """
        self.fitness = func
    
    def modify_position( self, X ):
        """
        Modify positions

        Parameters
        ----------
        X : ndarray
            Array of positions with dimensions = dimensions

        Returns
        -------
        None.

        """
        self.positions = X

    def calculate_velocity( self, weights ):
        """
        calculating and modifying the velocity of the particle

        Parameters
        ----------
        weights : ndarray
            weight coefficients of velocity multiploer as defined in the PSO

        Returns
        -------
        None.

        """
        for d in range( self.dimensions ):
            part_1 = weights[0] * self.velocities[d]                                                        # current particle velocity 
            part_2 = random.uniform( 0, weights[1] ) * ( self.best_position[d] - self.positions[d] )           # personal best component
            part_3 = random.uniform( 0, weights[1] ) * ( self.group_best_position[d] - self.positions[d] )     # group best component
            self.velocities[d] = part_1 + part_2 + part_3
                
    def set_informants( self, swarm_size, index ):
        """
        Set informants of this particle

        Parameters
        ----------
        swarm_size : int
            swarm size of PSO
        
        index : int
            index of this particle in the swarm array

        Returns
        -------
        None.

        """
        self.informantIndices = random.sample([ i for i in range( swarm_size ) if i != index ], self.n_informants )
    
    def modify_group( self, swarm ):
        """
        modify group (this particle and informants) best position and best fitness

        Parameters
        ----------
        swarm : particle array
            array containing particles as defined in PSO

        Returns
        -------
        None.

        """
        best = swarm[ self.informantIndices[0] ].personal_best       # first is best assumption
        best_index = 0
        for i in range( 1, self.n_informants ):
            iBest = swarm[ self.informantIndices[i] ].personal_best
            if( iBest < best ):
                best = iBest
                best_index = i
        if best < self.group_best:
            self.group_best = best
            self.group_positions = swarm[ self.informantIndices[ best_index ]].positions


# ---- particle swarm optimisation
class PSO():
    
    def __init__( self, swarm_size, n_iter, n_informants, model, weights = [ .9, 1, 1.05 ], precision = 1e-3 ):
        """
        Particle Swarm Optimisation class
            to handle particle objects and minimising the loss function of the neural network

        Parameters
        ----------
        swarm_size : int
            number of particles to generate and handle the optimisation.
        n_iter : int
            number of iterations in optimising.
        n_informants : int
            number of informants per particle. should be less than swarm size.
        model : NeuralNetwork
            Neural Network configuration as argument in this class instance.
        weights : list int, optional
            coefficients for velocity calculation. The default is [ .9, 1.05, 1.1 ].
        precision : double, optional
            to end iterations if precision is higher than calculated loss. The default is 1e-3.

        Returns
        -------
        None.

        """
        
        # for algorithm usage
        self.swarm_size = swarm_size
        self.n_iter = n_iter
        self.weights = weights
        self.precision = precision
        self.n_informants = n_informants
        
        self.min_bounds = -50
        self.max_bounds = 50
        
        self.model = model
        
        self.initialise_swarm()
        
        # global best
        self.global_best_fitness = 10       # first particle set as best
        self.global_best_position = self.swarm[0].positions
        
        self.metrics = { "accuracy" : 0, "loss" : 0 }
        # self.model.update_weights( self.swarm[0].positions )
    
    def initialise_swarm( self ):
        """
        initialise swarm of particles

        Returns
        -------
        None.

        """
        self.swarm = [ Particle( self.model.weight_dims[-1], self.n_informants ) 
                      for i in range( self.swarm_size ) ]
    
    def optimise_swarm( self, X, y ):
        """
        main function of the PSO class
            loops for n_iter iterations to minimise loss function of Neural Network

        Parameters
        ----------
        X : ndarray
            input data.
        y : ndarray
            class labels.

        Returns
        -------
        None.

        """
        # calculate loss and identify informants for each particle
        for particle in self.swarm:
            particle.calculate_fitness( self.model.train( X, y ) )
            particle.set_informants( self.swarm_size, self.swarm.index( particle ))
        
        # start loop
        for i in range( self.n_iter ):
            for particle in self.swarm:
                
                # calculate velocities and identify group best per particle
                particle.calculate_velocity( self.weights )
                particle.modify_group( self.swarm )
                
                # update particle position
                particle.positions = np.sum(( particle.positions, particle.velocities ), axis = 0 )
                
                # out of bounds positions will be reset to min and max values
                for d in range( self.model.weight_dims[-1] ):
                    if particle.positions[d] < self.min_bounds:
                        particle.positions[d] == self.min_bounds
                    if particle.positions[d] < self.max_bounds:
                        particle.positions[d] == self.max_bounds
                
                # update the weights array of the Neural Network to reflect position changes
                self.model.update_weights( particle.positions )
                
                # calculate new loss
                particle.calculate_fitness( self.model.train( X, y ) )
                self.model.update_accuracy( X, y )
                
                # update personal and global best                
                if particle.fitness <= particle.personal_best:
                    # print( "particle best modified", self.model.accuracy, self.model.loss )
                    particle.personal_best = particle.fitness
                    particle.best_position = particle.positions

                if particle.fitness <= self.global_best_fitness:
                    # print( "global best modified; iter", i, self.model.accuracy, self.model.loss )
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.positions
                    self.metrics["accuracy"] = round( self.model.accuracy, 4 )
                    self.metrics["loss"] = round( self.model.loss, 4 )
                    
                    print( "global best; iter", i+1, self.metrics )
            
        
        # update weights to global best after iteration
        self.model.update_weights( self.global_best_position )
        
            
                
        

# ---- data preprocessing

# load dataset and split into training and testing sets
dataset = load_iris()
x_ = dataset.data
y_ = dataset.target

x_train, x_test, y_train, y_test = train_test_split( x_, y_, test_size = 0.3, random_state = 123, stratify = y_ )

# scale x_ values to optimise model
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform( x_train )

x_test_scaled = scaler.transform( x_test )



# ---- application

# initialise neural network
nn = NeuralNetwork( 4, 30, 3, activation_func="tanh" )

# neural network is also trained INSIDE this PSO instance
pso = PSO( 50, 500, 10, nn )
pso.optimise_swarm( x_train_scaled, y_train )

# apply trained nn to test data
y_pred = nn.predict( x_test_scaled )
test_acc = ( y_pred == y_test ).mean()

print( test_acc )


