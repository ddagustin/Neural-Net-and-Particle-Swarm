# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:43:17 2021

COIT29224 Evolutionary Computation
Assessment 1 - Neural Network Weight Optimisation using
                Particle Swarms

@author: Daniel Roi Agustin
"""

# ---- import dependencies
import numpy as np
from scipy.special import expit, softmax

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
    
    def der_Linear( X ):
        return 1 # WRONG! should be an array
    
    def der_Sigmoid( X ):
        return expit( X ) * ( 1 - expit( X ) )
    
    def der_Tanh( X ):
        return 1 - ( np.tanh( X ) ) ** 2
    
    def der_SoftMax( X ):
        s = softmax( X, axis =  1).reshape(-1, 1)
        return np.diagflat( softmax( X, axis =  1) ) - np.dot( s, s.T )
    
    def der_ReLu( X ):
        return ( X > 0 ) * 1
    
    

ACTIVATIONS = {
    "linear" : activationFunction.Linear,
    "sigmoid" : activationFunction.Sigmoid,
    "tanh" : activationFunction.Tanh,
    "softmax" : activationFunction.Softmax,
    "relu" : activationFunction.ReLu,
    }

DERIVATIVES = {
    "linear" : activationFunction.der_Linear,
    "sigmoid" : activationFunction.der_Sigmoid,
    "tanh" : activationFunction.der_Tanh,
    "softmax" : activationFunction.der_SoftMax,
    "relu" : activationFunction.der_ReLu,
    }


if __name__ == "__main__":
    X = np.full((150, 3), 1/3)
    
   
    error = activationFunction.der_SoftMax(X)
    print( error.shape )
    