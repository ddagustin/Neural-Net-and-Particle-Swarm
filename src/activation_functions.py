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

ACTIVATIONS = {
    "linear" : activationFunction.Linear,
    "sigmoid" : activationFunction.Sigmoid,
    "tanh" : activationFunction.Tanh,
    "softmax" : activationFunction.Softmax,
    "relu" : activationFunction.ReLu,
    }
   