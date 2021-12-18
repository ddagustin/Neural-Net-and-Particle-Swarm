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
import random

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
    
    def __init__( self, swarm_size, n_informants, weights = [ .9, 1, 1.05 ], precision = 1e-3 ):
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
        self.weights = weights
        self.precision = precision
        self.n_informants = n_informants
        
        # global best
        self.global_best_fitness = np.Inf
        
        self.min_bounds = -50
        self.max_bounds = 50
                        
        self.metrics = { "accuracy" : 0, "loss" : 0 }
        # model.update_weights( self.swarm[0].positions )
    
    def initialise_swarm( self, dimensions ):
        """
        initialise swarm of particles

        Returns
        -------
        None.

        """
        self.swarm = [ Particle( dimensions, self.n_informants ) 
                      for i in range( self.swarm_size ) ]
    
    def optimise_swarm( self, model, X, y, n_iter ):
        """
        main function of the PSO class
            loops for n_iter iterations to minimise loss function of Neural Network

        Parameters
        ----------
        model : NeuralNetwork
            Neural Network configuration as argument in this class instance.
        X : ndarray
            input data.
        y : ndarray
            class labels.

        Returns
        -------
        None.

        """
        
        self.initialise_swarm( model.weight_dims[-1] )
        
        # calculate loss, identify informants for each particle and set initial global variables
        for particle in self.swarm:
            particle.calculate_fitness( model.train( X, y ) )
            particle.set_informants( self.swarm_size, self.swarm.index( particle ))
            if particle.fitness <= self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.positions
                
        
        # start loop
        for i in range( n_iter ):
            for particle in self.swarm:
                
                # calculate velocities and identify group best per particle
                particle.calculate_velocity( self.weights )
                particle.modify_group( self.swarm )
                
                # update particle position
                particle.positions = np.sum(( particle.positions, particle.velocities ), axis = 0 )
                
                # out of bounds positions will be reset to min and max values
                for d in range( model.weight_dims[-1] ):
                    if particle.positions[d] < self.min_bounds:
                        particle.positions[d] == self.min_bounds
                    if particle.positions[d] < self.max_bounds:
                        particle.positions[d] == self.max_bounds
                
                # update the weights array of the Neural Network to reflect position changes
                model.update_weights( particle.positions )
                
                # calculate new loss
                particle.calculate_fitness( model.train( X, y ) )
                model.update_accuracy( X, y )
                
                # update personal and global best                
                if particle.fitness <= particle.personal_best:
                    # print( "particle best modified", model.accuracy, model.loss )
                    particle.personal_best = particle.fitness
                    particle.best_position = particle.positions

                if particle.fitness <= self.global_best_fitness:
                    # print( "global best modified; iter", i, model.accuracy, model.loss )
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.positions
                    self.metrics["accuracy"] = round( model.accuracy, 4 )
                    self.metrics["loss"] = round( model.loss, 4 )
                    
                    print( "global best; iter", i+1, self.metrics )
            
        
        # update weights to global best after iteration
        model.update_weights( self.global_best_position )