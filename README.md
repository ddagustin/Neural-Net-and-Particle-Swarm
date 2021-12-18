# Neural-Net-and-Particle-Swarm
 application of particle swarms to optimise a neural network
 for fulfillment of COIT29224 Assessment 1 Requirements

Contains the following submodules, separaate from the main.py

1. activation_functions.py
   - contains the available functions for activation
   - dictionary for easy function calls
3. neural_network.py
   - sets up the architecture of the neural network
   - no back propagation and no error calculation (PSO takes care of this)
5. particle_swarm.py
   - algorithm for iterating the swarm to optimise the neural network
