import sys
sys.path.append("..")
from Neural_Network import nn
import numpy as np

################################################################################################################
# Train neural network
#=========================

def _train(nn_params, inputs, outputs, n_epochs, beta_target, verbose):

    # Train the Neural Network using the included fortran module

    nn.nn.nn_train(nn_params["network"], nn_params["act_fn"], nn_params["loss_fn"],
                   nn_params["opt"],     nn_params["weights"],
                   
                   inputs, outputs,

                   nn_params["batch_size"],     n_epochs,
                   nn_params["train_fraction"], nn_params["opt_params_array"], verbose)

    return nn_params




################################################################################################################
# Predict using the neural network
#====================================

def _predict(nn_params, features):

    # Train the Neural Network

    return nn.nn.nn_predict ( nn_params["network"],          nn_params["act_fn"],  nn_params["loss_fn"],
                              nn_params["opt"],              nn_params["weights"], np.asfortranarray(features),
                              nn_params["opt_params_array"])
        



################################################################################################################
# Neural Network configuration function
#==========================================

def _configure_nn(n_features, n_neurons_hidden_layers):
    
    network    = [n_features]                      # Initialize the network structure by number of features
    network.extend(n_neurons_hidden_layers)        # Extend the list with a list of number of neurons in hidden layers
    network.append(1)                              # Set the output layer to have 1 neuron
    network    = np.array(network)                 # Convert the network to NumPy array
    network    = np.asfortranarray(network)        # Convert the network to fortran style array

    return network, np.asfortranarray(np.random.random((sum((network[0:-1]+1)*network[1:])))-0.5)




################################################################################################################
# Set the neural network optimizer
#====================================

def _set_nn_optimizer(nn_params, optimizer, update_values):

    # Set the name of the optimizer in the neural network parameters
    # Available options : "adam"

    nn_params["opt"] = optimizer

    # Given the optimizer name, set it up

    if optimizer=='adam':

        # Initialize the optimizer with the default parameters
        
        nn_params["opt_params"] = {"alpha"   : 1e-3, 
                                   "beta_1"  : 0.9, 
                                   "beta_2"  : 0.999, 
                                   "eps"     : 1e-8, 
                                   "beta_1t" : 1.0, 
                                   "beta_2t" : 1.0}
        
        # Modify the optimizer parameters with the ones specified in the arguments

        for option_name in update_values:
            if option_name in nn_params["opt_params"]:
                nn_params["opt_params"][option_name] = update_values[option_name]
            else:
                print("Option %s not available for Neural Network Optimizer"%option_name)
                sys.exit(0)

        # Convert the optimizer parameters list to a fortran style NumPy array

        nn_params["opt_params_array"] = np.array([nn_params["opt_params"]["alpha"],
                                                  nn_params["opt_params"]["beta_1"],
                                                  nn_params["opt_params"]["beta_2"],
                                                  nn_params["opt_params"]["eps"],
                                                  nn_params["opt_params"]["beta_1t"],
                                                  nn_params["opt_params"]["beta_2t"]])

        nn_params["opt_params_array"] = np.asfortranarray(nn_params["opt_params_array"])

        return nn_params

    else:

        print("Optimizer not available")
        sys.exit(0)
