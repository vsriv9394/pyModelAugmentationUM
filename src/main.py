import numpy as np
import sys
from os import path
sys.path.append(".")
from os import path
from subprocess import call
import pickle

from neural_network_functions import _configure_nn, _set_nn_optimizer, _train, _predict
from sensitivity import _finite_differences, _adjoints_verify
from utils import _init_beta, _optim_classic

class FIML:
    
    # Field inversion and Machine Learning Class
    
    ################################################################################################################
    # Initialization function
    #=============================

    def __init__(self, kind           = "Classic", 
                       problems       = [], 
                       data           = [], 
                       problem_names  = [],
                       n_iter         = 100, 
                       folder_name    = "FIML",
                       restart        = 0, 
                       step_length    = 0.01, 
                       optpar1        = 0.1, 
                       optpar2        = 0.35, 
                       FD_derivs      = False, 
                       FD_step_length = 1e-6,
                       sav_iter       = 1):
        
        print("")
        print("=====================================================================================================")#|'''''''\    /'''''''|
        print("                         Field Inversion and Machine Learning Framework v1.0                         ")#|_       \  /       _|
        print("                        -----------------------------------------------------                        ")#  |   |\  \/  /|   |  
        print("       Developed at : Computational Aerosciences Laboratory, University of Michigan, Ann Arbor       ")#  |   | \    / |   |  
        print("  Methodology developed by: Dr. K. Duraisamy, Dr. E. J. Parish, Dr. A. P. Singh, Dr. J. R. Holland   ")#  |   |  \  /  |   |  
        print("                                      Author : Vishal Srivastava                                     ")#|''   ''| \/ |''   ''|
        print("=====================================================================================================")#|_______|    |_______|
        print("")

        
        self.kind           = kind                         # FIML Type - "Classic", "Direct" or "Embedded"
        self.problems       = problems                     # List of problem classes to be used for direct and inverse runs
        self.data           = data                         # List of truth data for use in objective function evaluations for each of the problems
        self.problem_names  = problem_names                # List of names for each of the problems
        self.n_iter         = n_iter                       # Maximum number of optimization iterations for FIML
        self.folder_name    = folder_name                  # Folder name to save files for this FIML run
        self.restart        = restart                      # Iteration number from which to restart the FIML run
        self.step_length    = step_length                  # Initial step length (s0) for optimization during FIML run :: Step length (s) = s0 * t1 ^ (t2 * log( iter+1 ))
        self.optpar1        = optpar1                      # Step length parameter (t1) for optimization during FIML run :: Step length (s) = s0 * t1 ^ (t2 * log( iter+1 ))
        self.optpar2        = optpar2                      # Step length parameter (t2) for optimization during FIML run :: Step length (s) = s0 * t1 ^ (t2 * log( iter+1 ))
        self.FD_derivs      = FD_derivs                    # Whether or not to use Finite difference derivatives during evaluation of gradient of objective function w.r.t. augmentation variables
        self.FD_step_length = FD_step_length               # Step length to use while evaluating the finite difference gradients
        self.sav_iter       = sav_iter                     # Number of iterations after which to save the augmentation field/Machine Learning details
        
        # Create subfolders for each case to be used for FIML
        # Generate features for each case and append to the above list

        if self.kind=="Classic":
            for problem_name in self.problem_names:
                call("mkdir -p %s/dataset_%s" %(folder_name, problem_name), shell=True)

        # Initialize the neural network parameters

        self.nn_params = {"network"          : None,
                          "weights"          : None,
                          
                          "batch_size"       : 20,
                          "n_epochs_short"   : 100,
                          "n_epochs_long"    : 1000,
                          "train_fraction"   : 1.0,
                          
                          "act_fn"           :"relu",
                          "loss_fn"          :"mse",
                          "opt"              : None,
                          "opt_params"       : None,
                          "opt_params_array" : None}

        # Optimization history

        self.optim_history = np.zeros((n_iter+1))
        if restart>0:
            self.optim_history[0:restart] = np.loadtxt("%s/optim.out"%folder_name)[0:restart]

        self.features = []
        for problem in self.problems:
            self.features.append(problem.features)

    ################################################################################################################
    # Neural Network configuration function
    #==========================================

    def configure_nn(self, n_neurons_hidden_layers):

        n_features = np.shape(self.features[0])[0]
        self.nn_params["network"], self.nn_params["weights"] = _configure_nn(n_features, n_neurons_hidden_layers)

    ################################################################################################################
    # Set the neural network optimizer
    #====================================

    def set_nn_optimizer(self, optimizer, update_values={}):

        self.nn_params = _set_nn_optimizer(self.nn_params, optimizer, update_values)

    ################################################################################################################
    # Train neural network
    #=========================

    def train(self, n_epochs, beta_target, verbose=0):

        inputs  = np.asfortranarray(np.hstack(self.features))
        outputs = np.asfortranarray(np.hstack(beta_target))

        self.nn_params = _train(self.nn_params, inputs, outputs, n_epochs, beta_target, verbose)

    ################################################################################################################
    # Predict using the neural network
    #====================================

    def predict(self):

        for problem in problems:

            problem.beta = _predict(self.nn_params, 
                                    np.asfortranarray(problem.features))

    ################################################################################################################
    # Save neural network to file corresponding to iteration
    #=========================================================

    def save_model(self, iteration):

        f = open("%s/model_%s_%d"%(self.folder_name, self.kind, iteration), "wb")
        pickle.dump(self.nn_params, f)
        f.close()

    ################################################################################################################
    # Load neural network from iteration file
    #===========================================

    def load_model(self, iteration):
        
        if path.exists("%s/model_%s_%d"%(self.folder_name, self.kind, iteration)):
            f = open("%s/model_%s_%d"%(self.folder_name, self.kind, iteration), "rb")
        else:
            print("Failed (File not found) - Exiting now\n\n")
            sys.exit(0)

        self.nn_params = pickle.load(f)
        f.close()

    ################################################################################################################
    # Get sensitivity of Objective function with beta
    #===================================================

    def get_sens(self, problem, data, check_sens=False):

        problem.direct_solve()

        if self.FD_derivs:

            sens = _finite_differences(problem, data, self.FD_step_length)

        else:

            sens = problem.adjoint_solve(data)

            if check_sens==True:
                
                _adjoints_verify(sens, problem, data, self.FD_step_length)

        return sens

    ################################################################################################################
    # Solve the inverse problem
    #=============================

    def inverse_solve(self):
    
        print("Running FIML")
        print("")

        # Choose a solution strategy based on the kind of FIML being used
        
        if self.kind=="Classic":
            _optim_classic(self)

        elif self.kind=="Direct":
            pass
            #self.inverse_solve_Direct()

        elif self.kind=="Embedded":
            pass
            #self.inverse_solve_Embedded()
