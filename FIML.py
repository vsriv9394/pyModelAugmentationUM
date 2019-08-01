import numpy as np
import time
import sys
from os import path
sys.path.append(".")
from os import path
from subprocess import call
from Neural_Network import nn
import pickle

class FIML:
    
    # Field inversion and Machine Learning Class
    
    ################################################################################################################
    # Initialization function
    #=============================

    def __init__(self, kind="Classic", eqns=[], data=[], ftr=[], n_iter=100, folder_name="FIML",
                       restart=0, step_length=0.01, optpar1=0.1, optpar2=0.35, FD_derivs=False, FD_step_length=1e-6,
                       sav_iter=1):
        
        print("")
        print("|'''''''\    /'''''''|=====================================================================================================")
        print("|_       \  /       _|                         Field Inversion and Machine Learning Framework v1.0                         ")
        print("  |   |\  \/  /|   |                          -----------------------------------------------------                        ")
        print("  |   | \    / |   |         Developed at : Computational Aerosciences Laboratory, University of Michigan, Ann Arbor       ")
        print("  |   |  \  /  |   |    Methodology developed by: Dr. K. Duraisamy, Dr. E. J. Parish, Dr. A. P. Singh, Dr. J. R. Holland   ")
        print("|''   ''| \/ |''   ''|                                      Author : Vishal Srivastava                                     ")
        print("|_______|    |_______|=====================================================================================================")
        print("")

        
        # FIML type - "Classic", "Direct" or "Embedded"
        
        self.kind        = kind
        
        # Folder name to save files (neural network, solution files and optimization convergence) during the FIML run
        
        self.folder_name = folder_name
        
        # Equations (for every individual case) to be used for FIML
        
        self.eqns        = eqns

        # Data (for each aforementioned case) to be used for FIML

        self.data        = data
        
        # Initialize an array for features

        self.features    = ftr
        
        # Whether to use Finite Differences for sensitivity evaluation

        self.FD_derivs = FD_derivs

        # Specify FD step length
        
        self.FD_step_length = FD_step_length
        
        #------------------------------------------------------------------------
        
        # Create subfolders for each case to be used for FIML
        # Generate features for each case and append to the above list

        for i_eqn in range(len(self.eqns)):
            call("mkdir -p %s/dataset_%d" %(folder_name, i_eqn+1), shell=True)

        #------------------------------------------------------------------------
        
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

        #------------------------------------------------------------------------

        # Iteration ID at which to stop the optimization

        self.n_iter      = n_iter

        # Number of iterations after which to save inverse solution periodically

        self.sav_iter    = sav_iter

        # Restart iteration for FIML

        self.restart     = restart

        #-------------------------------------------------------------------------
        # Step Length = Maximum step length * eta1 ^ ( eta2 * log10(iteration) )
        #-------------------------------------------------------------------------

        # Maximum step length to be used to advance optimization variables
        
        self.step_length = step_length

        # Optimization parameter eta1

        self.optpar1     = optpar1

        # Optimization parameter eta2

        self.optpar2     = optpar2

        # Optimization history

        self.optim_history = np.zeros((n_iter+1))
        if restart>0:
            self.optim_history[0:restart] = np.loadtxt("%s/optim.out"%folder_name)[0:restart]

    ################################################################################################################
    # Neural Network configuration function
    #==========================================

    def configure_nn(self, n_neurons_hidden_layers):

        # Set the number of features given the feature list

        n_features = np.shape(self.features[0])[0]

        # Set the number of neurons in the input layer to the number of features
        
        network    = [n_features]

        # Append the list with the number of neurons in the hidden layers as specified

        network.extend(n_neurons_hidden_layers)

        # Set the number of neurons in the output layer to 1
        
        network.append(1)

        # Convert the list to a fortran style NumPy array

        network    = np.array(network)

        # Modify neural network parameters of the network structure and number of weights

        self.nn_params["network"] = network
        self.nn_params["weights"] = np.random.random((sum((network[0:-1]+1)*network[1:])))-0.5

    ################################################################################################################
    # Set the neural network optimizer
    #====================================

    def set_nn_optimizer(self, optimizer, update_values={}):

        # Set the name of the optimizer in the neural network parameters - "adam"

        self.nn_params["opt"] = optimizer

        # Given the optimizer name, set it up

        if optimizer=='adam':

            # Initialize the optimizer with the default parameters
            
            self.nn_params["opt_params"] = {"alpha":1e-3, "beta_1":0.9, "beta_2":0.999, "eps":1e-8, "beta_1t":1.0, "beta_2t":1.0}
            
            # Modify the optimizer parameters with the ones specified in the arguments

            for option_name in update_values:
                if option_name in self.nn_params["opt_params"]:
                    self.nn_params["opt_params"][option_name] = update_values[option_name]
                else:
                    print("Option %s not available for Neural Network Optimizer"%option_name)
                    sys.exit(0)

            # Convert the optimizer parameters list to a fortran style NumPy array

            self.nn_params["opt_params_array"] = np.array([self.nn_params["opt_params"]["alpha"],
                                                           self.nn_params["opt_params"]["beta_1"],
                                                           self.nn_params["opt_params"]["beta_2"],
                                                           self.nn_params["opt_params"]["eps"],
                                                           self.nn_params["opt_params"]["beta_1t"],
                                                           self.nn_params["opt_params"]["beta_2t"]])

        else:

            print("Optimizer not available")
            sys.exit(0)

    ################################################################################################################
    # Get sensitivity of Objective function with beta
    #===================================================

    def get_sens(self,i_eqn,check_sens=False):

        # Solve the given equation to desired convergence

        self.eqns[i_eqn].direct_solve()

        sens = np.zeros_like(np.shape(self.eqns[i_eqn].beta)[0])

        #-------------------------------------------------------------------------------------
        
        # If finite differences is selected
        
        if self.FD_derivs:

            # Get objective function for the current optimization iteration
            
            obj     = self.eqns[i_eqn].getObj(self.data[i_eqn])
            
            # Loop over every component of beta vector

            for i in range(np.shape(self.eqns[i_eqn].beta)[0]):
                
                # Perturb i-th component of the beta vector

                self.eqns[i_eqn].beta[i] = self.eqns[i_eqn].beta[i] + self.FD_step_length
                
                # Solve the direct problem
                
                self.eqns[i_eqn].direct_solve()
                
                # Evaluate the perturbed objective function
                
                obj_ptb                  = self.eqns[i_eqn].getObj(self.data[i_eqn])

                # Evalaute the sensitivity using finite differences

                sens[i]                  = (obj_ptb-obj)/self.FD_step_length
                
                # Restore the i-th component to old value
                
                self.eqns[i_eqn].beta[i] = self.eqns[i_eqn].beta[i] - self.FD_step_length

        #-------------------------------------------------------------------------------------

        # Otherwise use the adjoint solver of the equation provided

        else:

            # Evaluate sensitivity

            sens = self.eqns[i_eqn].adjoint_solve(self.data[i_eqn])

        #-------------------------------------------------------------------------------------
        
        # Check if the sensitivity is correct using finite differences
        
        if check_sens==True:
            
            # Choose a random perturbation for the beta vector and normalize

            perturb = np.random.random((np.shape(sens)[0]))
            perturb = perturb / np.linalg.norm(perturb)
            
            # Evaluate the objective function
            
            obj     = self.eqns[i_eqn].getObj(self.data[i_eqn])
            
            # Perturb beta using the perturbation defined above
            
            self.eqns[i_eqn].beta = self.eqns[i_eqn].beta + perturb * 1e-6

            # Solve the equation to required convergence

            self.eqns[i_eqn].direct_solve()

            # Evaluate as the perturbed objective
            
            obj_ptb  = self.eqns[i_eqn].getObj(self.data[i_eqn])

            # Evaluate the finite difference derivative
            
            fd_der   = (obj_ptb-obj)/1e-6
            adj_der  = sens.dot(perturb)

            # Print results

            print("Finite Difference derivative :  %E"%fd_der)
            print("Adjoint derivative :            %E"%adj_der)



        return sens

    ################################################################################################################
    # Train neural network
    #=========================

    def train(self, n_epochs, beta_target, verbose=0):

        # Train the Neural Network using the included fortran module

        nn.nn.nn_train(np.asfortranarray(self.nn_params["network"]),
                        
                       self.nn_params["act_fn"],
                       self.nn_params["loss_fn"],
                       self.nn_params["opt"],
                       
                       np.asfortranarray(self.nn_params["weights"]),
                       
                       np.asfortranarray(np.hstack(self.features)),
                       np.asfortranarray(np.hstack(beta_target)),

                       self.nn_params["batch_size"],

                       n_epochs,
                       
                       self.nn_params["train_fraction"],

                       np.asfortranarray(self.nn_params["opt_params_array"]),
                       verbose)

    ################################################################################################################
    # Predict using the neural network
    #====================================

    def predict(self):

        # Train the Neural Network

        for i_eqn in range(len(self.eqns)):

            self.eqns[i_eqn].beta = nn.nn.nn_predict(np.asfortranarray(self.nn_params["network"]),
                            
                                                     self.nn_params["act_fn"],
                                                     self.nn_params["loss_fn"],
                                                     self.nn_params["opt"],
                                                     
                                                     np.asfortranarray(self.nn_params["weights"]),
                                                     np.asfortranarray(self.features[i_eqn]),

                                                     np.asfortranarray(self.nn_params["opt_params_array"]))
        
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
    # Solve the inverse problem
    #=============================

    def inverse_solve(self):
    
        print("Running FIML")
        print("")

        # Choose a solution strategy based on the kind of FIML being used
        
        if self.kind=="Classic":
            self.inverse_solve_Classic()

        elif self.kind=="Direct":
            self.inverse_solve_Direct()

        elif self.kind=="Embedded":
            self.inverse_solve_Embedded()

        np.savetxt("%s/optim.out"%self.folder_name, self.optim_history)

    ################################################################################################################
    # Solve the inverse problem and model using FIML-Classic
    #=========================================================

    def inverse_solve_Classic(self):

        # Record the initial time

        t0 = time.time()

        # Read the restart file or in the case of a fresh start initialize beta to 1.0 at all points
        
        sys.stdout.write("Initializing model augmentation field... ")

        for i_eqn in range(len(self.eqns)):

            if self.restart > 0:

                if path.exists("%s/dataset_%d/beta_%d"%(self.folder_name, i_eqn+1, self.restart)):

                    self.eqns[i_eqn].beta = np.loadtxt("%s/dataset_%d/beta_%d"%(self.folder_name, i_eqn+1, self.restart))

                else:
                    
                    sys.stdout.write("Failed (File not found) - Exiting now\n\n")
                    sys.exit(0)

            else:

                self.eqns[i_eqn].beta = np.ones_like(self.eqns[i_eqn].beta)

        sys.stdout.write("Done\n\n")
        
        # Set the iteration number as one less than the restart iteration

        iteration = self.restart-1

        # Loop over iterations from restart to maximum number of iterations

        for iteration in range(self.restart, self.n_iter):
            
            # Set/reset the total objective function value to 0.0

            obj_value   = 0.0

            # Loop over the cases to be processed
            
            for i_eqn in range(len(self.eqns)):

                # Get the sensitivities for beta and scale as required

                d_beta    = self.get_sens(i_eqn)
                d_beta    = d_beta / np.max(np.abs(d_beta))

                # Evaluate the value of objective function and add to the total objective function (sum of all the cases)

                obj_value = obj_value + self.eqns[i_eqn].getObj(self.data[i_eqn])

                # Change beta according to steepest descent with the given step length

                self.eqns[i_eqn].beta = self.eqns[i_eqn].beta - d_beta * self.step_length * self.optpar1 ** (self.optpar2 * np.log10(iteration+1))

                '''
                # Append the beta target

                beta_target.append(self.eqns[i_eqn].beta)
                '''

                # Save the solution if a period of sav_iter has passed since the last save

                if (iteration+1)%self.sav_iter==0:

                    np.savetxt("%s/dataset_%d/beta_%d"%(self.folder_name, i_eqn+1, iteration+1), self.eqns[i_eqn].beta)

            # Record time and print the objective function and time spent for the current iteration

            t1 = time.time()
            print("Iteration %9d\t\tObjective Function %E\t\tTime taken %E"%(iteration, obj_value, t1-t0))
            self.optim_history[iteration] = obj_value

            # Record time for the next iteration

            t0 = time.time()

        # Create a list of target beta for all cases for ML training
            
        beta_target = []

        for i_eqn in range(len(self.eqns)):
            self.eqns[i_eqn].direct_solve()
            beta_target.append(self.eqns[i_eqn].beta)

        # Evaluate the total objective function and print for the final iteration

        obj_value = 0.0
        for i_eqn in range(len(self.eqns)):
            obj_value = obj_value + self.eqns[i_eqn].getObj(self.data[i_eqn])
        t1 = time.time()
        print("Iteration %9d\t\tObjective Function %E\t\tTime taken %E"%(iteration+1, obj_value, t1-t0))
        self.optim_history[self.n_iter] = obj_value

        # Train the Neural Network and save model

        print("")
        print("Begin ML training for FIML-Classic")
        print("")
            
        self.train(self.nn_params["n_epochs_long"], beta_target, verbose=1)
        self.save_model(iteration+1)

    ################################################################################################################
    # Solve the inverse problem and model using FIML-Embedded
    #==========================================================
    
    def inverse_solve_Embedded(self):

        # Record the initial time

        t0 = time.time()

        # Read the restart file or in the case of a fresh start check if an ML model exists to start in the absence
        # of which train a new network to a value of 1.0 for all points following which assign beta for all cases as
        # predicted by the neural network
        
        sys.stdout.write("Initializing model augmentation field... ")

        if self.restart > 0:

            sys.stdout.write("Loading the machine learning model for restart iteration %d... "%self.restart)
            self.load_model(self.restart)
            sys.stdout.write("Done\n\n")

        else:

            sys.stdout.write("Looking for an existing Machine Learning model that can be used... ")
            if path.exists("%s/model_%s_0"%(self.folder_name, self.kind)):
                sys.stdout.write("Found\n")
                sys.stdout.write("Loading the machine learning model... ")
                self.load_model(0)
                sys.stdout.write("Done\n\n")
            else:

                beta_target = []

                for i_eqn in range(len(self.eqns)):

                    self.eqns[i_eqn].beta = np.ones_like(self.eqns[i_eqn].beta)
                    beta_target.append(self.eqns[i_eqn].beta)

                sys.stdout.write("Not Found\n")
                sys.stdout.write("Initializing a model with random weights and training (to produce beta=1 for all features)")
                self.train(self.nn_params["n_epochs_long"], beta_target, verbose=1)
                self.save_model(0)
        
        sys.stdout.write("Initializing augmentation field... ")

        self.predict()

        sys.stdout.write("Done\n\n")

        # Set the iteration number as one less than the restart iteration

        iteration = self.restart-1

        # Loop over iterations from restart to maximum number of iterations

        for iteration in range(self.restart, self.n_iter):
            
            # Set/reset the total objective function value to 0.0

            obj_value   = 0.0

            # Set/reset beta target list to empty

            beta_target = []

            # Loop over the cases to be processed
            
            for i_eqn in range(len(self.eqns)):

                # Get the sensitivities for beta and scale as required

                d_beta    = self.get_sens(i_eqn)
                d_beta    = d_beta / np.max(np.abs(d_beta))

                # Evaluate the value of objective function and add to the total objective function (sum of all the cases)

                obj_value = obj_value + self.eqns[i_eqn].getObj(self.data[i_eqn])

                # Change beta according to steepest descent with the given step length

                self.eqns[i_eqn].beta = self.eqns[i_eqn].beta - d_beta * self.step_length * self.optpar1 ** (self.optpar2 * np.log10(iteration+1))

                # Append the beta target

                beta_target.append(self.eqns[i_eqn].beta)

            # Record time and print the objective function and time spent for the current iteration

            t1 = time.time()
            print("Iteration %9d\t\tObjective Function %E\t\tTime taken %E"%(iteration, obj_value, t1-t0))
            self.optim_history[iteration] = obj_value

            print("-----------------------------------------------------------------------------------------------------------------")
            print("")
            print("Machine Learning convergence")
            print("")

            # Record time for the next iteration

            t0 = time.time()

            # Train the machine learning model and set beta for all problems as the prediction from the modified network

            self.train(self.nn_params["n_epochs_short"], beta_target, verbose=0)
            self.predict()
            print("")
                
            # Save the solution if a period of sav_iter has passed since the last save

            if (iteration+1)%self.sav_iter==0:

                self.save_model(iteration+1)

        # Evaluate the total objective function and print for the final iteration

        obj_value = 0.0
        for i_eqn in range(len(self.eqns)):
            obj_value = obj_value + self.eqns[i_eqn].getObj(self.data[i_eqn])
        t1 = time.time()
        print("Iteration %9d\t\tObjective Function %E\t\tTime taken %E"%(iteration+1, obj_value, t1-t0))
        self.optim_history[self.n_iter] = obj_value

    ################################################################################################################
    # Solve the inverse problem and model using FIML-Direct
    #=========================================================

    def inverse_solve_Direct(self):

        # Record the initial time

        t0 = time.time()

        # Set verbosity of Machine Learning to 0 (Don't print convergence)

        MLverbose = 0
        
        # Read the restart file or in the case of a fresh start check if an ML model exists to start in the absence
        # of which train a new network to a value of 1.0 for all points following which assign beta for all cases as
        # predicted by the neural network
        
        sys.stdout.write("Initializing model augmentation field... ")

        if self.restart > 0:

            sys.stdout.write("Loading the machine learning model for restart iteration %d... "%self.restart)
            self.load_model(self.restart)
            sys.stdout.write("Done\n\n")

        else:

            sys.stdout.write("Looking for an existing Machine Learning model that can be used... ")
            if path.exists("%s/model_%s_0"%(self.folder_name, self.kind)):
                sys.stdout.write("Found\n")
                sys.stdout.write("Loading the machine learning model... ")
                self.load_model(0)
                sys.stdout.write("Done\n\n")
            else:

                beta_target = []

                for i_eqn in range(len(self.eqns)):

                    self.eqns[i_eqn].beta = np.ones_like(self.eqns[i_eqn].beta)
                    beta_target.append(self.eqns[i_eqn].beta)

                MLverbose = 1
                sys.stdout.write("Not Found\n")
                sys.stdout.write("Initializing a model with random weights and training (to produce beta=1 for all features)")
                self.train(self.nn_params["n_epochs_long"], beta_target, verbose=1)
                self.save_model(0)
       
        sys.stdout.write("Initializing augmentation field... ")

        self.predict()

        sys.stdout.write("Done\n\n")

        # Set the iteration number as one less than the restart iteration

        iteration = self.restart-1

        # Loop over iterations from restart to maximum number of iterations

        for iteration in range(self.restart, self.n_iter):
            
            # Set/reset the total objective function value to 0.0

            obj_value   = 0.0

            # Set/reset beta target list to empty

            beta_target = []

            # Loop over the cases to be processed
            
            for i_eqn in range(len(self.eqns)):

                # Evaluate the sensitivities of objective function w.r.t. beta

                d_beta    = self.get_sens(i_eqn)

                # Evaluate the sensitivities of objective function w.r.t. weights

                d_weights = nn.nn.nn_get_weights_sens(np.asfortranarray(self.nn_params["network"]),
                                                      
                                                      self.nn_params["act_fn"],
                                                      self.nn_params["loss_fn"],
                                                      self.nn_params["opt"],
                                                      
                                                      np.asfortranarray(self.nn_params["weights"]),
                                                      np.asfortranarray(self.features[i_eqn]),
                                                      
                                                      1,
                                                      np.shape(self.eqns[i_eqn].beta)[0],
                                                      
                                                      np.asfortranarray(d_beta),
                                                      np.asfortranarray(self.nn_params["opt_params_array"]))
                
                # Evaluate the value of objective function and add to the total objective function (sum of all the cases)

                obj_value = obj_value + self.eqns[i_eqn].getObj(self.data[i_eqn])

                # Scale weights' sensitivities as required

                d_weights = d_weights / np.max(np.abs(d_weights))

                # Change weights according to steepest descent with the specified step length

                self.nn_params["weights"] = self.nn_params["weights"] - d_weights * self.step_length * self.optpar1 ** (self.optpar2 * np.log10(iteration+1))

            # Set beta for all problems as the prediction from the modified network

            self.predict()
                
            # Save the solution if a period of sav_iter has passed since the last save

            if (iteration+1)%self.sav_iter==0:

                self.save_model(iteration+1)

            # Record time and print the objective function and time spent for the current iteration

            t1 = time.time()
            print("Iteration %9d\t\tObjective Function %E\t\tTime taken %E"%(iteration, obj_value, t1-t0))
            self.optim_history[iteration] = obj_value

            # Record time for the next iteration

            t0 = time.time()

        # Evaluate the total objective function and print for the final iteration

        obj_value = 0.0
        for i_eqn in range(len(self.eqns)):
            obj_value = obj_value + self.eqns[i_eqn].getObj(self.data[i_eqn])
        t1 = time.time()
        print("Iteration %9d\t\tObjective Function %E\t\tTime taken %E"%(iteration+1, obj_value, t1-t0))
        self.optim_history[self.n_iter] = obj_value

    ################################################################################################################

    ################################################################################################################
    def chk_adj_derivatives(self, eqn_test_list=[]):
    
        if eqn_test_list:
            print("Checking accuracy of the adjoint framework for the specified problems")
            print("----------------------------------------------------------------------")
            print("")
            for i_eqn in eqn_test_list:
                print("Problem %d:"%i_eqn)
                self.get_sens(i_eqn, check_sens=True)
                print("")
            print("")

if __name__=="__main__":

    fiml = FIML()
