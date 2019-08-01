import numpy as np
from subprocess import call
from pyModelAugmentationUM.Inverse_Problem import Inverse_Problem

class FIML_Direct(Inverse_Problem):
    '''

    FIML_Direct <Class>: Direct version of FIML technique for Inverse Problems

    '''

    def __init__(self, NN,            forward_problem,      objective_function, features_function,\
                       folder_prefix, forward_problem_name, objective_name,     features_name):
        '''
    
        __init__ <Function>: Initialization function for Inverse_Problem class
        
        - args
          |
          |- NN <pyModelAugmentationUM.MyNN.Neural_Network>: Contains the Neural Network for FIML
          |   
          |- forward_solve <Function>: Solves the forward problem, given the augmentation variables
          |  |
          |  |- args:
          |  |  |- None
          |  |
          |  |- kwargs:
          |  |  |- beta_inv <1-D Numpy array>: Contains the augmentation variables for forward solve (default = 1.0)
          |  |
          |  |- return value:
          |     |- states <1-D NumPy array>: Contains the state variables for the forward problem
          |
          |- residual_function <Function>: Evaluate the residual function, given the states and augmentation variables
          |  |
          |  |- args: 
          |  |  |- states <1-D NumPy array>: Contains the state variables for the forward problem
          |  |  |- beta_inv <1-D NumPy array>: Contains the augmented variables for the forward problem
          |  |
          |  |- kwargs:
          |  |  |- None
          |  |
          |  |- return value:
          |     |- Residuals for all states <1-D NumPy array>
          |
          |- objective_function <Function>: Evaluate the objective function, given the states and augmentation variables
          |  |
          |  |- args: 
          |  |  |- states <1-D NumPy array>: Contains the state variables for the forward problem
          |  |  |- beta_inv <1-D NumPy array>: Contains the augmented variables for the forward problem
          |  |
          |  |- kwargs:
          |  |  |- None
          |  |
          |  |- return value:
          |     |- Value of the objective function <Scalar>
          |
          |- features_function <Function>: Evaluate the features, given the states with augmentation variables as unity
          |  |
          |  |- args: 
          |  |  |- states <1-D NumPy array>: Contains the state variables for the forward problem
          |  |
          |  |- kwargs:
          |  |  |- None
          |  |
          |  |- return value:
          |     |- Features required for Machine Learning <2-D NumPy array>
          |
          |- residual_name <str>: Name of the equation the residual is for (Useful in naming the run folder)
          |
          |- objective_name <str>: Name of the objective function (Useful in naming the run folder)
          |
          |- features_name <str>: Name of the feature group (Useful in naming the run_folder)
          |
          |- step <float>: Initial step length for the optimization
        
        - kwargs
          |- None
        
        - return value
          |- None
    
        '''

        self.NN = NN
        '''

        NN <pyModelAugmentationUM.MyNN.Neural_Network>: Custom version of Neural Network

        '''

        self.forward_problem = forward_problem
        '''

        forward_solve <Function>: Solves the forward problem, given the augmentation variables
        
        - args: 
          |- None
        
        - kwargs:
          |- beta_inv <1-D Numpy array>: Contains the augmentation variables for forward solve (default = 1.0)
        
        - return value:
          |- states <1-D NumPy array>: Contains the state variables for the forward problem
        
        '''

        self.objective_function = objective_function
        '''

        objective_function <Function>: Evaluate the objective function, given the states and augmentation variables
        
        - args: 
          |- states <1-D NumPy array>: Contains the state variables for the forward problem
          |- beta_inv <1-D NumPy array>: Contains the augmented variables for the forward problem
        
        - kwargs:
          |- None

        - return value:
          |- Value of the objective function <Scalar>
        
        '''

        self.features = features_function(self.forward_problem.solve())
        '''

        features <2-D NumPy array>: Features for Machine Learning 
        
        '''

        self.NN.init_weights(self.features)
        self.inv_var = self.NN.weights
        '''

        inv_var <1-D NumPy array>: Variables to be solved for in the inverse problem
        
        '''

        self.folder_name = "%s/%s_%s_%s"%(folder_prefix, residual_name, objective_name, features_name)
        '''
        
        folder_name <str>: Name of the folder where all the data for a given run is stored
        
        '''

        self.n_iterations = n_iterations
        '''
        
        n_iterations <int>: Number of iterations for the optimization to run
        
        '''

        self.restart_iter = restart_iter
        '''
        
        restart_iter <int>: Restart from this iteration (set to 0 for no restart)
        
        '''

        self.obj_hist     = None
        '''
        
        obj_hist <1-D NumPy array>: Stores the history of objective functions (includes the history before restart too)
        
        '''

        self.step         = step
        '''

        step <float>: Initial step length for the optimization

        '''

        self.adaptive     = True
        self.adapcoeff1   = 0.1
        self.adapcoeff2   = 0.35
        '''

        adaptive <bool>: Whether to use adaptive step size

        '''

        call("mkdir -p %s"%self.folder_name, shell=True)
        call("mkdir -p %s/solution"%self.folder_name, shell=True)
        call("mkdir -p %s/figs"%self.folder_name, shell=True)

        if self.restart_iter!=0:
            self.loadvar(self.restart_iter)
            self.NN.weights = self.inv_var
   


    def setBetaInv(self, inv_var):
        '''

        getBetaInv <Function>: Evaluates augmentation variables, given inv_var

        - args:
          |- inv_var <1-D NumPy array>: Variables to be solved for in the inverse problem

        - kwargs:
          |- None

        - return value:
          |- Augmentation variables for forward solve <1-D NumPy array>

        '''

        self.forward_problem.beta_inv = self.NN.predict(self.features, inv_var)

    
    
    def solve(self):
        '''

        solve <Function>: Solves the inverse problem

        '''
        
        self.loadobjhist()

        sens = 0.0
        
        for iteration in range(self.restart_iter, self.restart_iter + self.n_iterations):
            
            if self.adaptive==True:
                self.inv_var             = self.inv_var - sens * self.step * self.adapcoeff1**(self.adapcoeff2 * np.log(iteration+1))
            else:
                self.inv_var             = self.inv_var - sens * self.step
            
            sens                     = self.getSensitivity(iteration)
            print("%6d\t%E"%(iteration, self.obj_hist[iteration]))
            self.savevar(iteration)

        self.saveobjhist()

    

    def apply(self, iteration):
        '''

        apply <Function>: Apply the augmentation to the problem

        '''

        self.loadvar(iteration)
        self.setBetaInv(self.inv_var)
        states = self.forward_problem.solve()
        return states
