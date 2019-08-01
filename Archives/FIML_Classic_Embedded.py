import numpy as np
from subprocess import call
from pyModelAugmentationUM.Inverse_Problem import Inverse_Problem

class FIML_Classic_Embedded(Inverse_Problem):
    '''

    FIML_Direct <Class>: Direct version of FIML technique for Inverse Problems

    '''

    def __init__(self, NN,             forward_problem,      objective_function, features_function,\
                       folder_prefix,  forward_problem_name, objective_name,     features_name,
                       restart_iter=0, n_iterations=100,     step = 1e-2):
        '''
    
        __init__ <Function>: Initialization function for Inverse_Problem class
        
        - args
          |
          |- NN <pyModelAugmentationUM.KerasNN.Neural_Network>: Contains the Neural Network for FIML
          |   
          |- forward_problem <Class>: Defines the forward problem to be solved
          |  |
          |  |- Required components:
          |     |
          |     |- evalResidual(states <1-D NumPy array>, beta_inv <1-D NumPy array>) <Function>: Evaluates residuals
          |     |- solve() <Function>: Solve the forward problem
          |     |- states <1-D NumPy array>: Contains states for the forward problem
          |     |- beta_inv <1-D NumPy array>: Contains beta values for model augmentation
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
          |- forward_problem_name <str>: Name of the equation the residual is for (Useful in naming the run folder)
          |
          |- folder_prefix <str>: Prefix for the folder name where all the data for this inverse problem will be stored (the run folder)
          |
          |- objective_name <str>: Name of the objective function (Useful in naming the run folder)
          |
          |- features_name <str>: Name of the feature group (Useful in naming the run folder)
        
        - kwargs
          |- n_iterations <n_iterations>: Number of iterations to run the inverse problem for
          |
          |- restart_iter <int>: Where to start the inverse problem from
          |
          |- step <float>: Initial step length for the optimization
        
        - return value
          |- None
    
        '''

        self.NN = NN
        '''

        NN <Neural_Network>: Custom version of Neural Network

        '''

        self.forward_problem = forward_problem
        '''

        forward_problem <Class>: Defines the forward problem to be solved

        Required components:
        
        - evalResidual(states <1-D NumPy array>, beta_inv <1-D NumPy array>) <Function>: Evaluates residuals
        - solve() <Function>: Solve the forward problem
        - states <1-D NumPy array>: Contains states for the forward problem
        - beta_inv <1-D NumPy array>: Contains beta values for model augmentation
        
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

        self.features = features_function(self.forward_problem)
        '''

        features <2-D NumPy array>: Features for Machine Learning 
        
        '''

        self.inv_var = self.forward_problem.beta_inv.copy()
        '''

        inv_var <1-D NumPy array>: Variables to be solved for in the inverse problem
        
        '''

        self.folder_name = "%s/%s_%s_%s"%(folder_prefix, forward_problem_name, objective_name, features_name)
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

        adaptive <bool>: Whether to use adaptive step for optimization

        '''

        call("mkdir -p %s"%self.folder_name, shell=True)
        call("mkdir -p %s/solution"%self.folder_name, shell=True)
        call("mkdir -p %s/figs"%self.folder_name, shell=True)

        if self.restart_iter!=0:

            self.loadvar(self.restart_iter)
        
    

    def getBetaInv(self, inv_var):
        '''

        getBetaInv <Function>: Evaluates augmentation variables, given inv_var

        - args:
          |- inv_var <1-D NumPy array>: Variables to be solved for in the inverse problem

        - kwargs:
          |- None

        - return value:
          |- Augmentation variables for forward solve <1-D NumPy array>

        '''
        
        return inv_var

    
    
    def solve(self):
        '''

        solve <Function>: Solves the inverse problem

        '''
        
        self.loadobjhist()

        sens = 0.0

        for iteration in range(self.restart_iter, self.restart_iter + self.n_iterations):
        
            if self.adaptive==True:
                self.inv_var             = self.inv_var - sens * self.step * self.adapcoeff1**(self.adapcoeff2 * np.log10(iteration+1))
            else:
                self.inv_var             = self.inv_var - sens * self.step 

            sens = self.getSensitivity(iteration)
            print("%6d\t%E"%(iteration, self.obj_hist[iteration]))
            self.savevar(iteration)

        #self.NN.train(self.inv_var, self.features, verbose=1, plot=False)

        self.saveobjhist()

    

    def apply(self, iteration):
        '''

        apply <Function>: Apply the augmentation to the problem

        '''

        self.loadvar(iteration)
        self.forward_problem.beta_inv = self.getBetaInv(self.inv_var)
        states = self.forward_problem.solve()
        return states
    
    
    
    def applyML(self, NN_file_name):
        '''

        apply <Function>: Apply the augmentation to the problem

        '''

        self.NN.load_model(NN_file_name)
        self.forward_problem.beta_inv = self.getBetaInv(self.NN.predict(self.features)[:,0])
        states = self.forward_problem.solve()
        return states
