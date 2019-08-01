import numpy as np
import adolc as ad

class Inverse_Problem:
    '''

    Inverse Problem <Class>: Class containing the inverse problem to be solved

    '''



    def __init__(self, forward_solve, residual_function, objective_function, features_function,\
                                      residual_name,     objective_name,     features_name):
        '''
    
        __init__ <Function>: Initialization function for Inverse_Problem class
        
        - args
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
        
        - kwargs
          |- None
        
        - return value
          |- None
    
        '''

        self.forward_solve = forward_solve
        '''

        forward_solve <Function>: Solves the forward problem, given the augmentation variables
        
        - args: 
          |- None
        
        - kwargs:
          |- beta_inv <1-D Numpy array>: Contains the augmentation variables for forward solve (default = 1.0)
        
        - return value:
          |- states <1-D NumPy array>: Contains the state variables for the forward problem
        
        '''

        self.residual_function = residual_function
        '''
        
        residual_function <Function>: Evaluate the residual function, given the states and augmentation variables
        
        - args: 
          |- states <1-D NumPy array>: Contains the state variables for the forward problem
          |- beta_inv <1-D NumPy array>: Contains the augmented variables for the forward problem
        
        - kwargs:
          |- None
        
        - return value:
          |- Residuals for all states <1-D NumPy array>
        
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

        self.features = features_function(self.forward_solve(beta_inv=1.))
        '''

        features <2-D NumPy array>: Features for Machine Learning 
        
        '''

        self.inv_var = None
        '''

        inv_var <1-D NumPy array>: Variables to be solved for in the inverse problem
        
        '''

        self.folder_name = "%s_%s_%s"%(residual_name, objective_name, features_name)
        '''
        
        folder_name <str>: Name of the folder where all the data for a given run is stored
        
        '''

        self.n_iterations = None
        '''
        
        n_iterations <int>: Number of iterations for the optimization to run
        
        '''

        self.restart_iter = 0
        '''
        
        restart_iter <int>: Restart from this iteration (set to 0 for no restart)
        
        '''

        self.obj_hist     = None
        '''
        
        obj_hist <1-D NumPy array>: Stores the history of objective functions (includes the history before restart too)
        
        '''



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



    def getSensitivity(self, iteration):

        '''

        getSensitivity <Function>: Calculates sensitivity of objective function w.r.t. inv_var

        - args:
          |- iteration <int>: Current iteration
            
        - kwargs:
          |- None
            
        - return value:
          |- Sensitivity of objective function to inv_var <1-D NumPy array>

        '''
        
        self.forward_problem.beta_inv = self.getBetaInv(self.inv_var)
        self.forward_problem.solve()
        self.obj_hist[iteration]      = self.objective_function(self.forward_problem.states, self.forward_problem.beta_inv, self.forward_problem)

        ad_inv_var = ad.adouble(self.inv_var)
        ad_states  = ad.adouble(self.forward_problem.states)

        ad.trace_on(1)

        ad.independent(ad_states)
        ad.independent(ad_inv_var)

        ad_beta_inv = self.getBetaInv(ad_inv_var)
        ad_res      = self.forward_problem.evalResidual(ad_states, ad_beta_inv)
        ad_obj      = self.objective_function(ad_states, ad_beta_inv, self.forward_problem)

        ad.dependent(ad_res)
        ad.dependent(ad_obj)

        ad.trace_off()

        jac    = ad.jacobian(1, np.hstack((self.forward_problem.states, self.inv_var)))

        num_states = np.shape(self.forward_problem.states)[0]

        psi    = np.linalg.solve(jac[0:-1,0:num_states].T, jac[-1,0:num_states].T)
        sens   = jac[-1,num_states:] - np.matmul(psi.T, jac[0:-1,num_states:])

        return sens/np.max(np.abs(sens))
        #return sens/np.linalg.norm(np.abs(sens))



    def loadvar(self, iteration):
        '''

            loadvar <Function>: Imports inv_var from file

            - args
              |- iteration <int>

            - kwargs
              |- None
            
            - return value
              |- None

        '''

        self.inv_var = np.loadtxt("%s/solution/var.%06d"%(self.folder_name, iteration))



    def savevar(self, iteration):
        '''

            savevar <Function>: Save inv_var to file

            - args
              |- iteration <int>

            - kwargs
              |- None

            - return value
              |- None

        '''

        np.savetxt("%s/solution/var.%06d"%(self.folder_name, iteration+1), self.inv_var)



    def loadobjhist(self):
        '''

            loadobjhist <Function>: Imports obj_hist from file

            - args
              |- iteration <int>

            - kwargs
              |- None
            
            - return value
              |- None

        '''

        self.obj_hist                      = np.zeros((self.restart_iter + self.n_iterations))
        if self.restart_iter!=0:
            self.obj_hist[0:self.restart_iter] = np.loadtxt("%s/obj_hist"%self.folder_name)[0:self.restart_iter]



    def saveobjhist(self):
        '''

            saveobjhist <Function>: Save obj_hist to file

            - args
              |- iteration <int>

            - kwargs
              |- None

            - return value
              |- None

        '''

        np.savetxt("%s/obj_hist"%self.folder_name, self.obj_hist)
