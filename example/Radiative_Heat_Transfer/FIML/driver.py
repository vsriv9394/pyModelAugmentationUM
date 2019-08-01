import numpy as np
import sys
sys.path.append("..")
sys.path.append("../../..")
from subprocess import call
from RHT import RHT
from FIML import FIML
from plotting import *
from Features import *

#========================================================================== PROBLEM DEFINITION =========================================================================#

# Set names of as many variables as required (Features names will be used to search in a dictionary 
#                                             of user-defined functions in the companion file of Features.py
#                                             and FIML_type is mandatory with options "Classic", "Direct" or "Embedded")

FIML_type      = "Classic"
Problem        = "RadiativeHeatTransfer"
Features_name  = "TempFtr"

# Set name of the folder where the files will be saved

folder_name    = "%s_%s_%s"%(FIML_type, Problem, Features_name)

# Create a list of parameters corresponding to the different cases that will be used in augmenting the model

#T_inf_list     = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
T_inf_list     = [50]

#============================================================================ PRE PROCESSING ===========================================================================#

# Initialize features function

ftr_fn     = Features_Dict[Features_name]

# Create a list of equations which will contain python classes containing solvers for all the cases mentioned above

eqns       = []

# Create a list of truth data arrays that will be used in objective function evaluation

data       = []

# Create a list of features for Machine Learning

features   = []

# Populate the above lists based on the parameter list above

for i_eqn in range(len(T_inf_list)):

    T_inf = T_inf_list[i_eqn]

    eqns.append(RHT(T_inf=T_inf, plot=False, verbose=False))
    data.append(np.loadtxt("../True_solutions/solution_%d"%T_inf))
    features.append(ftr_fn(T_inf))

# Define step length for optimization along with the adaptive parameters optpar1 and optpar2

step_length = 3e-3
optpar1     = 0.1
optpar2     = 0.05

# Define the restart iteration for the inverse solve

restart     = 0

# Define the maximum iteration number (including that before restart to be reached)

maxiter     = 1000

# Whether to apply post-processing

postprocess = True

# Check whether the adjoint formulation of the direct solver is accurate

#check_sens_eqn_list = [0,1,2,3,4,5,6,7,8,9]
check_sens_eqn_list = [0]

#====================================================================== CONFIGURE NEURAL NETWORK =======================================================================#

# Define the number of neurons in the hidden layers as a list

Hidden_Layers = [10]

# Choose an optimizer for the neural network and edit any default values by an update dictionary

nn_opt               = 'adam'
nn_opt_params_update = {'alpha':0.003, 'beta_1':0.7}      # A possible update would look like --->  nn_opt_params_update = {'alpha':0.01, 'beta_1':0.99, 'beta_2':0.9999, 'eps':1e-9}

# Choose an activation function, number of epochs to be trained on and batch size

act_fn               = 'sigmoid'       # Current options are - 'relu', 'sigmoid'
n_epochs_long        = 50000         # Used for FIML-Classic and to initialize FIML-Embedded and Direct
n_epochs_short       = 100          # Used during training between iterations for FIML-Embedded
batch_size           = 128
weight_factor        = 1.00

#=========================================================================== INVERSE SOLVER ============================================================================#

fiml = FIML(kind        = FIML_type, 
            eqns        = eqns,
            data        = data,
            ftr         = features, 
            n_iter      = maxiter, 
            folder_name = folder_name,
            restart     = restart,
            step_length = step_length,
            optpar1     = optpar1,
            optpar2     = optpar2)

fiml.configure_nn(Hidden_Layers)
fiml.set_nn_optimizer(nn_opt, update_values=nn_opt_params_update)
fiml.nn_params['act_fn']         = act_fn
fiml.nn_params['n_epochs_long']  = n_epochs_long
fiml.nn_params['n_epochs_short'] = n_epochs_short
fiml.nn_params['batch_size']     = batch_size
fiml.nn_params['weights']        = fiml.nn_params['weights'] * weight_factor

fiml.chk_adj_derivatives(eqn_test_list = check_sens_eqn_list)

fiml.inverse_solve()

#=========================================================================== POST PROCESSING ===========================================================================#

if postprocess==True:

    call("mkdir -p %s/figs"%fiml.folder_name, shell=True)
    
    mysemilogy(0, np.linspace(0., fiml.n_iter, fiml.n_iter+1), fiml.optim_history, '-ob', 2.0, None)
    myfig(0, "Iterations", "Objective Function", "Optimization convergence")
    myfigsave(fiml.folder_name, 0)
    
    for i_eqn in range(len(T_inf_list)):
    
        eqn = eqns[i_eqn]
    
        baseline_data = np.loadtxt("../Model_solutions/solution_%d"%T_inf_list[i_eqn])
    
        if fiml.kind=="Classic":
            legend_str = "Inverse"
        else:
            legend_str = "Inverse-ML"
    
        myplot(i_eqn+1, eqn.y,          data[i_eqn], 'ok', 1.0, 'Reference')
        myplot(i_eqn+1, eqn.y, baseline_data,        '-r', 2.0, 'Baseline')
        myplot(i_eqn+1, eqn.y,         eqn.T,        '-b', 2.0, legend_str)
        
        myplot((i_eqn+1)*10, eqn.y,         eqn.beta,     '-b', 2.0, legend_str)
    
        if fiml.kind=="Classic":
            fiml.predict()
            fiml.eqns[i_eqn].direct_solve()
            myplot(i_eqn+1, eqn.y, eqn.T, '-g', 2.0, 'ML')
            myplot((i_eqn+1)*10, eqn.y, eqn.beta, '-g', 2.0, 'ML')
        
        myfig(i_eqn+1, "$$y$$", "$$T$$", "Temperature Profile ($$T_\\infty$$=%d)"%T_inf_list[i_eqn], legend=True)
        myfig((i_eqn+1)*10, "$$y$$", "$$\\beta$$", "Augmentation ($\\beta$) profile ($$T_\\infty$$=%d)"%T_inf_list[i_eqn], legend=True)
    
        myfigsave(fiml.folder_name, i_eqn+1)
        myfigsave(fiml.folder_name, (i_eqn+1)*10)
        
        myfigshow()
