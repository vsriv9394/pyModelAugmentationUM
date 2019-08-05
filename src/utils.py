import numpy as np
from os import path
import time
import sys

def _init_beta(problems, problem_names, folder_name, restart):
    
    sys.stdout.write("Initializing model augmentation field... ")

    for (problem, problem_name) in zip(problems, problem_names):
        
        if restart > 0:
            
            if path.exists("%s/dataset_%s/beta_%d"%(folder_name, problem_name, restart)):
                
                problem.beta = np.loadtxt("%s/dataset_%s/beta_%d"%(folder_name, problem_name, restart))

            else:
                
                sys.stdout.write("Failed (Restart file for beta not found for case %s) "%problem_name)
                sys.stdout.write("- Exiting now\n\n")
                sys.exit(0)

        else:
            
            problem.beta = np.ones_like(problem.beta)

    sys.stdout.write("Done\n\n")




def _optim_classic(fiml):
    
    t0 = time.time()

    # --- INITIALIZE --- #

    _init_beta(fiml.problems, fiml.problem_names, fiml.folder_name, fiml.restart)

    iteration = fiml.restart - 1
    
    # --- LOOP OVER ITERATIONS --- #

    for iteration in range(fiml.restart, fiml.n_iter):
        
        obj_value = 0.0
        
        # --- LOOP OVER PROBLEMS --- #

        for (problem, data, problem_name) in zip(fiml.problems, fiml.data, fiml.problem_names):

            d_beta = fiml.get_sens(problem, data)
            d_beta = d_beta / np.max(np.abs(d_beta))

            obj_value = obj_value + problem.getObj(data)

            # Update beta for every problem individually

            problem.beta = problem.beta - d_beta * fiml.step_length *\
                                          fiml.optpar1 ** (fiml.optpar2 * np.log10(iteration+1))

            if (iteration+1)%fiml.sav_iter==0:
                
                np.savetxt("%s/dataset_%s/beta_%d"%(fiml.folder_name, problem_name, iteration), problem.beta)

        t1 = time.time()
        sys.stdout.write("Iteration %9d\t\t"%iteration)
        sys.stdout.write("Objective Function %E\t\t"%obj_value)
        sys.stdout.write("Time taken %E\n"%(t1-t0))
        fiml.optim_history[iteration] = obj_value
        t0 = time.time()

    obj_value = 0.0
    for (problem, data) in zip(fiml.problems, fiml.data):
        obj_value = obj_value + problem.getObj(data)
    
    t1 = time.time()
    sys.stdout.write("Iteration %9d\t\t"%(iteration+1))
    sys.stdout.write("Objective Function %E\t\t"%obj_value)
    sys.stdout.write("Time taken %E\n"%(t1-t0))
    fiml.optim_history[iteration+1] = obj_value

    beta_target   = []
    fiml.features = []

    for problem in fiml.problems:
        beta_target.append(problem.beta)
        fiml.features.append(problem.features)

    sys.stdout.write("\nBegin ML training for FIML Classic\n")

    fiml.train(fiml.nn_params["n_epochs_long"], beta_target, verbose=1)
    fiml.save_model(iteration+1)
    
    np.savetxt("%s/optim.out"%fiml.folder_name, fiml.optim_history)
