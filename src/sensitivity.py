import numpy as np




def _adjoints_verify(sens, problem, data, fd_step):
    
    obj  = problem.getObj(data)
    
    ptb          = np.random.random((np.shape(problem.beta)[0]))
    ptb          = ptb / np.linalg.norm(ptb)
    problem.beta = problem.beta + ptb * fd_step

    problem.direct_solve()
    obj_ptb = problem.getObj(data)

    fd_deriv  = (obj_ptb - obj) / fd_step
    
    problem.beta = problem.beta - ptb * fd_step

    print("Discrete Adjoint Derivative  : %E"%sens.dot(ptb))
    print("Finite Difference Derivative : %E"%fd_deriv)




def _finite_differences(problem, data, fd_step):
    
    sens = np.zeros_like(problem.beta)

    obj  = problem.getObj(data)

    for iPoint in range(np.shape(problem.beta)[0]):
        
        problem.beta[iPoint] = problem.beta[iPoint] + fd_step
        
        problem.direct_solve()
        obj_ptb = problem.getObj(data)
        
        sens[iPoint]  = (obj_ptb - obj) / fd_step
    
        problem.beta[iPoint] = problem.beta[iPoint] - fd_step

    return sens
