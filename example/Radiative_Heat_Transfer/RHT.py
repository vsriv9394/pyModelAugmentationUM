import numpy as np
from os import path
from subprocess import call
import adolc as ad
from plotting import *
import scipy.sparse.linalg as spla

class RHT:

    # Class to generate data from the Radiative heat transfer model

    def __init__(self, T_inf=5.0, npoints=129, dt=1e-2, n_iter=1000, tol=1e-8, lambda_reg=1e-5, plot=True, verbose=True, savesol=False):

        self.T_inf      = T_inf                             # Temperature of the one-dimensional body
        self.y          = np.linspace(0., 1., npoints)      # Coordinates
        self.dy2        = (self.y[1]-self.y[0])**2          # dy^2 to be used in the second derivative
        self.T          = np.zeros_like(self.y)             # Initial temperature of the body at the coordinates specified above
        self.beta       = np.ones_like(self.y)              # Augmentation profiles of the model
        self.dt         = dt                                # Time step to be used in the simulation
        self.n_iter     = n_iter                            # Maximum number of iterations to be run during direct solve
        self.tol        = tol                               # Maximum value of residual at which direct solve can be terminated
        self.plot       = plot                              # Boolean flag whether to plot the solution at the end of simulation
        self.verbose    = verbose                           # Boolean flag whether to display residual convergence on screen
        self.savesol    = savesol                           # Boolean flag whether to save the converged temperatures
        self.lambda_reg = lambda_reg                        # Regularization constant for objective function

        if path.exists("Model_solutions/solution_%d"%T_inf):
            self.T  = np.loadtxt("Model_solutions/solution_%d"%T_inf)

    #-----------------------------------------------------------------------------------------------------------------------------------

    def evalResidual(self, T, beta):

        # Evaluate the residual

        res = np.zeros_like(T, dtype=T.dtype)
        
        res[1:-1] = (T[0:-2]-2.0*T[1:-1]+T[2:])/self.dy2 + 5E-4 * beta[1:-1] * (self.T_inf**4 - T[1:-1]**4)
        

        # Apply homogeneous boundary conditions
        
        res[0]     = -T[0]
        res[-1]    = -T[-1]

        return res

    #-----------------------------------------------------------------------------------------------------------------------------------

    def getJac(self, betaJac=False):

        # Evaluate the jacobian matrix for residual

        ad.trace_on(1)

        ad_T = ad.adouble(self.T)
        if betaJac==True:
            ad_b = ad.adouble(self.beta)
        else:
            ad_b = self.beta

        ad.independent(ad_T)
        if betaJac==True:
            ad.independent(ad_b)

        ad_res = self.evalResidual(ad_T, ad_b)

        ad.dependent(ad_res)

        ad.trace_off()

        if betaJac==False:
            jac = ad.jacobian(1, self.T)
        else:
            jac = ad.jacobian(1, np.hstack((self.T, self.beta)))

        return jac

    #-----------------------------------------------------------------------------------------------------------------------------------
    
    def implicitEulerUpdate(self):

        # Update the states using implicit Euler time stepping

        res  = self.evalResidual(self.T, self.beta)
        dRdT = self.getJac()

        self.T = self.T + np.linalg.solve(np.eye(np.shape(self.y)[0])/self.dt - dRdT, res)

        return np.linalg.norm(res)

    #-----------------------------------------------------------------------------------------------------------------------------------

    def direct_solve(self):

        # Iteratively solve the equations for the model until either the tolerance is achieved or the maximum iterations have been done

        for iteration in range(self.n_iter):

            # Update the states for this iteration

            res_norm = self.implicitEulerUpdate()
            if self.verbose==True:
                print("%9d\t%E"%(iteration, res_norm))


            # Check if the residual is within tolerance, if yes, save the data and exit the simulation, if no, continue
            
            if res_norm<self.tol:
                if self.savesol==True:
                    call("mkdir -p Model_solutions", shell=True)
                    print("Saving solution to file")
                    np.savetxt("Model_solutions/solution_%d"%self.T_inf, self.T)
                break
        

        # Once the simulation is terminated, show the results if plot is True

        if self.plot==True:

            myplot("Temp_prof", self.y, self.T, '-b', 2.0, None)
            myplot("Temp_prof", self.y, np.loadtxt("True_solutions/solution_%d"%self.T_inf), '-r', 2.0, None)
            myfig("Temp_prof", "y", "Temperature", "Temperature Profile for Radiative Heat Transfer")
            myfigshow()

    #-----------------------------------------------------------------------------------------------------------------------------------

    def adjoint_solve(self, data):
        
        # Solve the discrete adjoint equation to obtain sensitivities of the objective function w.r.t. augmentation field

        dJdT, dJdb = self.getObjJac(data)

        jac  = self.getJac(betaJac=True)
        dRdt = jac[:,0:np.shape(self.T)[0]]
        dRdb = jac[:,np.shape(self.T)[0]:]
        del jac
        psi  = np.linalg.solve(dRdt.T,dJdT)
        sens = dJdb - np.matmul(dRdb.T, psi)

        return sens

    #-----------------------------------------------------------------------------------------------------------------------------------

    def getObjRaw(self, T, data, beta):

        # Objective function evaluation
        
        return np.mean((T-data)**2) + self.lambda_reg * np.mean((beta-1.0)**2)

    #-----------------------------------------------------------------------------------------------------------------------------------

    def getObj(self, data):

        # Objective function evaluation with lesser number of arguments (only the data)
        
        return self.getObjRaw(self.T, data, self.beta)

    #-----------------------------------------------------------------------------------------------------------------------------------

    def getObjJac(self, data):

        # Evaluate the jacobian matrix for the objective function

        ad.trace_on(1)

        ad_T = ad.adouble(self.T)
        ad_b = ad.adouble(self.beta)

        ad.independent(ad_T)
        ad.independent(ad_b)

        ad_obj = self.getObjRaw(ad_T, data, ad_b)

        ad.dependent(ad_obj)

        ad.trace_off()

        jac = ad.jacobian(1, np.hstack((self.T, self.beta)))

        return jac[0,0:np.shape(self.T)[0]], jac[0,np.shape(self.T)[0]:]



        







if __name__=="__main__":

    for T_inf in np.linspace(5.,50.,10):
        rht = RHT(T_inf=T_inf, savesol=True)
        rht.direct_solve()
