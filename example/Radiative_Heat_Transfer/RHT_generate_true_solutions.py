import numpy as np
from subprocess import call
import adolc as ad
from plotting import *

class RHT_True:

    # Class to generate reference data (considered truth) for Radiative heat transfer

    def __init__(self, T_inf=5.0, npoints=129, dt=1e-2, n_iter=1000, tol=1e-8, plot=True):

        self.T_inf  = T_inf                             # Temperature of the one-dimensional body
        self.y      = np.linspace(0., 1., npoints)      # Coordinates
        self.dy2    = (self.y[1]-self.y[0])**2          # dy^2 to be used in the second derivative
        self.T      = np.zeros_like(self.y)             # Initial temperature of the body at the coordinates specified above
        self.dt     = dt                                # Time step to be used in the simulation
        self.n_iter = n_iter                            # Maximum number of iterations to be run during direct solve
        self.tol    = tol                               # Maximum value of residual at which direct solve can be terminated
        self.plot   = plot                              # Boolean flag whether to plot the solution at the end of simulation

    #-----------------------------------------------------------------------------------------------------------------------------------

    def getEmiss(self, T):

        # Function to ascertain the local radiative emissivity, given the temperature

        return 1e-4 * (1. + 5.*np.sin(3*np.pi*T/200.) + np.exp(0.02*T))

    #-----------------------------------------------------------------------------------------------------------------------------------

    def GaussSeidelUpdate(self, T):

        # Evaluate the residual

        res = np.zeros_like(T, dtype=T.dtype)
        T_copy = T.copy()

        emiss = self.getEmiss(T)

        T[1:-1:2] = 0.5 * (T[0:-2:2]+T[2::2]) + 0.5 * self.dy2 * ( emiss[1:-1:2] * (self.T_inf**4 - T[1:-1:2]**4) +\
                                                                             0.5 * (self.T_inf    - T[1:-1:2]   ) )
        
        T[2:-1:2] = 0.5 * (T[1:-2:2]+T[3::2]) + 0.5 * self.dy2 * ( emiss[2:-1:2] * (self.T_inf**4 - T[2:-1:2]**4) +\
                                                                             0.5 * (self.T_inf    - T[2:-1:2]   ) )
        

        # Apply homogeneous boundary conditions

        return np.linalg.norm(T - T_copy)

    #-----------------------------------------------------------------------------------------------------------------------------------

    def solve(self):

        for iteration in range(self.n_iter):

            # Update the states for this iteration

            res_norm = self.GaussSeidelUpdate(self.T)
            print("%9d\t%E"%(iteration, res_norm))


            # Check if the residual is within tolerance, if yes, save the data and exit the simulation, if no, continue
            
            if res_norm<self.tol:
                call("mkdir -p True_solutions", shell=True)
                np.savetxt("True_solutions/solution_%d"%self.T_inf, self.T)
                break
        

        # Once the simulation is terminated, show the results if plot is True

        if self.plot==True:

            myplot("Temp_prof", self.y, self.T, '-b', 2.0, None)
            myfig("Temp_prof", "y", "Temperature", "Temperature Profile for Radiative Heat Transfer")
            myfigshow()





if __name__=="__main__":

    for T_inf in np.linspace(5.,50.,10):
        rht = RHT_True(T_inf=T_inf, n_iter=100000, tol=1e-13)
        rht.solve()
