import numpy as np
from pyFluid1DUM.utils import *

def TemperatureFeatures(T_inf):
    
    y  = np.linspace(0., 1., 129)
    T  = np.loadtxt("../Model_solutions/solution_%d"%T_inf)

    f1 = y * (1.0 - y)
    f2 = T/T_inf
    
    return np.vstack((f1, f2))

Features_Dict = {"TempFtr" : TemperatureFeatures}
