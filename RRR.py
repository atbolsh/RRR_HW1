import numpy as np
from numpy.linalg import norm
import random as r

def RRR_error(v, proj1, proj2):
    b = proj2(v)
    a = proj1(2*b - v)
    return a - b

def RRR(v, proj1, proj2, trace=False, beta=0.01, cutoff = 0.0001, maxIter=10000):
    error = 1000*np.ones(np.shape(v)) #Nonzero initial error
    i = 0
    if trace:
        trajectory = []
    while norm(error) > cutoff and i < maxIter:
        i += 1
        error = RRR_error(v, proj1, proj2)
        v = v + beta*error
        if trace:
            trajectory.append(np.copy(v))
    if norm(error) > cutoff:
        print "Warning: maximum iterations exceeded, no convergence"
    if trace:
        return proj1(v), trajectory
    else:
        return proj2(v)



