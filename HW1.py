import numpy as np
from numpy.linalg import norm
import random as r

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def PA_helper(x):
    if x < -1:
        return -1
    elif -1 <= x <= 1:
        return x
    else:
        return 1

def PA(v):
    """Projection onto the square (-1, 1)x(-1,1)."""
    return np.array([PA_helper(v[0]), PA_helper(v[1])])

def PB(v):
    """Projection onto the line y = x."""
    m = (v[0] + v[1])/2.0
    return np.array([m, m])

def RRR_error(v, proj1=PA, proj2=PB):
    b = proj2(v)
    a = proj1(2*b - v)
    return a - b

def RRR(v, proj1 = PA, proj2 = PB, trace=False, beta=0.01, cutoff = 0.0001, maxIter=10000):
    error = np.array([1000, 1000]) #Nonzero initial error
    i = 0
    if trace:
        trajectory = []
    while norm(error) > cutoff and i < maxIter:
        i += 1
        error = RRR_error(v, proj1, proj2)
        v = v + beta*error
        if trace:
            trajectory.append(np.copy(v))
    if trace:
        return proj1(v), trajectory
    else:
        return proj2(v)


def unit_vector(v):
    return v/norm(v)
     
def make_background():
    fig, ax = plt.subplots(1)
    rect = Rectangle((-1, -1), 2, 2)
    pc = PatchCollection([rect], facecolor='chartreuse', edgecolor='g')
    ax.add_collection(pc)
    xs = np.linspace(-5, 5)
    plt.plot(xs, xs, 'b-')
    return fig, ax

def make_vec_field(scale=0.1):
    start = np.linspace(-5, 5, 41)
    template = [[np.array([start[i], start[j]]) for i in range(len(start))] for j in range(len(start))]
    points = []
    for line in template:
        points += line
    vecs = [scale*RRR_error(v) for v in points]
    U = [v[0] for v in vecs]
    V = [v[1] for v in vecs]
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    plt.quiver(X, Y, U, V, headwidth=2, headlength=2)
    return None 

def make_fixed():
    X = np.linspace(-1, 1, 1000)
    plt.plot(X, X, 'b.')
    return None

def prob2(initialPoints = [np.array([-0.1, -0.9]),
                  np.array([1.5, -0.5]),
                  np.array([-2.5, -3.5]),
                  np.array([-2.5, 3.5])]):
    """Draw trajectories for problem 2."""
    fig, ax = make_background()
    make_fixed()
    results = []
    traces = []
    for v in initialPoints:
        a, trajectory = RRR(v, PA, PB, True)
        results.append(a)
        traces.append(([w[0] for w in trajectory], [w[1] for w in trajectory]))
    for trace in traces:
       plt.plot(trace[0], trace[1], 'r-')
    plt.plot([u[0] for u in initialPoints], [u[1] for u in initialPoints], 'r.')
    plt.plot([v[0] for v in results], [v[1] for v in results], 'r*')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    make_vec_field()
    plt.title('Sample RRR Flow')
    plt.show()
    return None

prob2()











