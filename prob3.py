import numpy as np
from numpy.linalg import norm
import random as r

import matplotlib.pyplot as plt
from PIL import Image

from RRR import *

im = 'DrawingHands.jpg'        #Filename
m = 0.0                        #Pure white
M = 255.0                      #Pure black

def image_as_array(fname):
    """Returns grayscale image as 1D array, plus original shape."""
    f = Image.open(fname)
    a2D = np.array(f)
    f.close()
    shape = np.shape(a2D)
    print shape
    #Convert to 1D floating point. Easier that way.
    a = a2D.reshape((shape[0]*shape[1],)).astype(float)
    return a, shape 

def show_array(a, shape, fname=None):
    plt.imshow(255 - a.reshape(shape), 'binary')
    if fname:
        plt.savefig(fname)
        plt.clf()
    else:
        plt.show()
    return None


#   This function is a little dense, so I will explain.
#   np.argsort returns an array of sequences such that, if b[i] == j, then
#   a.sort()[i] == a[j].
#   In other words, a[b] == a.sort()
#
#   What we need is an array c such that if c[i] == j, then 
#   a.sort()[j] == a[i]; such that a.sort()[c] == a
#   
#   This is the function 'r' from problem 3.
#
#   An easy way to do this is argsort(argsort). This puts the right index at the right index.
#   
def sort_sequence(a):
    """This is the 'r' function from the problem."""
    return np.argsort(np.argsort(a))


#   Again, this puts the correct elements of 'target' at the correct points in a.
def histogram_projection(a, target):
    s = sort_sequence(a)
    return target[s]


def main():
    a, shape = image_as_array(im)
    show_array(a, shape, 'original.pdf')
    target = np.linspace(m, M, len(a))
    fixPic = histogram_projection(a, target)
    show_array(fixPic, shape, 'rescaled.pdf')
    return fixPic.reshape(shape)

def bimodal(black_factor=1/6.0):
    """Projects to a skewed bimodal."""
    a, shape = image_as_array(im)
    show_array(a, shape, 'original.pdf')
    l = len(a)
    blacks = m*np.ones(int(black_factor*l)) #Integer division
    whites = M*np.ones(l - int(black_factor*l))
    target = np.concatenate((blacks, whites))
    fixPic = histogram_projection(a, target)
    show_array(fixPic, shape, 'bimodal.pdf')
    return fixPic.reshape(shape)


#main()
#bimodal()
