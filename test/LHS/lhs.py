import numpy as np
import random as rd
import matplotlib.pyplot as plt
from pyDOE import *

def Sampling(Bound_low, Bound_up, n):
    # n is the number of samples
    dim = len(Bound_low)
    sample = lhs(dim, samples=n, criterion="m")
    for i in range(dim):
        length = Bound_up[i] - Bound_low[i]
        for j in range(n):
            sample[j,i] = Bound_low[i] + (length * sample[j,i])
    return sample
def LHS1d(n):
    a = np.linspace(0,1,n+1)
    b = np.arange(n)
    k = rd.sample(b.tolist(),n)
    doe = np.ones(n)
    for i in range(n):
        doe[i] = rd.random()*(a[k[i]+1]-a[k[i]])+a[k[i]]
    return np.sort(doe)

def LHS(n, samples):
    doe = LHS1d(samples)
    H = np.zeros([samples,n])
    for j in range(n):
        H[:, j] = np.random.permutation(doe)
    return H

if __name__ == '__main__':
    #print(LHS1d(10))
    #doe = LHS(2,30)
    low = [0,0]
    up = [1,1]
    doe = Sampling(low,up,30)
    plt.scatter(doe[:,0],doe[:,1])
    plt.show()
