import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lag
"""
plt.clabel
a = np.identity(3)
a = np.ones((3,3))
print(a)

b = [[2, 3, 4,],
     [3, 4, 5],
     [6, 7, 8]]
b = np.array(b)
print(b)
print(np.dot(a, b))
print(b[0,1])

def d_matrix_exp(M,k):
    eig = lag.eig(M)
    N = np.zeros((M.shape[0], M.shape[0]))
    N[0,:] = 

    return M * k
    #raise NotImplementedError()
"""
E1 = np.array([[0.28106291, 0.51904762, 0.64148366, 0.34731372],
               [0.26979199, 0.26259106, 0.06760985, 0.37285164],
               [0.52170856, 0.06242276, 0.48040518, 0.05721432],
               [0.46227145, 0.61229525, 0.36713064, 0.72517594]])
N = np.zeros((4,4))
eigvalues, eigvectors = lag.eig(E1)
for n in range(4):
    N[:,n] = eigvectors[n,:]

print("N: {}".format(N))
print(eigvalues)
print(eigvectors)
print(lag.inv(eigvectors))

N1 = eigvectors
N2 = lag.inv(N1)
D = np.zeros((4,4))
m, n= 0, 0
while(m<D.shape[0]):
    m = m + 1
    n = n + 1
    D[m-1,n-1] = eigvalues[m-1]

print("D:")
print(D)
"""
E1_zero = d_matrix_exp(E1,0)
E1_exp = d_matrix_exp(E1,10)
print(E1_zero,"\n")
print(E1_exp)
print(np.eye(4))
print(lag.matrix_power(E1,1))
"""