# Jacobi iterative method to solve linear system
# Jacobi implements the regular method
# SparseJacobi implements the Jacobi method using CSR storage and sparse matrix multiplication

import numpy as np
from pprint import pprint
from sparse_methods import MatrixtoCSR, SparseGet
from operations import MMultiply, SparseDot
import time

def Jacobi(A, b, error_tol=10e-6):
    t1 = time.time()
    (m, n) = np.shape(A)
    x = np.zeros_like(b)
    x_prev = np.zeros_like(b)
    error = float('Inf')
    itr = 0
    while error > error_tol:
        for i in range(m):
            sigma = MMultiply(A[i], x_prev)
            sigma -= A[i][i]*x_prev[i]
            x[i] = (b[i] - sigma)/A[i][i]
        
        x_prev = x
        error = sum(abs(MMultiply(A, x) - b))
        itr += 1
    t2 = time.time() - t1        
    return x, error, itr, t2

def SparseJacobi(Acsr, b, error_tol=10e-6):
    t1 = time.time()
    m = Acsr['m']
    x = np.zeros_like(b)
    x_prev = np.zeros_like(b)
    D = np.zeros(m, dtype=Acsr['A'].dtype)
    for i in range(m):
        D[i] = SparseGet(Acsr, i, i)
        #D = Acsr.diagonal()

    error = sum(abs(SparseDot(Acsr, x) - b))
    print (0, error)
    Error = [error]
    itr = 0
    while error > error_tol:
        for i in range(m):
            sigma = SparseDot(Acsr, x_prev, i)
            sigma -= D[i]*x_prev[i]
            x[i] = (b[i] - sigma)/D[i]
            
        x_prev = x
        error = sum(abs(SparseDot(Acsr, x) - b))
        Error.append(error)
        print (itr+1, error)
        itr += 1
    
    t2 = time.time() - t1
    print(t2)    
    return x#, Error, itr, t2

# Test cases	
if __name__ == "__main__":
    A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]], dtype=np.float64)
    Acsr = MatrixtoCSR(A)
    b = np.array([6., 25., -11., 15.])
    #x, error, itr = Jacobi(A, b, 1e-10)
    xsp = SparseJacobi(Acsr, b, 10e-6)
    #print ("Error:", error, errorsp)
    #print ("Iters:", itr, itrsp)
    print ("x =", xsp)