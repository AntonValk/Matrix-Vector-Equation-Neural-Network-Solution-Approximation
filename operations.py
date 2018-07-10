# Operator functions
# MMultiply multiplies matrices. Written to ensure that any inbuilt optimal multiplication routines in python 
# are not used and the fundamental matrix multiplication is done
# SparseDot uses the CSR format to perform sparse matrix multiplication efficiently

import numpy as np
from sparse_methods import MatrixtoCSR

def MMultiply(a, b, outer=False):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a*b
        
    if len(a.shape) == 1:
        if outer:
            a = a[:, np.newaxis]
        else:
            a = a[np.newaxis, :]
    if len(b.shape) == 1:
        if outer:
            b = b[np.newaxis, :]
        else:
            b = b[:, np.newaxis]
        
    (m, n) = np.shape(a)
    (p, q) = np.shape(b)
    
    if n != p:
        raise ValueError("Dimensions of a and b incompatible")
    
    ab = np.zeros((m, q), dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            for k in range(q):
                ab[i][k] += a[i][j]*b[j][k]
    
    if ab.shape[1] == 1:
        ab = ab.transpose()[0]
   
    return ab

def SparseDot(Acsr, x, row=None):
    if row != None:
        dot = 0
        for i in range(Acsr['IA'][row], Acsr['IA'][row+1]):
            dot += Acsr['A'][i]*x[Acsr['JA'][i]]
            
        return dot
        
    m = Acsr['m']
    dot = np.zeros(m, dtype=Acsr['A'].dtype)
    for i in range(m):
        dot[i] = SparseDot(Acsr, x, i)
        
    return dot
 
# Test cases 
if __name__ == "__main__":
    """
    A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
    """
    A = np.array([[10, 20, 0, 0, 0, 0],
                  [0, 30, 0, 40, 0, 0],
                  [0, 0, 50, 60, 70, 0],
                  [0, 0, 0, 0, 0, 80]], dtype=np.float64)
                            
    b = np.array([6., 25., -11., 15., 0.3, -1])
    x = np.zeros_like(b)
    csr = MatrixtoCSR(A)
    print (b)
    print (x)
    print (MMultiply(A, b))
    print ("")
    print (np.dot(A, b))
    print ("")
    sd = SparseDot(csr, b)
    print (sd)