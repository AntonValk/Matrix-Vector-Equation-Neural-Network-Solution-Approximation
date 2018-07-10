# The main test script for the project
# GenSparseMatrix takes an input density and generates an nxn sparse matrix using a uniform distribution.
# It ensures that the resulting sparse matrix is diagonally dominant

import numpy as np
from scipy import sparse
from sparse_methods import *
from jacobi import *
#from LUPP import *

def GenSparseMatrix(n, density):
    dok = sparse.rand(n, n, density, format='dok', random_state=1)
    M = np.zeros((n, n))
    #print dok
    for item in dok.items():
        M[item[0][0], item[0][1]] = item[1]
        
    for i in range(n):
        rowsum = sum(M[i,:])
        rowsum -= M[i, i]
        if rowsum > M[i, i]:
            M[i, i] = rowsum+0.01
    
    return M

def create_three_band_matrix(size, mean, std, factor):
	matrix = np.eye(size, dtype = float, k = 0)
	matrix += np.eye(size, dtype = float, k = 1)
	matrix += np.eye(size, dtype = float, k = -1)
	for i in range (0, size):
		if i == 0: 
			matrix[i][i+1] *= np.random.normal(mean, std)
			matrix[i][i] *= np.abs(np.random.normal(mean, std)) + factor + np.abs(matrix [i][i+1])
			matrix[i][i] *= np.random.choice([-1,1])
		elif i == size - 1:
			matrix[i][i-1] *= np.random.normal(mean, std)
			matrix[i][i] *= np.abs(np.random.normal(mean, std)) + factor + np.abs(matrix [i][i-1])
			matrix[i][i] *= np.random.choice([-1,1])
		else:
			matrix[i][i+1] *= np.random.normal(mean, std)
			matrix[i][i-1] *= np.random.normal(mean, std)
			matrix[i][i] *= np.abs(np.random.normal(mean, std)) + factor + np.abs(matrix [i][i-1]) + np.abs(matrix[i][i+1])
			matrix[i][i] *= np.random.choice([-1,1])
	return matrix

    
if __name__ == "__main__1":
    A = GenSparseMatrix(1000, 0.2)
    b = np.ones(1000, dtype=np.float64)
    Acsr = MatrixtoCSR(A)
    #x, t = LUPP(A, b)
    #print t
    """
    x, er, it, t = Jacobi(A, b)
    print er, it, t
    """
    x = SparseJacobi(Acsr, b, 10e-6)
    print ("done")

if __name__ == "__main__":
    
    # M = np.array([[1.32465, 0, 0, 0],
    #              [5.234, 8.6, 0, 0],
    #              [0, 0, 3.456, 10],
    #              [0, 6, 0, -1.0997]])

    M = create_three_band_matrix(1000,0,10,50)
    b = np.random.rand(1000,1)
    
    # M = np.array([[10, 20, 0, 0, 0, 0],
    #               [0, 30, 0, 40, 0, 0],
    #               [0, 0, 50, 60, 70, 0],
    #               [0, 0, 0, 0, 0, 80]])
    csr = MatrixtoCSR(M)
    #pprint(csr)
    print (SparseGet(csr, 3, 3))
    m = csr['m']

    # A = np.array([[10., -1., 2., 0.],
    #           [-1., 11., -1., 3.],
    #           [2., -1., 10., -1.],
    #           [0.0, 3., -1., 8.]])
    # csr1 = MatrixtoCSR(A)
    #b = np.array([6., 25., -11., 15.])
    # print (SparseGet(csr1, 3, 3))
    #x, error, itr = Jacobi(A, b, 1e-10)
    xsp = SparseJacobi(csr, b, 10e-6)
    #print ("Error:", error, errorsp)
    #print ("Iters:", itr, itrsp)
    print ("x =", xsp)
    # D = np.zeros(m, dtype=csr['A'].dtype)
    # #D = []

    # for i in range(m):
    #     D[i] = SparseGet(csr, i, i)
    # print(D)