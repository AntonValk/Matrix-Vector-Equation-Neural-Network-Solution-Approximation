# sparse_tester
# Tester file

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, zeros, diag, diagflat, dot
import pandas as pd
from keras.models import Sequential, load_model
from scipy.sparse.linalg import spsolve
import os
import tensorflow as tf
import time
from scipy.sparse import linalg
from scipy import sparse
from scipy import linalg as la
import scipy
from scipy.sparse import csr_matrix
from scipy.spatial import distance
#from iterative_solvers import sparse_gauss_seidel_scipy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
PROBLEM_SIZE = 100000
DATA_LENGTH = 1000

# Checks if the matrix x is diagonally dominant
def is_diagonally_dominant(x):
    abs_x = np.abs(x)
    return np.all(2*np.diag(abs_x) > np.sum(abs_x, axis =1 ))

def normalize(x):
	maximum_element = x.max()
	minimum_element = x.min()
	if maximum_element >= abs(minimum_element):
		return x/maximum_element
	return x/abs(minimum_element)

def get_diagonals(x):
	diagonal_1 = x.diagonal(k = -1)
	diagonal_2 = x.diagonal(k = 0)
	diagonal_3 = x.diagonal(k = 1)
	appended = np.append(diagonal_1, diagonal_2)
	return np.append(appended, diagonal_3)

def get_diag_matrix(x):
	size = (int) ((len(x) + 2) / 4)
	diagonal_1 = x[:size - 1]
	diagonal_2 = x[size - 1:2*size - 1]
	diagonal_3 = x[size*2 - 1:size*3 - 2]
	vector = x[-size:]
	matrix1 = np.diagflat(diagonal_1, -1)
	matrix2 = np.diagflat(diagonal_2)
	matrix3 = np.diagflat(diagonal_3, 1)
	#print(diagonal_1)
	#print(diagonal_2)
	#print(diagonal_3)
	matrix = matrix1 + matrix2
	matrix += matrix3
	return matrix

def get_vector(x):
	size = int ((len(x) + 2) / 4)
	vector = x[size*3 - 2:]
	#print("vector:", vector)
	return vector

# Creates a diagonally dominant tridiagonal matrix (positive semi-definite)
def create_three_band_matrix(size):
	udiag = np.ones(size)
	ldiag = np.ones(size)
	diag = -4*np.ones(size)
	matrix = scipy.sparse.dia_matrix(([diag, udiag, ldiag], [0, 1, -1]), shape=(size, size)).tocsr(copy=False)
	return matrix

def create_five_band_matrix(size, factor):
	udiag = (np.random.rand(size) + np.random.normal(0, 1, size))*np.random.choice([-1,1])
	ldiag = (np.random.rand(size) + np.random.normal(0, 1, size))*np.random.choice([-1,1])
	udiag2 = (np.random.rand(size) + np.random.normal(0, 1, size))*np.random.choice([-1,1])
	ldiag2 = (np.random.rand(size) + np.random.normal(0, 1, size))*np.random.choice([-1,1])
	diag = (abs(udiag) + abs(ldiag) + abs(udiag2) + abs(udiag2) + abs(np.random.normal(0, factor, size)))*factor*np.random.choice([-1,1]) 
	matrix = scipy.sparse.dia_matrix(([diag, udiag, ldiag, udiag2, ldiag2], [0, 1, -1, 2, -2]), shape=(size, size)).tocsr(copy=False)
	return matrix

def create_vector(size):
	vector = np.zeros(size)
	for i in range(0, size):
		vector[i] = np.random.uniform(0, 1)
	return vector

def sparse_jacobi(A, b, x, maxIter, tolerance):
	D = A.diagonal()
	LU = A - diag(D)
	for ii in range(maxIter):
		#error = A.dot(x) - b
		x = np.linalg.inv(diag(D)).dot(-LU.dot(x) + b)
		new_error = A.dot(x) - b
		if ii%5 == 0:
			print(ii, abs(new_error).mean())
		if abs(new_error).mean() <= tolerance: # converged
			print ("Converged at iteration:", ii)
			break
	return x

def Jacobi(A, b, guess, MAXITER, TOLL):    
    n = len(b)
    xk = guess 
    D = sparse.diags(A.diagonal(), 0, format = 'csr',)
    L = sparse.tril(A, format = 'csr')
    U = sparse.triu(A, format = 'csr')     
    T = -(linalg.inv(D))*(L+U)
    c = (linalg.inv(D))*b
    i = 0
    err = TOLL + 1
    while i < MAXITER:
        x = T*xk + c
        err = np.linalg.norm(x-xk, 1)/np.linalg.norm(x,1)
        xk = x
        i += 1
        if i%10 == 0: 
        	print (i, err)
        if err < TOLL:
        	print ("Converged at iteration:", i)
        	break      
    return xk



def GaussSeidel(A, b, guess, MAXITER, TOLL):
    n = len(b)
    xk = guess
    D = sparse.diags(A.diagonal(), 0, format = 'csr',)
    L = sparse.tril(A, format = 'csr')
    U = sparse.triu(A, format = 'csr')
    
    T = -(linalg.inv(D+L))* U
    c = (linalg.inv(D+L))* b
    
    i = 0
    err = TOLL + 1
    while i < MAXITER:
        x = T*xk + c
        err = np.linalg.norm(x-xk, 1)/np.linalg.norm(x,1)
        xk = x
        i += 1
        if i%10 == 0:
        	print (i, err)
        if err < TOLL:
        	print ("Converged at iteration:", i)
        	break
    return xk

# def sparse_gauss_seidel(A, b, x_k, tol=1e-6, maxiters=200):
#     """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
#     Method.
#     Inputs:
#         A ((n,n) csr_matrix): An nxn sparse CSR matrix.
#         b ((n,) ndarray): A vector of length n.
#         tol (float, opt): the convergence tolerance.
#         maxiters (int, opt): the maximum number of iterations to perform.
#     Returns:
#         x ((n,) ndarray): the solution to system Ax = b.
#     """

#     A_a = np.copy(A.toarray())
#     D = np.diag(A_a)

#     #D = sparse.diags(A.diagonal(), format = 'csr',).toarray()
#     #print(D.shape())
#     d_inv = []
#     for i in range(len(b)):
#         d_inv.append(1./D[i])

#     x_kmasuno = np.zeros_like(x_k)
#     this = False
#     tries = maxiters
#     error = []

#     while this is False and tries > 0:
#         for i in range(len(x_k)):
#             rowstart = A.indptr[i]
#             rowend = A.indptr[i+1]
#             Aix = np.dot(A.data[rowstart:rowend], x_k[A.indices[rowstart:rowend]])
#             x_kmasuno[i] = x_k[i] + d_inv[i]*(b[i] - Aix)

#         if ((la.norm((x_k - x_kmasuno))) <= tol):
#         #if abs(A.dot(x_kmasuno) - b).mean() <= tol:
#             this = True
#             difference = (A.dot( x_kmasuno ) - b)
#             error.append( la.norm( difference)) # ''', ord=np.inf'''

#         else:
#             difference = (A.dot( x_kmasuno ) - b)
#             error.append(la.norm(difference))

#         x_k = np.copy(x_kmasuno)
#         tries -= 1
#         if tries%10 == 0:
#         	print ("Iteration:", maxiters - tries)

#     #b = np.zeros_like((x_k))
#     roar = np.column_stack((b,x_k))
#     print ("Converged at iteration:", maxiters - tries)
#     return x_k

# def sparse_gauss_seidel_scipy(A, b, x,tol=1e-6, maxiters=200):
#     """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
#     Method.
#     Inputs:
#         A ((n,n) csr_matrix): An nxn sparse CSR matrix.
#         b ((n,) ndarray): A vector of length n.
#         tol (float, opt): the convergence tolerance.
#         maxiters (int, opt): the maximum number of iterations to perform.
#     Returns:
#         x ((n,) ndarray): the solution to system Ax = b.
#     """

#     #A_a = np.copy(A.toarray())
#     #D = np.diag(A_a)
#     D = A.diagonal()
#     #D = sparse.diags(A.diagonal(), format = 'csr',).toarray()

#     d_inv= []
#     for i in range(len(b)):
#         d_inv.append(1./D[i])

#     x_k = np.copy(x)                 #cambio de direccion
#     x_kmasuno = np.zeros(len(b))
#     this = False
#     tries = maxiters
#     error = []

#     while this is False and tries > 0:
#         for i in range(len(x_k)):
#             rowstart = A.indptr[i]
#             rowend = A.indptr[i+1]
#             Aix = np.dot(A.data[rowstart:rowend], x_k[A.indices[rowstart:rowend]])
#             x_kmasuno[i] = x_k[i] + d_inv[i]*(b[i] - Aix)

#         if ((la.norm((x_k - x_kmasuno), ord=np.inf)) <= tol):
#             this = True
#             difference = (A.dot( x_kmasuno ) - b)
#             error.append( la.norm( difference, ord=np.inf))

#         else:
#             difference = (A.dot( x_kmasuno ) - b)
#             error.append(la.norm( difference, ord=np.inf))

#         x_k = np.copy(x_kmasuno)
#         tries -= 1
#         if tries%10 == 0:
#             print ("Iteration:", maxiters - tries)

#     #b = np.zeros_like((x_k))
#     roar = np.column_stack((b,x_k))
#     print ("Converged at iteration:", maxiters - tries)
#     return roar[:,1:]

def sparse_gauss_seidel_scipy(A, b, x,tol=1e-6, maxiters=2500):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.
    Inputs:
        A ((n,n) csr_matrix): An nxn sparse CSR matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.
    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """

    #A_a = np.copy(A.toarray())
    #D = np.diag(A_a)
    D = A.diagonal()
    #D = sparse.diags(A.diagonal(), format = 'csr',).toarray()

    d_inv= []
    for i in range(len(b)):
        d_inv.append(1./D[i])

    x_k = np.copy(x)                 #cambio de direccion
    x_kmasuno = np.zeros(len(b))
    this = False
    tries = maxiters
    #error = []

    while this is False and tries > 0:
        for i in range(len(x_k)):
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            Aix = np.dot(A.data[rowstart:rowend], x_k[A.indices[rowstart:rowend]])
            x_kmasuno[i] = x_k[i] + d_inv[i]*(b[i] - Aix)

        #if ((la.norm((x_k - x_kmasuno), ord=np.inf)) <= tol):
        #if ((x_k**2 - x_kmasuno**2).max() <= tol):
        if (distance.euclidean(x_k, x_kmasuno) <= tol):
            this = True
            #difference = (A.dot( x_kmasuno ) - b)
            #error.append( la.norm( difference, ord=np.inf))

        #else:
            #difference = (A.dot( x_kmasuno ) - b)
            #error.append(la.norm( difference, ord=np.inf))

        if tries%100 == 0:
            print ("Iteration:", maxiters - tries, "Distance:", distance.euclidean(x_k, x_kmasuno))
        x_k = np.copy(x_kmasuno)
        tries -= 1
    #b = np.zeros_like((x_k))
    #roar = np.column_stack((b,x_k))
    print ("Converged at iteration:", maxiters - tries)
    return x_k

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def get_prediction(vector):
	#matrix_max = np.max(matrix)
	vector_max = np.max(vector)
	#matrix_norm = matrix/matrix_max
	vector_norm = vector/vector_max

	data_in = vector_norm
	test = data_in.reshape(1, PROBLEM_SIZE, 1)

	model = load_model("fd_{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH))
	model_guess = model.predict(test, 1)
	#model_guess /= matrix_max
	model_guess *= vector_max
	return model_guess

# Problem (A+N)x=b, where A is tridiagonal diag. dominant, N random noise
# The main diagonal of A has values 25+2*N(10,2), b has uniform values between [-25,25]
# The noise matrix N is a dense matrix with random values [0,7.5]
if __name__ == "__main__":

	matrix = create_three_band_matrix(PROBLEM_SIZE)

	########################################################
	# norm_A = scipy.sparse.linalg.norm(matrix)
	# norm_invA = scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(matrix))
	# cond = norm_A*norm_invA
	# print("Matrix Condition Number:", cond)
	########################################################

	vector = 100*np.random.rand(PROBLEM_SIZE, 1)
	# plt.spy(matrix)
	# plt.show()
	#noise = 1.5*np.random.rand(PROBLEM_SIZE, PROBLEM_SIZE)
	#matrix += noise
	solution = spsolve(matrix, vector)

	# matrix_max = np.max(matrix)
	# vector_max = np.max(vector)
	# matrix_norm = matrix/matrix_max
	# vector_norm = vector/vector_max

	# data_in = np.append(get_diagonals(matrix_norm), vector_norm)
	# test = data_in.reshape(1, 4*PROBLEM_SIZE - 2, 1)

	# model = load_model("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH))
	# model_guess = model.predict(test, 1)
	# model_guess /= matrix_max
	# model_guess *= vector_max

	save_sparse_csr("C1.npz", matrix)
	#b_df = pd.DataFrame(vector)
	#b_df.to_csv("b.csv", header=None, index=None)
	np.save("C1.npy", vector)
	start = time.process_time()
	model_guess = get_prediction(vector)
	print("Time:", time.process_time()-start)
	#df = pd.DataFrame(model_guess)
	#df.to_csv("model_guess.csv", header=None, index=None)
	np.save("model_guessC1.npy", model_guess)

	print("Model Guess MSE:", ((model_guess - solution.T)**2).mean())
	

	#prediction = np.array(model_guess)
	#prediction = prediction.reshape(3,1)

	# print("Solving using Gauss Seidel...")
	# init_guess = np.zeros(PROBLEM_SIZE)
	# print("With initial guess = 0")
	# start = time.process_time()
	# #sparse_gauss_seidel(matrix, vector, init_guess)
	# GaussSeidel(matrix, vector, init_guess, 2000, 10e-6).T
	# end = time.process_time()
	# print("Time:", end-start)
	# print("With initial guess equal to model prediction")
	# start = time.process_time()
	# #sparse_gauss_seidel(matrix, vector, init_guess)
	# GaussSeidel(matrix, vector, model_guess.T, 2000, 10e-6).T
	# end = time.process_time()
	# print("Time:", end-start) #doesn't account for cache

#########################################################################
	print("Solving using sparse Gauss Seidel...")
	init_guess = np.zeros(PROBLEM_SIZE)
	print("With initial guess = 0")
	start = time.process_time()
	x1 = sparse_gauss_seidel_scipy(matrix, vector, init_guess, maxiters=5000)
	#GaussSeidel(matrix, vector, init_guess, 2000, 10e-6).T
	end = time.process_time()
	print("Time:", end-start)
	print("With initial guess equal to model prediction")
	start = time.process_time()
	x2 = sparse_gauss_seidel_scipy(matrix, vector, model_guess.T, maxiters=5000)
	#GaussSeidel(matrix, vector, model_guess.T, 2000, 10e-6).T
	end = time.process_time()
	print("Time:", end-start) #doesn't account for cache

	print((solution**2 - x1**2).mean())
	print((solution**2 - x2**2).mean())

#########################################################################

	# print(model.summary())
	# print("Solving using Jacobi...") 
	# init_guess = np.zeros(PROBLEM_SIZE)
	# print("With initial guess = 0")
	# start = time.process_time()
	# sparse_jacobi(matrix, vector, init_guess, 500, 10e-6)[:,0]
	# end = time.process_time()
	# #print("Time:", end-start) #doesn't account for cache
	# print("With initial guess equal to model prediction")
	# start = time.process_time()
	# sparse_jacobi(matrix, vector, model_guess.T, 500, 10e-6)[:,0]
	# end = time.process_time()
	# #print("Time:", end-start) #doesn't account for cache