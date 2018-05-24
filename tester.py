# Tester file

# import the necessary packages
import numpy as np
from numpy import array, zeros, diag, diagflat, dot
import pandas as pd
from keras.models import Sequential, load_model
from scipy.linalg import solve

PROBLEM_SIZE = 10
DATA_LENGTH = 240000

def get_diag_matrix(x):
	size = (int) ((len(x) + 2) / 4)
	diagonal_1 = x[:size - 1]
	diagonal_2 = x[size - 1:2*size - 1]
	diagonal_3 = x[size*2 - 1:size*3 - 2]
	vector = x[-size:]
	matrix1 = np.diagflat(diagonal_1, -1)
	matrix2 = np.diagflat(diagonal_2)
	matrix3 = np.diagflat(diagonal_3, 1)
	matrix = matrix1 + matrix2
	matrix += matrix3
	return matrix

def get_vector(x):
	size = int ((len(x) + 2) / 4)
	vector = x[size*3 - 2:]
	return vector

def discretize(x):
	return np.round(x, 2)

def normalize(x):
	max = np.amax(np.abs(x))
	return x / max

def get_diagonals(x):
	diagonal_1 = np.diag(x, k = -1)
	diagonal_2 = np.diag(x, k = 0)
	diagonal_3 = np.diag(x, k = 1)
	appended = np.append(diagonal_1, diagonal_2)
	return np.append(appended, diagonal_3)

# Creates a diagonally dominant tridiagonal matrix (positive semi-definite)
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

def gaussSeidel(A, b, initX, maxIter, tolerance):
	# A is n-by-n np array
	# b is 1-dim np array with size n
	# initX is 1-dim np array with size n (initial guess)
	D = np.diag(np.diag(A))
	U = np.triu(A) - D
	L = np.tril(A) - D

	# initialize x
	if initX.all() != -1:
		x = np.copy(initX)
	else:
		x = np.zeros(b.shape)

	# GS iteration
	for ii in range(maxIter):
		error = np.linalg.norm(A.dot(x) - b)
		x = np.linalg.inv(D+L).dot(-U.dot(x) + b) 
		new_error = np.linalg.norm(A.dot(x) - b)
		if abs(new_error - error).mean() <= tolerance: # converged
			print ("iteration:", ii)
			break
	return x

def jacobi(A, b, x, N, tolerance):
	D = diag(A)
	R = A - diagflat(D)
	for i in range(N):
		error = np.linalg.norm(A.dot(x) - b)
		x = (b - dot(R,x))/D
		new_error = np.linalg.norm(A.dot(x) - b)
		if abs(new_error - error).mean() <= tolerance: # converged
			print ("iteration:", i)
			break
	return x


#matrix = 150*np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#vector = 150*np.array([1, 1, 1])

#matrix = np.array([[6, 4, 0], [5, 7, 1], [0, 3, 4]])
#vector = np.array([4, 5, -2])

# matrix = np.array([[54, 49, 0], [-39, -109, 64], [0, 99, 118]])
# vector = np.array([124, 0.7, -18])

matrix = 1/4*create_three_band_matrix(PROBLEM_SIZE, 1, 1, 1.3)
vector = 25*np.random.rand(PROBLEM_SIZE, 1)

solution = solve(matrix, vector)

matrix_max = np.max(matrix)
vector_max = np.max(vector)
matrix_norm = matrix/matrix_max
vector_norm = vector/vector_max

data_in = np.append(get_diagonals(matrix_norm), vector_norm)
test = data_in.reshape(1, 4*PROBLEM_SIZE - 2, 1)

model = load_model("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH))
model_guess = model.predict(test, 1)
model_guess /= matrix_max
model_guess *= vector_max

print("Guess Vector:", model_guess)
print("Exact Solution:", solution.T)
#print("Error Squared:", (solution.T - model_guess)**2)

#prediction = np.array(model_guess)
#prediction = prediction.reshape(3,1)

print("Solving using Gauss Seidel...")
init_guess = np.zeros(PROBLEM_SIZE)
print("With initial guess = 0")
print(gaussSeidel(matrix, vector, init_guess, 1000, 10e-6)[:,0])
print("With initial guess equal to model prediction")
print(gaussSeidel(matrix, vector, model_guess.T, 1000, 10e-6).T)

# print("Solving using Jacobi...")
# init_guess = np.zeros(PROBLEM_SIZE)
# print("With initial guess = 0")
# print(jacobi(matrix, vector, init_guess, 300000, 10e-7)[:,0])
# print("With initial guess equal to model prediction")
# print(jacobi(matrix, vector, model_guess.T, 300000, 10e-7)[:,0])
