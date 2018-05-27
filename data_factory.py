import numpy as np
import pandas as pd
import csv
from scipy.linalg import solve_banded, solve

DATA_LENGTH = 10000
PROBLEM_SIZE = 100
FEATURES_PATH = ("features-{}.csv".format(PROBLEM_SIZE))
LABELS_PATH = (("labels-{}.csv".format(PROBLEM_SIZE)))

# Checks if the matrix x is diagonally dominant
def is_diagonally_dominant(x):
    abs_x = np.abs(x)
    return np.all(2*np.diag(abs_x) >= np.sum(abs_x, axis =1 ))

def normalize(x):
	max = np.amax(np.abs(x))
	return x / max

def discretize(x):
	return np.round(x, 2)

def get_diagonals(x):
	diagonal_1 = np.diag(x, k = -1)
	diagonal_2 = np.diag(x, k = 0)
	diagonal_3 = np.diag(x, k = 1)
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


def SolveBanded(A, D):
	# Find the diagonals
	ud = np.insert(np.diag(A,1), 0, 0) # upper diagonal
	d = np.diag(A) # main diagonal
	ld = np.insert(np.diag(A,-1), len(d)-1, 0) # lower diagonal
	# simplified matrix
	ab = np.matrix([ud,d,ld,])
	return solve_banded((1, 1), ab, D )


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

def create_vector(size):
	vector = np.zeros(size)
	for i in range(0, size):
		vector[i] = np.random.uniform(0, 1)
	return vector

def file_is_empty(filename):
    with open(filename)as fin:
        for line in fin:
            line = line[:line.find('#')]  # remove '#' comments
            line = line.strip() #rmv leading/trailing white space
            if len(line) != 0:
                return False
    return True

def write_csv(filename, array, overwrite):
	if overwrite:
		with open(filename,"w+") as my_csv:
			csvWriter = csv.writer(my_csv,delimiter=',')
			csvWriter.writerow(array)
	else:
		with open(filename,"a") as my_csv:
			csvWriter = csv.writer(my_csv,delimiter=',')
			csvWriter.writerow(array)


if __name__ == "__main__":
	for i in range(0, DATA_LENGTH):
		matrix = create_three_band_matrix(PROBLEM_SIZE, 0, 100, np.random.uniform(2000, 2500))
		matrix = normalize(matrix)
		#matrix = discretize(matrix)
		vector = create_vector(PROBLEM_SIZE)
		#vector = discretize(vector)
		data_in = np.append(get_diagonals(matrix), vector)
		data_out = solve(matrix, vector)
		#data_out = discretize(solve(matrix, vector))
		#write_csv(FEATURES_PATH, data_in, True)
		#write_csv(LABELS_PATH, data_out, True)
		write_csv(FEATURES_PATH, data_in, False)
		write_csv(LABELS_PATH, data_out, False)

#array1 = np.loadtxt(open(LABELS_PATH, "rb"), delimiter=",", skiprows=0)
#array2 = np.loadtxt(open(LABELS_PATH, "rb"), delimiter=",", skiprows=1)
#array1 = np.ndarray.flatten(array1)
#array2 = np.ndarray.flatten(array2)
# df_features = pd.read_csv(FEATURES_PATH)
# df_labels = pd.read_csv(LABELS_PATH)
# labels_array = df_labels.iloc[0].values
# features_array = df_features.iloc[0].values
# print(labels_array)
# print(features_array)
# arr1 = get_diag_matrix(features_array)
# vec1 = get_vector(features_array)
# print(arr1)
# print(arr1.dot(labels_array))
# print(vec1)
#print(arr1)
#print(vec1)

#matrix = create_three_band_matrix(3, 0, 0.25, np.random.uniform(0.1, 100))
#A = discretize(normalize(matrix))
#b = discretize(create_vector(3))

# print (discretize(matrix))
# print (A)
# print (b)
# print ("The matrix is diagonally dominant:", is_diagonally_dominant(A))
# solution = SolveBanded(A, b)
#solution2 = solve(A, b)
# print ("Solution is correct:",(discretize(np.matmul(A,solution2)) == b).all())
#print(A)
#ar = np.append(get_diagonals(A), vector)
#print(ar)
#size = (len(ar) + 2) / 4
#print("Size", size)
#print(get_diag_matrix(ar))
#print(get_vector(ar))
#print (df_labels.iloc[0])
#array = df_labels.values()
#print(array)
#print("File is empty:",file_is_empty(LABELS_PATH))
