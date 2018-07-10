# Sparse Matrix Implementation
import numpy as np
import pandas as pd
import csv
import scipy
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, dia_matrix

DATA_LENGTH = 50
PROBLEM_SIZE = 100000
FEATURES_PATH = ("features-{}-t.csv".format(PROBLEM_SIZE))
LABELS_PATH = (("labels-{}-t.csv".format(PROBLEM_SIZE)))

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
def create_three_band_matrix(size, factor):
	udiag = (np.random.rand(size) + np.random.normal(0, 1, size))*np.random.choice([-1,1])
	ldiag = (np.random.rand(size) + np.random.normal(0, 1, size))*np.random.choice([-1,1])
	diag = (abs(udiag) + abs(ldiag) + abs(np.random.normal(0, factor, size)))*np.random.choice([-1,1]) 
	matrix = scipy.sparse.dia_matrix(([diag, udiag, ldiag], [0, 1, -1]), shape=(size, size)).tocsr(copy=False)
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

def create_examples(num_examples, size, factor, filename_features, filename_labels):
	for i in range(0, num_examples):
		matrix = create_three_band_matrix(size, factor)
		matrix = normalize(matrix)
		vector = create_vector(size)
		data_in = np.append(get_diagonals(matrix), vector)
		data_out = spsolve(matrix, vector)
		write_csv(filename_features, data_in, False)
		write_csv(filename_labels, data_out, False)


if __name__ == "__main__":
	for i in range (0,1):
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 1.125,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 1.125)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 1.75,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 1.75)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 2.5,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 2.5)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 3.5,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 3.5)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 4,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 4)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 4.5,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 4.5)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 5,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 5)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 8,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 8)
		create_examples(DATA_LENGTH, PROBLEM_SIZE, 10,"{}{}".format(FEATURES_PATH, i),"{}{}".format(LABELS_PATH, i))
		print("Finished", 10)


	# for i in range(0, PROBLEM_SIZE):
	# 	matrix = create_three_band_matrix(PROBLEM_SIZE, 1.2)
	# 	matrix = normalize(matrix)
	# 	#print(get_diagonals(matrix))
	# 	vector = create_vector(PROBLEM_SIZE)
	# 	data_in = np.append(get_diagonals(matrix), vector)
	# 	data_out = spsolve(matrix, vector)
	# 	#data_out = discretize(solve(matrix, vector))
	# 	#write_csv(FEATURES_PATH, data_in, True)
	# 	#write_csv(LABELS_PATH, data_out, True)
	# 	write_csv(FEATURES_PATH, data_in, False)
	# 	write_csv(LABELS_PATH, data_out, False)
