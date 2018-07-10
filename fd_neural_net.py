# A simple neural network implementation using Keras

# import the necessary packages
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import os
import tensorflow as tf
#from sqlalchemy import create_engine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress memory warnings

PROBLEM_SIZE = 100000

FEATURES_PATH = ("fd-features-{}-ttt.csv".format(PROBLEM_SIZE))
LABELS_PATH = ("fd-labels-{}-ttt.csv".format(PROBLEM_SIZE))

X = pd.read_csv(FEATURES_PATH)
Y = pd.read_csv(LABELS_PATH) 

#X = pd.read_csv(FEATURES_PATH, iterator=True, chunksize=1000, dtype=np.float64)
#Y = pd.read_csv(LABELS_PATH, iterator=True, chunksize=1000,dtype=np.float64) 

# Memory Problem
# https://stackoverflow.com/questions/42900757/sequentially-read-huge-csv-file-in-python

# ## gives TextFileReader, which is iteratable with chunks of 10000 rows.
# X = pd.read_csv(FEATURES_PATH, iterator=True, chunksize=5000) 
# ## df is DataFrame. If error do list(tp)
# X = pd.concat(list(X), ignore_index=True) 
# ## if version 3.4, use tp
# X = pd.concat(X, ignore_index=True)


# ## gives TextFileReader, which is iteratable with chunks of 10000 rows.
# Y = pd.read_csv(FEATURES_PATH, iterator=True, chunksize=5000) 
# ## df is DataFrame. If error do list(tp)
# Y = pd.concat(list(Y), ignore_index=True) 
# ## if version 3.4, use tp
# Y = pd.concat(Y, ignore_index=True)


X_train = X.values
y_train = Y.values

print("Number of training examples:", len(y_train), len(X_train))
DATA_LENGTH = len(y_train)

# process the data to fit in a keras CNN properly
# input data needs to be (N, C, X, Y) - shaped where
# N - number of samples
# C - number of channels per sample
# (X, Y) - sample size
# Source: https://groups.google.com/forum/#!topic/keras-users/SBQBYGqFmAA

X_train = X_train.reshape((DATA_LENGTH, PROBLEM_SIZE, 1))

# Apply some pre-processing:
# standardize train features [-1,1] -> [0,1]
# rescale etc (will improve relu activation)

# Good resource on network architecture: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
model = Sequential()
# Input shape Source: https://www.quora.com/How-do-I-set-an-input-shape-in-Keras
# Input shape Source: https://stackoverflow.com/questions/44088859/keras-1d-cnn-how-to-specify-dimension-correctly
#model.add(Convolution1D(filters=256, kernel_size=1, padding='SAME', activation='linear', input_shape= (4*PROBLEM_SIZE-2, 1)))
#model.add(MaxPooling1D())
#model.add(Flatten())
#model.add(Dropout(0.2))
model.add(Flatten(input_shape=(PROBLEM_SIZE,1)))
#model.add(Dropout(0.5))
model.add(Dense(512, activation='linear'))
#model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
#model.add(Flatten())
#model.add(Dense(2048, activation='relu'))
model.add(Dense(PROBLEM_SIZE, activation='linear'))
# Check what happens if the linear activation layer is removed
#model.add(Activation('linear')) 
print(model.summary())
####################################################
# full example at: https://www.kaggle.com/alexanderlazarev/simple-keras-1d-cnn-features-split/code

#y_train = np_utils.to_categorical(y_train, nb_class)
#y_valid = np_utils.to_categorical(y_valid, nb_class)

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='mse',optimizer=sgd,metrics=['accuracy'])

# test1 = [0,-0.01,-1,0.48,-0.13,0,0.01,0.29,0.79,0.94]
# test_ar = np.array(test1)
# test = test_ar.reshape(1, 4*PROBLEM_SIZE - 2, 1)
# #X_train = X_train.reshape((DATA_LENGTH, 4*PROBLEM_SIZE - 2, 1))
# t = model.predict(test,1)
# ans = np.array([-0.29, 1.8, -7.37])
# print("Guess Vector:", t)
# print("Solution:", ans)
# print("Error:", abs(ans-t))

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
# to actiavte type in cmd: tensorboard --logdir=path/to/log-directory
# default: tensorboard --logdir=./logs

nb_epoch = 5
model.fit(X_train, y_train, epochs=nb_epoch, validation_split=0.1, batch_size=10, verbose=2, callbacks=[tensorboard])
model.save("fd_{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))