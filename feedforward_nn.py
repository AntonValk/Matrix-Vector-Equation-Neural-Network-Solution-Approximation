# A simple neural network implementation using Keras

# import the necessary packages
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress memory warnings

PROBLEM_SIZE = 100
DATA_LENGTH = 250000 - 1
FEATURES_PATH = ("features-{}.csv".format(PROBLEM_SIZE))
LABELS_PATH = (("labels-{}.csv".format(PROBLEM_SIZE)))

X = pd.read_csv(FEATURES_PATH)
Y = pd.read_csv(LABELS_PATH) 

X_train = X.values
y_train = Y.values

# process the data to fit in a keras CNN properly
# input data needs to be (N, C, X, Y) - shaped where
# N - number of samples
# C - number of channels per sample
# (X, Y) - sample size
# Source: https://groups.google.com/forum/#!topic/keras-users/SBQBYGqFmAA

X_train = X_train.reshape((DATA_LENGTH, 4*PROBLEM_SIZE - 2, 1))

# Apply some pre-processing:
# standardize train features [-1,1] -> [0,1]
# rescale etc (will improve relu activation)

# model = Sequential()

# batch_size = 128
# nb_classes = 2
# nb_epoch = 10

# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                     border_mode='valid',
#                     input_shape=(1, img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(4*PROBLEM_SIZE - 2))
# model.add(Activation('relu'))
# model.add(Dense(PROBLEM_SIZE))

# standardize train features [-1,1] -> [0,1]
# scale etc


# Good resource on network architecture: https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
model = Sequential()
# Input shape Source: https://www.quora.com/How-do-I-set-an-input-shape-in-Keras
# Input shape Source: https://stackoverflow.com/questions/44088859/keras-1d-cnn-how-to-specify-dimension-correctly
model.add(Convolution1D(filters=256, kernel_size=1, padding='SAME', activation='linear', input_shape= (4*PROBLEM_SIZE-2, 1)))
#model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
#model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(PROBLEM_SIZE, activation='linear'))
print(model.summary())
# Check what happens if the linear activation layer is removed
#model.add(Activation('linear')) 

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

nb_epoch = 3
model.fit(X_train, y_train, epochs=nb_epoch, validation_split=0.1, batch_size=1000, verbose=2, callbacks=[tensorboard])
model.save("{}model_{}examples_v3.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))