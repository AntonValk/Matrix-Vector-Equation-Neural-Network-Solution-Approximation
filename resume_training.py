# A simple neural network implementation using Keras

# import the necessary packages
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import os
import tensorflow as tf

PROBLEM_SIZE = 100000

# FEATURES_PATH = ("features-{}.csv".format(PROBLEM_SIZE))
# LABELS_PATH = (("labels-{}.csv".format(PROBLEM_SIZE)))
# # Suppress memory warnings: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# X = pd.read_csv(FEATURES_PATH)
# Y = pd.read_csv(LABELS_PATH) 

FEATURES_PATH = ("features-{}-t.csv".format(PROBLEM_SIZE))
LABELS_PATH = (("labels-{}-t.csv".format(PROBLEM_SIZE)))
# Suppress memory warnings: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


X = pd.read_csv(FEATURES_PATH)
Y = pd.read_csv(LABELS_PATH) 

X_train = X.values
y_train = Y.values
DATA_LENGTH = len(y_train)
X_train = X_train.reshape((DATA_LENGTH, 4*PROBLEM_SIZE - 2, 1))


model = load_model("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
# to actiavte type in cmd: tensorboard --logdir=path/to/log-directory
# default: tensorboard --logdir=./logs

nb_epoch = 2
model.fit(X_train, y_train, epochs=nb_epoch, validation_split=0.1, batch_size=1000, verbose=2, callbacks=[tensorboard])
model.save("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))



FEATURES_PATH = ("features-{}-1.csv".format(PROBLEM_SIZE))
LABELS_PATH = (("labels-{}-1.csv".format(PROBLEM_SIZE)))
# Suppress memory warnings: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


X = pd.read_csv(FEATURES_PATH)
Y = pd.read_csv(LABELS_PATH) 

X_train = X.values
y_train = Y.values
DATA_LENGTH = len(y_train)
X_train = X_train.reshape((DATA_LENGTH, 4*PROBLEM_SIZE - 2, 1))


model = load_model("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
# to actiavte type in cmd: tensorboard --logdir=path/to/log-directory
# default: tensorboard --logdir=./logs

nb_epoch = 2
model.fit(X_train, y_train, epochs=nb_epoch, validation_split=0.1, batch_size=1000, verbose=2, callbacks=[tensorboard])
model.save("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))