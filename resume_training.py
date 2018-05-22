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

PROBLEM_SIZE = 10
DATA_LENGTH = 240000 - 1
FEATURES_PATH = ("features-{}.csv".format(PROBLEM_SIZE))
LABELS_PATH = (("labels-{}.csv".format(PROBLEM_SIZE)))

X = pd.read_csv(FEATURES_PATH)
Y = pd.read_csv(LABELS_PATH) 

X_train = X.values
y_train = Y.values
X_train = X_train.reshape((DATA_LENGTH, 4*PROBLEM_SIZE - 2, 1))

model = load_model("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))

nb_epoch = 2
model.fit(X_train, y_train, epochs=nb_epoch, validation_split=0.1, batch_size=2000, verbose=2)
model.save("{}model_{}examples.h5".format(PROBLEM_SIZE, DATA_LENGTH + 1))