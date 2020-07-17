import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from scipy import stats
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import MaxPooling1D, UpSampling1D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Reshape
from keras.layers import Input, Dense, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers import RepeatVector
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.models import Sequential
import numpy as np
import os
import matplotlib.pyplot as plt
import random as rand
import pandas as pd
import tensorflow
import keras
import statistics as st
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from collections import Counter


### Functions for F1 score (obtained online, have to cite), which we will use in the models later
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1(y_true, y_pred):
    def rec(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        rec = true_positives / (possible_positives + K.epsilon())
        return rec

    def prec(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        prec = true_positives / (predicted_positives + K.epsilon())
        return prec
    prec = prec(y_true, y_pred)
    rec = rec(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))