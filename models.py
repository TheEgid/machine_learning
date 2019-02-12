from sklearn.model_selection import train_test_split
import pandas as pd
from time import time
import timeit
from datetime import datetime, timedelta
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adadelta, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization






