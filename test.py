import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import scipy.io as sio
import gc

sale_mont = np.array([[1,2,3,4,5],[2,3,4,np.nan,6]])
print sale_mont
nan_where = np.isnan(sale_mont)
# print np.argwhere(sale_mont==2)
sale_mont[nan_where] = 0
print sale_mont