# -*- coding: utf-8 -*-
from math import sqrt
from numpy import concatenate
import numpy as np
#from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras import optimizers
import keras
from datetime import datetime
import math
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_errip_dir(err_ip):
        if os.path.exists(err_ip):
           return
        else:
           os.makedirs(err_ip)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def load_data(file_path):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    dataset = read_csv(file_path, parse_dates=['date'], index_col='date', date_parser=dateparse)
    dataset.dropna(axis=0, how='any', inplace=True)
    dataset.index.name = 'date'
    return dataset

def normalize_and_make_series(dataset, look_back):
    values = dataset.values
    values = values.astype('float64')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    column_num = dataset.columns.size
    reframed = series_to_supervised(scaled, look_back, 1)
    # drop columns we don't want to predict
    drop_column = []
    for i in range(look_back * column_num+1, (look_back + 1) * column_num):
        drop_column.append(i)
    reframed.drop(reframed.columns[drop_column], axis=1, inplace=True)
    return reframed, scaler

def split_data(dataset, reframed, look_back):
    column_num = dataset.columns.size
    train_size = len(dataset[dataset.index < split_time])

    values = reframed.values
    train = values[:train_size, :]
    test = values[train_size:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape(train_X.shape[0], look_back, column_num)
    test_X = test_X.reshape(test_X.shape[0], look_back, column_num)

    return train_X, train_y, test_X, test_y

def build_model(look_back, train_X):
    acti_func = 'relu'
    neurons = 128
    loss = 'mse'
    #optimizer = 'sgd'
    #dropout = 0.4
    batch_size = 32   
    optimizer = 'adam' 
    model = Sequential()
    model.add(Bidirectional(LSTM(neurons,
                   activation=acti_func,
                   return_sequences=True), input_shape=(look_back, train_X.shape[2])))
    model.add(Bidirectional(LSTM(neurons,activation=acti_func)))
    model.add(Dense(1)) 
    model.add(Activation('linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def del_errip_dir(path):
    if os.path.exists(path):
       ls = os.listdir(path)
       for i in ls:
           c_path = os.path.join(path, i)
           if os.path.isdir(c_path):
              self.del_errip_dir(c_path)
           else:
              os.remove(c_path)

start_time = time.clock()


file_path = ' '
split_time = ' '
look_back =

make_errip_dir(err_ip)
del_errip_dir('./'+err_ip+'/log')

dataset = load_data(file_path)
print(dataset.head())

reframed, scaler = normalize_and_make_series(dataset, look_back)
print(reframed.head())

train_X, train_y, test_X, test_y = split_data(dataset, reframed, look_back)

print(train_X[:1])
print(train_X.shape[0],train_X.shape[1], train_X.shape[2])

batch_size = 32

model = build_model(look_back, train_X)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=0)
earlystopper = EarlyStopping(monitor='val_loss',patience=20, verbose=1)
history = model.fit(train_X, train_y, epochs=200, batch_size=batch_size, validation_data=(test_X, test_y), 
                    verbose=2, shuffle=False,callbacks=[TensorBoard(log_dir='./'+err_ip+'/log')])
print(history.history['loss'])
print(history.history['val_loss'])
#model.save('my_model.h5')
train_predict = model.predict(train_X, batch_size)

#每隔30min，重新训练一次模型
test_time = 5
test_step = int(math.ceil(len(test_X)/test_time))
test_predict = []
test_predict = model.predict(test_X, batch_size)

test_X = test_X.reshape((test_X.shape[0]*look_back, test_X.shape[2]))
test_X = test_X[:test_y.shape[0], 1:]

inv_y = np.c_[test_y, test_X]
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
print('The length of inv_y is %d'%(len(inv_y)))
# invert scaling for forecast

inv_yhat = np.c_[test_predict, test_X]
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
print('The length of inv_yhat is %d'%(len(inv_yhat)))

# calculate root mean squared error
test_score = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test Score: %.2f RMSE' % test_score) 

pred = pd.DataFrame(data=inv_yhat)
pred.to_csv(' ')

pred = pd.DataFrame(data=inv_y)
pred.to_csv(' ')

end_time = time.clock()

print(str(end_time - start_time))
