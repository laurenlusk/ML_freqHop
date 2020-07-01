# -*- coding: utf-8 -*-
"""
Spyder Editor

Code tutorial from the following link
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

Good souce for basic LSTM. Need one with multivariate though to handle freq hopper
Could use to analyze 1 freq at a time, but then they wouldn't relate
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def getSin(x):
    t = np.arange(0,x,0.1)
    amp = np.sin(t)
    return amp

def getCos(x):
    t = np.arange(0,x,0.1)
    amp = np.cos(t)
    return amp

def addNoise(pure,snr):
    """
    Adds noise to a pure signal
       pure = pure signal
       snr = signa-to-noise-ratio
       noise = Addiditve White Gaussain Noise
    """
    watts = pure**2
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - snr
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(watts))
    
    return pure+noise

def fillData(data,predictions):
    """
    Creates list the size of data with the values in predictions
    so the two can be graphed side by side
    
    data = orginial data
    predictions = predicted data
    """
    tot_predict = [None]*(len(data)-len(predictions))
    return tot_predict + predictions
    
def split_data(data, test_size):
     """
    splits data to training, validation and testing parts
    """
     ntest = int(round(len(data) * (1 - test_size)))+1
 
     train, test = data[:ntest], data[ntest:]
 
     return train,test
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df.values

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# create inveted differenced series
def inverse_diff_series(series,differenced,interval=1):
    inverted = list()
    for i in range(len(differenced)):
        value = series[i] + differenced[i]
        inverted.append(value)
    return pd.Series(inverted)

def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

def forecast_lstm(model, batch_size, row):
	X = row[0:1]
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def splitNoisyTrain(noise,data):
    train, test1 = split_data(noise, test_size=0.4)
    train1, test = split_data(data, test_size=0.4)
    orig = np.concatenate((train,test))
    return orig,train,test

def splitCleanTrain(noise,data):
    train1, test = split_data(noise, test_size=0.4)
    train, test1 = split_data(data, test_size=0.4)
    orig = np.concatenate((train,test))
    return orig,train,test
        
# generate sin wave
# data = getSin(10)
data = getCos(20)
## to add noise
# noise = addNoise(data,0.5)

# split data into tain, evaluate, test data
# train, test = split_data(pure,test_size=0.4)
train, test = split_data(data, test_size=0.4)
orig = np.concatenate((train,test))

## train on noisy data, test on clean
# orig,train,test = splitNoisyTrain(noise,data)

## train on clean data, test on noisy
# orig,train,test = splitCleanTrain(noise,data)

# transform data to stationary
train = difference(train)
test = difference(test)

# turn into supervised data
df_train = timeseries_to_supervised(train, lag=1)
df_test = timeseries_to_supervised(test, lag=1)


# create LSTM model
batchSize = 1
epoch = 1500
neurons = 1
lstm_model = fit_lstm(df_train, batchSize, epoch, neurons)
# forecast entire training dataset
train_reshaped = df_train[:, 0].reshape(len(df_train), 1, 1)
lstm_model.predict(train_reshaped, batch_size=batchSize)

# walk-forward validation on the test data
predictions = list()
for i in range(len(df_test)):
 	# make one-step forecast
 	X, y = df_test[i, 0:-1], df_test[i, -1]
 	yhat = forecast_lstm(lstm_model, 1, X)
 	# invert differencing
 	yhat = inverse_difference(data, yhat, len(df_test)+1-i)
 	# store forecast
 	predictions.append(yhat)
 	expected = data[len(df_train) + i + 1]
 	print('Time Step=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
 
# report performance
rmse = sqrt(mean_squared_error(data[-len(predictions):], predictions))
print('Test RMSE: %.3f' % rmse)

#%%
# line plot of observed vs predicted
# create list of none values to plot the predicted values along full set of data
tot_predict = fillData(data,predictions)
# plot the lines and add legend
plt.plot(orig[-len(predictions):],label='Data')
plt.plot(predictions,label='Neurons = %d, RMSE: %.3f' % (neurons, rmse))
plt.legend(loc='lower right')
plt.title('Neurons')
plt.show()


