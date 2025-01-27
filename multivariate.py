# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:48:19 2020
Code tutorial from the following link:
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

@author: Owner
"""
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as k
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def addChanData(dataset):
    values = dataset.values.tolist()
    ind = list()
    for x in values:
        ind.append(x.index(1))
    dataset.insert(0,"Occupied Channel",ind)
    return dataset

def init(csv):
    # load data
    dataset = pd.read_csv(csv)
    names = list()
    names += [('chan%d' % (j+1)) for j in range(79)]
    dataset.columns = names
    dataset = addChanData(dataset)
    
    # save to file
    dataset.to_csv('data.csv')
    return dataset

def fillData(data,predictions):
    """
    Creates list the size of data with the values in predictions
    so the two can be graphed side by side
    
    data = orginial data
    predictions = predicted data
    """
    tot_predict = [None]*(len(data)-len(predictions))
    return tot_predict + predictions

def plotInit(dataset):
    values = dataset['Occupied Channel']
    plt.figure()
    for x in range(0,len(values),625):
        plt.hlines(values[x],x,x+624)
    # plot    
    plt.xlabel('Time (us)')
    plt.ylabel('Channel Number')
    plt.show()
    
def plotData(values,color='k',label=''):
    for x in range(0,len(values),625):
        if x ==0:
            # create label for the dataset
            plt.hlines(values[x],x,x+624,colors=color,label=label)
        else:
            plt.hlines(values[x],x,x+624,colors=color)
    
def series_to_supervised(data, n_in=1, n_out=0, dropnan=True):
    """
    	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols,names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1,i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range (0,n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
           names += [('var%d(t+%d)' % (j+1,i)) for j in range(n_vars)] 
    # put it all together
    agg = pd.concat(cols,axis=1)
    agg.colums = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#%%    
df = init('3_cycles.csv')
# plotInit(df)

# load dataset
dataset = pd.read_csv('data.csv',header=0,index_col=0)
values = dataset.values
values = values.astype('float32')

#%%
# normalize features
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

# supervised learning param
numOfLags = 1
numFeatures = 80
numFeaturesNotTime = numFeatures - 1
# frame as supervised learning
reframed = series_to_supervised(scaled, numOfLags)
# print(reframed.shape)

# split into train and test sets
values = reframed.values
percentOfTraining = 0.45
numTrain = int(percentOfTraining * reframed.shape[0])
# numTrain = 54300
train = values[:numTrain,:]
test = values[numTrain:,:]
# split into inputs and outputs
numObs = numOfLags * numFeatures
train_X, train_y = train[:, :numObs], train[:,-numFeatures]
test_X, test_y = test[:, :numObs], test[:, -numFeatures]
# print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0],numOfLags,numFeatures))
test_X = test_X.reshape((test_X.shape[0], numOfLags, numFeatures))

#design network
neurons = 80
batchSize = 5
e=50
model = k.Sequential()
model.add(k.layers.LSTM(neurons,input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(k.layers.Dense(1))
model.compile(loss='mae',optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=e, batch_size=batchSize,
                    validation_data=(test_X,test_y),verbose=2,shuffle=False)
#%%
# plot histroy
plt.figure()
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.title('3 Cycles, %.0f%% Training, Batch Size: %d, Epochs: %d' % (percentOfTraining*100,batchSize,e))
plt.legend()
plt.show()

#%%
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0],numOfLags*numFeatures))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat,test_X[:,-numFeaturesNotTime:]),axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
dfYinv = pd.DataFrame(inv_yhat)
dfYinv.to_csv('y_inv.csv')
# invert scaling for actual
test_y = test_y.reshape((len(test_y),1))
inv_y = np.concatenate((test_y,test_X[:,-numFeaturesNotTime:]),axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculatee RMSE
rmse = np.sqrt(mean_squared_error(inv_y,inv_yhat))
print('Test RMSE: %.3f' % rmse)

#%%
# plot results
# plotResults(dataset, inv_yhat)
plt.figure()
plotData(inv_y,'b','Real')
plotData(inv_yhat,'r','Model')
plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
plt.figtext(0.91,0.4,'Test RMSE: %.3f' % rmse)
plt.xlabel('Time (us)')
plt.ylabel('Channel')
plt.title('3 Cycles, %.0f%% Training, Batch Size: %d, Epochs: %d' % (percentOfTraining*100,batchSize,e))
plt.show()