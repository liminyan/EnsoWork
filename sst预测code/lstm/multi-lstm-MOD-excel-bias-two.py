from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import numpy
import pandas as pd

#
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


# fit an LSTM network to training data
def fit_lstm(trainX, trainY, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = trainX, trainY
    #print(X)
    #print(y)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    model.fit(X, y, epochs=100, batch_size=n_batch, verbose=2)

    #for i in range(nb_epoch):
    #    model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
    #    model.reset_states()
    return model


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts(model, n_batch, testX, testY):
    forecasts = [0 for i in range(len(testX))]
    #print(testX)
    for i in range(len(testX)):
        X, y = testX[i][:], testY[i][:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts[i] = forecast
    return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts,n_seq):
    #print(test)
    #print("**********")
    print(forecasts)
    for i in range(n_seq):
        actual = [row[i] for row in test]
        #print(actual)
        predicted = [forecast[i] for forecast in forecasts]
        #print(predicted)
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i + 1), rmse))


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    #print(len(forecasts))
    #for i in range(len(forecasts)):
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()


# load dataset
#series = read_csv('bias.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# configure


#dataframe = read_excel('bias.csv', usecols=[1], engine='python', skipfooter=0)
#dataset = dataframe.values

fileread = "C:/Users/Admin/Desktop/bias-two.xlsx"
dataframe = pd.read_excel(fileread)
dataset = dataframe.values
# 将整型变为float
#dataset = dataset.astype('float32')
#print(dataset)
plt.plot(dataset)
plt.show()

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# convert an array of values into a dataset matrix
def create_dataset0(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def create_dataset(dataset, look_back=1, dim_y=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-dim_y-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
    for i in range(len(dataset)-look_back-dim_y-1):
        b = dataset[(i + look_back):(i + look_back + dim_y), 0]
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)


# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)


# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
#print(len(dataset))
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print(train)
# use this function to prepare the train and test datasets for modeling
look_back = 12
#print(train)
trainX, trainY = create_dataset(train, look_back,6)

testX, testY = create_dataset(test, look_back,6)
#print(trainX)
#print(trainY)
n_lag = look_back
n_seq = 6
n_train = train_size
n_test = test_size
n_epochs = 1500
n_batch = 1
n_neurons = 4
# prepare data
#scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
#print(train)
#print(test)
model = fit_lstm(trainX, trainY, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, testX, testY)
#print(forecasts)
# inverse transform forecasts and test
#forecasts = inverse_transform(dataset, forecasts, scaler, n_test + 2)
#print(forecasts)
#for i in range(len(forecasts)):
 #print(forecasts[:][i])
 #if(i==len(forecasts)):
#print("-------------------------------------------")
actual = testY
#actual = inverse_transform(dataset, actual, scaler, n_test + 2)
#for i in range(len(actual)):
 #print(actual[:][i])
# evaluate forecasts
#print(actual.shape)
#print(forecasts.shape)
#for i in range(len(forecasts)):
# print(forecasts[:][i])
#print(numpy.array(forecasts))
evaluate_forecasts(actual, forecasts,n_seq)
# plot forecasts
#plot_forecasts(dataset, forecasts, n_test + 2)
