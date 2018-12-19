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


# fit an LSTM network to training data
def fit_lstm(trainX, trainY, n_batch, n_neurons):
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

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts,n_seq):
    #print(test)
    #print("**********")
    #print(forecasts)
    for i in range(n_seq):
        actual = [row[i] for row in test]
        #print(actual)
        predicted = [forecast[i] for forecast in forecasts]
        #print(predicted)
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i + 1), rmse))

dataframe = read_csv('bias.csv', usecols=[1], engine='python', skipfooter=0)
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
n_lag = look_back
n_seq = 6
n_train = train_size
n_test = test_size
n_batch = 1
n_neurons = 4

#print(train)
trainX, trainY = create_dataset(train, look_back,n_seq)

testX, testY = create_dataset(test, look_back,n_seq)
#print(trainX)
#print(trainY)

#scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
#print(train)
#print(test)
model = fit_lstm(trainX, trainY, n_batch, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, testX, testY)
#print(forecasts)
#for i in range(len(forecasts)):
 #print(forecasts[:][i])
 #if(i==len(forecasts)):
#print("-------------------------------------------")
actual = testY
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
