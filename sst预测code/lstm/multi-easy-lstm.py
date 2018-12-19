import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# load the dataset
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
# use this function to prepare the train and test datasets for modeling
look_back = 5
#print(train)
trainX, trainY = create_dataset(train, look_back,5)

testX, testY = create_dataset(test, look_back,5)
print(trainX)
print(trainY)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
#print(trainX.shape)
#print(trainX)
#print(trainY)
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(5))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
#print(trainPredict)
testPredict = model.predict(testX)
#print(trainY)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
#print(testPredict.shape)
# shift train predictions for plotting
trainPredictPlot = [[0 for col in range(5)] for row in range(252)]
#print(numpy.array(trainPredictPlot).shape)
trainPredictPlot = numpy.array(trainPredictPlot)
trainPredictPlot[:, :] = 0
#print(trainPredictPlot)
#print(len(trainPredict)+look_back)
#print(trainPredict.shape)
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
trainPredictPlot[5:195] = trainPredict
testPredictPlot = [[0 for col in range(5)] for row in range(252)]
#print(numpy.array(trainPredictPlot).shape)
testPredictPlot = numpy.array(testPredictPlot)
testPredictPlot[:, :] = 0
#(testPredictPlot)
#print(len(trainPredict)+look_back)
#print(trainX.shape)
#print(testPredictPlot.shape)
testPredictPlot[200:240, :] = testPredict
# shift test predictions for plotting
#testPredictPlot = numpy.empty_like(dataset)
#testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
t = 5
X = []
for x in range(250):
    t += 1
    X.append(t)
plt.plot(X[0:190],trainPredictPlot[5:195,0])
#print(trainPredictPlot)
plt.plot(X[200:240],testPredictPlot[200:240,0])
#print(testPredictPlot)
plt.show()

