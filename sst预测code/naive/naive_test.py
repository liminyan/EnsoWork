# -*- coding: UTF-8 -*-

from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Lambda, Dropout
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import numpy

from SolveData import SolveData
from Data import Data
from Log import Log
from GetData import GetData

class Naive_net(object):
	"""docstring for Naive_net"""
	def __init__(self, data, log):
		super(Naive_net, self).__init__()
		self.data = data

		log.print('init-------'+str(data.name)+'--------begin')
		self.model = Sequential()
		log.print('init-------'+str(data.name)+'--------end')
	
	def bulid(self):

		# self.model.add(Lambda(lambda x: (x -self.data.mean_px) / self.data.std_px, input_shape= self.data.input_shape))
		# self.model.add(Dense(4, input_shape= self.data.input_shape,return_sequences=True))
		self.model.add(LSTM(4,input_shape=(1, self.data.pre_month)))
		# self.model.add(Dense(10, activation='relu'))
		# self.model.add(Dense(9, activation='relu'))
		# self.model.add(Dense(8, activation='relu'))
		# self.model.add(Dense(7, activation='relu'))
		# self.model.add(Dense(6, activation='relu'))
		# self.model.add(Dense(5, activation='relu'))
		# self.model.add(Dense(4, activation='relu'))
		# self.model.add(Dense(16, activation='relu'))
		# self.model.add(Dense(4, activation='relu'))
		self.model.add(Dense(1))
		self.model.compile(loss='mse', optimizer='sgd')
		# （BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）
		
	def train(self):
		
		# self.model.compile(
		# 	# loss=keras.losses.categorical_crossentropy,
		# 	loss = keras.losses.kullback_leibler_divergence,#交叉熵
		# 	optimizer = keras.optimizers.Adadelta(),
		# 	metrics = ['accuracy'])
		print(self.data.x_train.shape)
		self.data.x_train = numpy.reshape(self.data.x_train, (self.data.x_train.shape[0], 1, self.data.x_train.shape[1]))
		# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
		self.model.fit(
			self.data.x_train, self.data.y_train,
        	batch_size = 10,
        	epochs = self.data.epochs,
			verbose = 2)


	def test_self(self):

		predictions = self.model.predict_classes(self.data.x_test, verbose=0)

		return predictions

	def predict(self,test):

		predictions = self.model.predict_classes(test, verbose=0)

		return predictions

def main():
	
	myGetData = GetData()
	myGetData.load_file("nino3")
	myGetData.get_ans_xlsx("temp.xlsx")
	myLog = Log()
	Data_list = myGetData.dived_into_12_month(0.8,4)
	Net_list = []
	for x in range(0,len(Data_list)):

		temp_net = Naive_net(Data_list[x],myLog)
		temp_net.bulid()
		temp_net.train()

	# Net_list.append(temp_net)
	

if __name__ == '__main__':
	main()