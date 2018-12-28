# -*- coding: UTF-8 -*-

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

from SolveData import SolveData
from Data import Data
from Log import Log


class Naive_net(object):
	"""docstring for Naive_net"""
	def __init__(self, data,log):
		super(Naive_net, self).__init__()
		self.data = data

		log.print('init-------'+data.name+'--------begin')
		self.model = Sequential()
		log.print('init-------'+data.name+'--------end')
	
	def bulid(self):

		self.model.add(Lambda(lambda x: (x -self.data.mean_px) / self.data.std_px, input_shape= self.data.input_shape))

	def train(self):
		
		self.model.compile(
              # loss=keras.losses.categorical_crossentropy,   
              loss = keras.losses.kullback_leibler_divergence,#交叉熵
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

		self.model.fit(self.data.x_train, self.data.y_train,
          batch_size=self.data.batch_size,
          epochs=self.data.epochs,
          verbose=1)


	def pred(self):
		predictions = self.model.predict_classes(data.x_test, verbose=0)


def main():
	
	myData = Data("all data")
	myLog = Log()
	myLog.print ('--------tets--------')
	myNaive_net_list = []
	DataList = myData.dived_into_12_month()
	for x in DataList:
		print(x.name);
		myNaive_net = Naive_net(myData,myLog)
		myNaive_net_list.append(myNaive_net)
		
	
if __name__ == '__main__':
	main()