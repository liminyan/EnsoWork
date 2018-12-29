# -*- coding: UTF-8 -*-
import numpy as np


class Data(object):
	"""
	Data means get all EnSo Data
	provide get Data detial interface
	"""
	def __init__(self, name):
		super(Data, self).__init__()
		self.name = name

	def print_name(self):
		print(self.name)

	def set_pre_month(self,month):
		self.pre_month = month;

	def set_train(self,batch_size,epochs,x_train,y_train,x_test,x_ans):
		print(x_train.shape)
		mean_px = x_train.mean().astype(np.float32)
		std_px = x_train.std().astype(np.float32)
		self.mean_px = mean_px
		self.std_px = std_px
		self.input_shape = (self.pre_month,)
		self.x_train = x_train
		self.y_train = y_train
		self.batch_size = batch_size
		self.epochs = epochs
		self.x_test = x_test
		self.x_ans = x_ans



# myData = Data(1)
# mlist = myData.dived_into_12_month()
