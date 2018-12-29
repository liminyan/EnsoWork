import os
import numpy as np
import re
from collections import Counter
import json
import matplotlib.pyplot as plt
import xlrd
from openpyxl import load_workbook
from scipy.fftpack import fft,ifft
from Data import Data

def get_number(filename):
	pattern = re.compile(r"([0-9]*)")
	word = pattern.findall(filename)
	word = [item for item in word if len(item) > 0]
	return int(word[0])

def file_name(file_dir):   
	L = []
	O = []
	for root, dirs, files in os.walk(file_dir):  
		for file in files:  
			if os.path.splitext(file)[1] == '.txt':  
				L.append(os.path.join(root, file))
			else:
				O.append(os.path.join(root, file))
	L.sort()
	return L,O

class GetData(object):
	"""docstring for GetData"""
	def __init__(self):
		super(GetData, self).__init__()
		self.data = {}
		self.number = 15;
		self.per_data = 6;
		self.year = 0;
		self.data_value = []
		self.ans_value = []
		self.data_key = []
		self.ans_key = []

	def load_file(self,name):
		self.file_list,self.anser = file_name(name)
		self.number = len(self.file_list)
		for i in range(0,self.number):
			print(self.file_list[i])
			f=open(self.file_list[i])
			self.data[get_number(self.file_list[i])] = [];
			self.data_value.append([])
			self.data_key.append([]);
			for line in f:
				a = [float(x) for x in line.split()]
				self.data[get_number(self.file_list[i])].append(a)
				self.data_value[i].append(a);
			self.data_key[i].append(get_number(self.file_list[i]))
		

	def get_ans_xlsx(self,name):
		workbook = load_workbook(name)
		sheets = workbook.get_sheet_names()        
		booksheet = workbook.get_sheet_by_name(sheets[0])
		rows = booksheet.rows
		columns = booksheet.columns
		self.ans = {}
		for row in rows:
			line = [col.value for col in row]
			if (str(line[0]).isdigit()):
				self.ans[int(line[0])] = line[1:]
				self.ans_key.append(int(line[0]))
				self.ans_value.append(line[1:]);
				self.year += 1
	

	def get_all_data(self):
		pass

	def dived_into_12_month(self,train_rate,train_month):
		"""
		train_rate : the rate: train size / total data size
		
		train_month : input: the number of month   
		"""
		data_test_number = 12;
		data_test_number = self.number;
		data_size = self.per_data;
		# now just choose member = 0
		# TODO weight net for diff member
		
		x_train_list = []
		# month   year 
		y_train_list = []
		# month   year 
		real_data_size = []
		

		for x in range(0,12):
			x_train_list.append([]);
			y_train_list.append([]);
			real_data_size.append(0);


		# member = 0;
		for member in range(0,10):

			for item in range(0,data_test_number):

				year = int(item/12);
				month = int(item%12);
				b = self.ans_value[year][month]
				flag_Data_Valid = 0;
				#valid flag
				temp_x_list= []
				for x in range(0,min(1+item,data_size)):
					if self.data_value[item-x][member][0+x]>-99:
						# 排除空白数据 
						a = self.data_value[item-x][member][0+x]
						temp_x_list.append(a)
						flag_Data_Valid = 1;

				if (flag_Data_Valid == 1) and (len(temp_x_list )>= 6):
					print(year,month)
					real_data_size[month] += 1
					y_train_list[month].append(b)
					x_train_list[month].append(temp_x_list[0:train_month])

		Data_list = []

		print(x_train_list[0])
		print(y_train_list[0])
		print(np.array(x_train_list[0]).shape)
		print(np.array(y_train_list[0]).shape)
		for x in range(0,12):
			test_number = int(train_rate*real_data_size[x])
			print(x,' test: ',test_number)
			temp_data = Data(x+1)
			temp_data.set_pre_month(train_month)
			temp_data.set_train(4,200,
								np.array(x_train_list[x][1:test_number]), 
								np.array(y_train_list[x][1:test_number]),
								np.array(x_train_list[x][test_number: ]),
								np.array(y_train_list[x][test_number: ])
								)
			Data_list.append(temp_data)

		return Data_list
		
		


# myGetData = GetData()
# myGetData.load_file("nino3")
# myGetData.get_ans_xlsx("temp.xlsx")
# a = myGetData.dived_into_12_month(0.5,4)
# print(a)