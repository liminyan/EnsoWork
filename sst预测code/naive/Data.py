# -*- coding: UTF-8 -*-


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

	def dived_into_12_month(self):
		List = []
		for x in range(0,12):
			List.append(Data(x));
		return List

# myData = Data(1)
# mlist = myData.dived_into_12_month()
