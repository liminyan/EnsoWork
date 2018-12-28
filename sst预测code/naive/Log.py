# -*- coding: UTF-8 -*-

class Log(object):
	"""docstring for Log"""
	def __init__(self):
		super(Log, self).__init__()
		self.flag = 1;
	
	def set_flag_to_debug(self,flag):
		""" 
			flag = 1 print debug message
			flag = 0 not print debug message
		"""
		self.flag = flag

	def print(self,data):
		if self.flag == 1:
			print(data)

# myLog = Log()
# myLog.print("test")
# myLog.set_flag_to_debug(0)
# myLog.print("test")


		