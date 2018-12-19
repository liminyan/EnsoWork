import numpy as np
import os
path = "C:/Users/Admin/Documents/WeChat Files/wuli101/Files/FIO_ESM_NINO/to_songroy/nino34/"  # 指定需要读取文件的目录
files = os.listdir(path)  # 采用listdir来读取所有文件
files.sort()  # 排序

for file_ in files:  # 循环读取每个文件名

  filename = path + file_
  #print(filename)
  with open(filename, 'r') as file_to_read:
   sum = [0 for m in range(6)]
   while True:
    lines = file_to_read.readline() # 整行读取数据
    #print(lines)
    if not lines:
      break
      pass
    a = [float(k) for k in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
    for i in range(len(a)):
     sum[i] = sum[i] + a[i]
   sum = np.array(sum) / 10
   print(sum)
    #print(sum)
    #Efield.append(b_tmp)
    #pos = np.array(pos) # 将数据从list类型转换为array类型。
    #Efield = np.array(Efield)
#print(pos)
#print(Efield)
 #print(sum)