import numpy as np
import os
path = "C:/Users/Admin/Documents/WeChat Files/wuli101/Files/FIO_ESM_NINO/to_songroy/nino3/"  # 指定需要读取文件的目录
files = os.listdir(path)  # 采用listdir来读取所有文件
files.sort(reverse=True)  # 排序
s = []  # 创建一个空列表
a = [0,0,0,0,0,0]
i = 1

for file_ in files[0:247]:  # 循环读取每个文件名
 files_c = files[i:i+4]
 #print(files_c)
 i = i + 1

 j =1
 for file1 in files_c:
  filename = path + file1
  #print(filename)
  with open(filename, 'r') as file_to_read:
   sum = 0
   while True:
    lines = file_to_read.readline() # 整行读取数据
    #print(lines)
    if not lines:
      break
      pass
    a = [float(k) for k in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
    sum = sum + a[j]
   sum = sum / 10
   j = j + 1
   print("%.3f" % sum, end=' ')
 print("\n")
    #print(sum)
    #Efield.append(b_tmp)
    #pos = np.array(pos) # 将数据从list类型转换为array类型。
    #Efield = np.array(Efield)
#print(pos)
#print(Efield)
 #print(sum)