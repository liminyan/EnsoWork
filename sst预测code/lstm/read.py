import os
path = "C:/Users/Admin/Documents/WeChat Files/wuli101/Files/FIO_ESM_NINO/to_songroy/nino34/"  # 指定需要读取文件的目录
files = os.listdir(path)  # 采用listdir来读取所有文件
files.sort()  # 排序
s = []  # 创建一个空列表
for file_ in files:  # 循环读取每个文件名
    print(path +file_)
    if not os.path.isdir(path + file_):  # 判断该文件是否是一个文件夹
        f_name = str(file_)
        #        print(f_name)
        s.append(f_name)  # 把当前文件名返加到列表里
        f.write(f_name + '\n')  # 写入之前的文本中

#print(s)