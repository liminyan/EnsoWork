import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
import xlrd
###########1.数据生成部分##########
fileread = "C:/Users/Admin/Desktop/four_fea.xlsx"
end = 4
dataframe = pd.read_excel(fileread)
dataset = dataframe.values
print(dataset)

train_size = int(len(dataset) * 0.8)
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(train)
print(test)
x_train, y_train = train[:,0:end],train[:,end]
x_test, y_test = test[:,0:end],test[:,end]
y_test_most = test[:,0]
###########2.回归部分##########
def try_different_method(model,name):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)

    result = model.predict(x_test)
    train_result = model.predict(x_train)
    f = open('fitresult.txt','w')
    f.write(str(train_result))
    f.write(str(result))
    f.close()
    print(np.array(train_result))
    print(result)
    #print(y_test)
    rmse1 = math.sqrt(mean_squared_error(y_test, result))
    rmse2 = math.sqrt(mean_squared_error(y_test, y_test_most))
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)), y_test_most, 'yo-', label='x_most')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')

    plt.title('rmse1: %f,rmse2: %f,socre: %f' % (rmse1, rmse2, score))
    plt.legend()
    #plt.savefig("rmse"+name+".png")
    plt.show()

###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()


###########4.具体方法调用部分##########
try_different_method(model_DecisionTreeRegressor,"DecisionTreeRegressor")
try_different_method(model_LinearRegression,"LinearRegression")
try_different_method(model_SVR,"SVR")
try_different_method(model_KNeighborsRegressor,"KNeighborsRegressor")
try_different_method(model_RandomForestRegressor,"RandomForestRegressor")
try_different_method(model_AdaBoostRegressor,"AdaBoostRegressor")
try_different_method(model_GradientBoostingRegressor,"GradientBoostingRegressor")
try_different_method(model_BaggingRegressor,"BaggingRegressor")
try_different_method(model_ExtraTreeRegressor,"ExtraTreeRegressor")

