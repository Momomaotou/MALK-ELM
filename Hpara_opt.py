
from DataReading import nsl_data_read, NB15_data_read, New_data_read
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel as hpk
from sklearn.metrics.pairwise import polynomial_kernel as poly
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from MKLpy.preprocessing import \
    normalization
import time
from keras.utils import to_categorical
from keras import callbacks as callb
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D,Activation,Dropout,Reshape,LSTM, Bidirectional, ConvLSTM1D
import os
from hyperoptim import GASearch, Hparams
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#NSL-KDD数据集
Xtr,Ytr = nsl_data_read("data/KDDTrain+.txt", sampling_prop=0.05)
Xte,Yte = nsl_data_read("data/KDDTest+.txt", sampling_prop=0.2)

#数据归一化，使用最大最小标准化方法
Xtr = MinMaxScaler().fit_transform(Xtr)
Xte = MinMaxScaler().fit_transform(Xte)
Xtr = pd.DataFrame(Xtr)
Xte = pd.DataFrame(Xte)

#数据格式转化为矩阵
Xtr = np.array(Xtr).astype(float)
Ytr = np.array(Ytr)
Xte = np.array(Xte).astype(float)
Yte = np.array(Yte)


class KELM:
    def __init__(self,kernel='rbf',C=1,gamma=0.1):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def kernel_func(self,X1,X2):
        if self.kernel == 'rbf':
            K = np.exp(-self.gamma * np.sum((X1[: np.newaxis :] - X2[np.newaxis : :]) ** 2,axis=-1))
        elif self.kernel == 'poly':
            K = (np.dot(X1,X2.T) + 1) ** self.gamma
        elif self.kernel == 'sigmoid':
            K = np.tanh(self.gamma * np.dot(X1,X2.T) + 1)
        return K

    def fit(self,X,y):
        self.X = X
        self.y = y
        self.H = self.kernel_func(X,X)
        self.beta = np.dot(np.linalg.inv(self.H + np.eye(X.shape[0]) / self.C),y)

    def predict(self,X):
        K = self.kernel_func(X,self.X)
        y_pred = np.dot(K,self.beta)
        return y_pred



KELMClassifer = KELM()

# 创建SVM模型对象
clf = SVC()

# 定义要搜索的超参数范围
param_grid = {'C': [6,7,8,9,10], 'gamma': [0.5, 1, 4, 16, 32]}

# 创建GridSearchCV对象并指定模型、超参数范围等信息
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='accuracy')



# 在数据上运行网格搜索
grid_search.fit(Xtr,Ytr)

# 输出最佳超参数组合及其得分
print("Best parameters found: ", grid_search.best_params_)
print("Score of the best combination: ", grid_search.best_score_)