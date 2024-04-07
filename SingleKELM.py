
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
from hyperoptim import GASearch

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, 'venv\Lib\site-packages')


#base_learner = SVC(C=100)


#读入数据

#NSL-KDD数据集
Xtr,Ytr = nsl_data_read("data/KDDTrain+.txt", sampling_prop=0.05)
Xte,Yte = nsl_data_read("data/KDDTest+.txt", sampling_prop=0.2)
#,Ytr = NB15_data_read("data/UNSW-NB15/UNSW_NB15_training-set.csv", 0.08)
#Xte,Yte = NB15_data_read("data/UNSW-NB15/UNSW_NB15_testing-set.csv", 0.03)
#tempX,tempY = New_data_read('data\TempDataset.csv')
#Xtr,Xte,Ytr,Yte = train_test_split(tempX,tempY,test_size=0.4)

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
#Ytr = to_categorical(Ytr)
#Yte = to_categorical(Yte)
def convert_to_matrix(labels):
    unique_labels = list(set(labels))  # 获取所有不重复的标签

    num_rows = len(labels)  # 行数等于样本数量
    num_cols = len(unique_labels)  # 列数等于标签种类数量

    matrix = np.zeros((num_rows, num_cols), dtype=int)  # 创建全零矩阵
    for i in range(0, num_rows):

        matrix[i][labels[i]] = 1  # 根据索引位置设置相应元素为1

    return matrix

class KELM:
    def __init__(self,C=1,gamma=0.1):
        self.C = C
        self.gamma = gamma


    def fit(self,X,K,y):
        self.X = X
        y = convert_to_matrix(y)
        self.y = np.array(y)
        self.H = K
        self.beta = np.dot(np.linalg.inv(self.H + np.eye(X.shape[0]) / self.C),y)
    def predict(self,K):
        y_pred = np.dot(K,self.beta)
        return y_pred

#num_classes = len(np.unique(Ytr))
#KLtr = generators.HPK_generator(Xtr, degrees=range(1,4))
#KLte = generators.HPK_generator(Xte, degrees=range(1,4))
#KLtr = [hpk(Xtr,     degree=d) for d in range(1,2)]
#KLte = [hpk(Xte,Xtr, degree=d) for d in range(1,2)]
#KLtr = [poly(Xtr,     degree=d) for d in range(1,3)]
#KLte = [poly(Xte,Xtr, degree=d) for d in range(1,3)]
#KLtr = [rbf(Xtr,     gamma=32)]
#KLte = [rbf(Xte,Xtr, gamma=32)]


#AverageMKL simply computes the average of input kernels
#It looks bad but it is a really strong baseline in MKL ;)
#parameters of the network
learning_rate = 1e-5
batch_size    = 32
activation    = 'sigmoid'
num_model    = 4   #num hidden layers
num_neurons   = 64  #num neurons per layer
max_epochs    = 100
#optional equipments
reduce_lr  = callb.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
earlystop  = callb.EarlyStopping(
    monitor='val_loss', patience=10, mode='min',verbose=1)

#调整数据格式和尺寸，使其适应神经网络模型
Xtr = pd.DataFrame(Xtr)
Xte = pd.DataFrame(Xte)
Xtr = Xtr.values.reshape(Xtr.shape[0], 1, Xtr.shape[1])
Xte = Xte.values.reshape(Xte.shape[0], 1, Xte.shape[1])
#One-Hot编码
Ytr = to_categorical(Ytr)
Yte = to_categorical(Yte)

#设置模型CNN1
model_cnn1 = Sequential()
model_cnn1.add(Conv1D(64, 3, padding="same", activation="relu", input_shape=(Xtr.shape[1],Xtr.shape[2])))
model_cnn1.add(MaxPooling1D(2, 2, padding='same'))
model_cnn1.add(Conv1D(128, 3, padding="same", activation="relu"))
model_cnn1.add(Flatten())
model_cnn1.add(Dense(128,name="hidden", activation="relu"))
model_cnn1.add(Dropout(0.5))
model_cnn1.add(Dense(5))
model_cnn1.add(Activation('softmax'))
model_cnn1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#设置模型LSTM1
model_lstm1 = Sequential()
model_lstm1.add(LSTM(64, activation="relu", input_shape=(Xtr.shape[1],Xtr.shape[2]), return_sequences=True))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(MaxPooling1D(2, 2, padding='same'))
model_lstm1.add(LSTM(128, activation="relu", return_sequences=True))
model_lstm1.add(Flatten())
model_lstm1.add(Dense(128,name="hidden", activation="relu"))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Dense(5))
model_lstm1.add(Activation('softmax'))
model_lstm1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#设置模型ConvLSTM1
model_ConvLSTM1 = Sequential()
model_ConvLSTM1.add(ConvLSTM1D(64, 3,padding="same", activation="relu",input_shape=(Xtr.shape[1],Xtr.shape[2],1), return_sequences=True))
model_ConvLSTM1.add(Dropout(0.5))
model_ConvLSTM1.add(ConvLSTM1D(128, 3, padding="same", activation="relu", return_sequences=True))
model_ConvLSTM1.add(Flatten())
model_ConvLSTM1.add(Dense(128,name="hidden", activation="relu"))
model_ConvLSTM1.add(Dropout(0.5))
model_ConvLSTM1.add(Dense(5))
model_ConvLSTM1.add(Activation('softmax'))
model_ConvLSTM1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#定义隐藏层表示的提取方法
def extract_reps(X, model):
    ''' this function extracts intermediate representations
            developed by network for each input example in X.
        '''
    #representations = []
    layer_name = 'hidden'
    partial_model = Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output)
    # rep_l contains the representation developed at the
    # layer l for each input examples
    rep_l = partial_model.predict(X).astype(np.double)
    rep_l = normalization(rep_l)  # we always prefer to normalize data
    #representations.append(rep_l)
    return rep_l

#生产深度映射核
XLtr = []
XLva = []
'''
#提取模型CNN1特征表达
model_cnn1.fit(
    Xtr, Ytr,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xte, Yte),
    verbose=1,
    callbacks=[reduce_lr, earlystop])
XLtr.append(extract_reps(Xtr, model_cnn1))
XLva.append(extract_reps(Xte, model_cnn1))
'''
#提取模型CNN1特征表达
model_ConvLSTM1.fit(
    Xtr, Ytr,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xte, Yte),
    verbose=1,
    callbacks=[reduce_lr, earlystop])
XLtr.append(extract_reps(Xtr, model_ConvLSTM1))
XLva.append(extract_reps(Xte, model_ConvLSTM1))

#根据特征表达产生对应的核矩阵
KLtr = []
KLte = []
for X in XLtr:
    KLtr.append(X   @ X.T)
for Xva, X in zip(XLva, XLtr):
    KLte.append(Xva @ X.T)
#反One-Hot编码
Ytr = torch.Tensor(Ytr)
Yte = torch.Tensor(Yte)
_,Ytr = torch.max(Ytr,1)
_,Yte = torch.max(Yte,1)


KELMClassifer = KELM()
KELMClassifer.fit(Xtr, KLtr[0], Ytr)       #train the classifier
y_pred = KELMClassifer.predict(KLte[0]).argmax(axis=1)       #predict the output class



accuracy = accuracy_score(Yte, y_pred)

print ('Accuracy score: %.4f' % accuracy)

print(confusion_matrix(Yte, y_pred))