from MKLpy.preprocessing import \
    normalization, rescale_01, rescale, centering
from DataReading import nsl_data_read, NB15_data_read, New_data_read
from sklearn.model_selection import train_test_split
from FeatureRead import nsl_data_read_feature, NB15_data_read_feature, New_data_read_feature
from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel as hpk
from sklearn.metrics.pairwise import polynomial_kernel as poly
from sklearn.metrics.pairwise import rbf_kernel as rbf
from keras import callbacks as callb
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D,Activation,Dropout,Reshape,LSTM, Bidirectional, ConvLSTM1D
from MKLpy.algorithms import EasyMKL, AverageMKL, FHeuristic, PWMK
from keras.utils import to_categorical
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
from KELM import KELM
import torch
import time
import numpy as np
import pandas as pd


base_learner = KELM()

#读入数据


#NSL-KDD数据集
#Xtr,Ytr = nsl_data_read("data/KDDTrain+.txt", 0.04)
#Xva,Yva = nsl_data_read("data/KDDTest+.txt", 0.2)
#读取NSL-KDD
Xtr_l,Ytr_ts = nsl_data_read_feature("data/KDDTrain+.txt", sampling_prop=0.04)
Xva_l,Yva_ts = nsl_data_read_feature("data/KDDTest+.txt", sampling_prop=0.2)

Xtr = Xtr_l[0]#全部特征
Xva = Xva_l[0]
Xtr_t = Xtr_l[1] #特征组1
Xva_t = Xva_l[1]
Xtr_s = Xtr_l[2] #特征组2
Xva_s = Xva_l[2]
Xtr_o = Xtr_l[3] #特征组3
Xva_o = Xva_l[3]
#Xtr_t = Xtr_l
#Xva_t = Xva_l
#Xtr_s = Xtr_l
#Xva_s = Xva_l

'''
#NB15数据集
#读取NSL-KDD的时间特征和空间特征
Xtr_l,Ytr_ts = NB15_data_read_feature("data/UNSW-NB15/UNSW_NB15_training-set.csv", sampling_prop=0.05)
Xva_l,Yva_ts = NB15_data_read_feature("data/UNSW-NB15/UNSW_NB15_testing-set.csv", sampling_prop=0.02)
Xtr = Xtr_l[0]
Xva = Xva_l[0]
Ytr = Ytr_ts
Yva = Yva_ts
Xtr_t = Xtr_l[1] #时间特征为列表第2项
Xva_t = Xva_l[1]
Xtr_s = Xtr_l[2] #空间特征为列表第1项
Xva_s = Xva_l[2]
Xtr_o = Xtr_l[3] #空间特征为列表第1项
Xva_o = Xva_l[3]
#Xtr_t = Xtr_l
#Xva_t = Xva_l
#Xtr_s = Xtr_l
#Xva_s = Xva_l
#Xtr_o = Xtr_l
#Xva_o = Xva_l
'''
'''
#新数据集
tempX,tempY = New_data_read('data\TempDataset.csv')
Xtr_l,Xva_l,Ytr,Yva = train_test_split(tempX,tempY,test_size=0.4)


Xtr = Xtr_l[0]#全部特征
Xva = Xva_l[0]
Xtr_t = Xtr_l[1] #特征组1
Xva_t = Xva_l[1]
Xtr_s = Xtr_l[2] #特征组2
Xva_s = Xva_l[2]
Xtr_o = Xtr_l[3] #特征组3
Xva_o = Xva_l[3]
Ytr_ts = Ytr
Yva_ts = Yva
'''

#数据归一化，使用最大最小标准化方法
Xtr = MinMaxScaler().fit_transform(Xtr)
Xva = MinMaxScaler().fit_transform(Xva)
Xtr = pd.DataFrame(Xtr)
Xva = pd.DataFrame(Xva)
#数据格式转化为矩阵
Xtr = np.array(Xtr).astype(float)
Ytr = np.array(Ytr_ts)
Xva = np.array(Xva).astype(float)
Yva = np.array(Yva_ts)
#生成全局核
KLtr = [hpk(Xtr,     degree=d) for d in range(1,3)]
KLva = [hpk(Xva,Xtr, degree=d) for d in range(1,3)]
#KLtr = [rbf(Xtr,     gamma=g) for g in (32,64)]
#KLva = [rbf(Xva,Xtr,     gamma=g) for g in (32,64)]
#数据归一化，使用最大最小标准化方法
Xtr_t = MinMaxScaler().fit_transform(Xtr_t)
Xtr_s = MinMaxScaler().fit_transform(Xtr_s)
Xtr_o = MinMaxScaler().fit_transform(Xtr_o)
Xva_t = MinMaxScaler().fit_transform(Xva_t)
Xva_s = MinMaxScaler().fit_transform(Xva_s)
Xva_o = MinMaxScaler().fit_transform(Xva_o)
#生成多项式局部核

KLtr_1 = hpk(Xtr_s,     degree=1)
KLva_1 = hpk(Xva_s,Xtr_s, degree=1)
KLtr_2 = hpk(Xtr_t,     degree=1)
KLva_2 = hpk(Xva_t,Xtr_t, degree=1)
KLtr_3 = hpk(Xtr_o,     degree=1)
KLva_3 = hpk(Xva_o,Xtr_o, degree=1)



#调整数据格式和尺寸，使其适应神经网络模型
Xtr_t = pd.DataFrame(Xtr_t)
Xtr_s = pd.DataFrame(Xtr_s)
Xva_t = pd.DataFrame(Xva_t)
Xva_s = pd.DataFrame(Xva_s)
Xtr_o = pd.DataFrame(Xtr_o)
Xva_o = pd.DataFrame(Xva_o)
Xtr_t = Xtr_t.values.reshape(Xtr_t.shape[0], 1, Xtr_t.shape[1])
Xva_t = Xva_t.values.reshape(Xva_t.shape[0], 1, Xva_t.shape[1])
Xtr_s = Xtr_s.values.reshape(Xtr_s.shape[0], 1, Xtr_s.shape[1])
Xva_s = Xva_s.values.reshape(Xva_s.shape[0], 1, Xva_s.shape[1])
Xtr_o = Xtr_o.values.reshape(Xtr_o.shape[0], 1, Xtr_o.shape[1])
Xva_o = Xva_o.values.reshape(Xva_o.shape[0], 1, Xva_o.shape[1])
#One-Hot编码
Ytr = to_categorical(Ytr)
Yva = to_categorical(Yva)
Ytr_ts = to_categorical(Ytr_ts)
Yva_ts = to_categorical(Yva_ts)

#parameters of the network
learning_rate = 1e-5
batch_size    = 32
activation    = 'sigmoid'
num_model    = 4   #num hidden layers
num_neurons   = 64  #num neurons per layer
max_epochs    = 100

#model setting
#optional equipments
reduce_lr  = callb.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
earlystop  = callb.EarlyStopping(
    monitor='val_loss', patience=10, mode='min',verbose=1)
#设置模型CNN1
model_cnn1 = Sequential()
model_cnn1.add(Conv1D(32, 3, padding="same", activation="relu", input_shape=(Xtr_s.shape[1],Xtr_s.shape[2])))
model_cnn1.add(MaxPooling1D(2, 2, padding='same'))
model_cnn1.add(Conv1D(64, 3, padding="same", activation="relu"))
model_cnn1.add(Flatten())
model_cnn1.add(Dense(64,name="hidden", activation="relu"))
model_cnn1.add(Dropout(0.5))
model_cnn1.add(Dense(5))
model_cnn1.add(Activation('softmax'))
model_cnn1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#设置模型CNN2
model_cnn2 = Sequential()
model_cnn2.add(Conv1D(32, 3, padding="same", activation="relu", input_shape=(Xtr_t.shape[1],Xtr_t.shape[2])))
model_cnn2.add(MaxPooling1D(2, 2, padding='same'))
model_cnn2.add(Conv1D(64, 3, padding="same", activation="relu"))
model_cnn2.add(Flatten())
model_cnn2.add(Dense(64,name="hidden", activation="relu"))
model_cnn2.add(Dropout(0.5))
model_cnn2.add(Dense(5))
model_cnn2.add(Activation('softmax'))
model_cnn2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#设置模型CNN3
model_cnn3 = Sequential()
model_cnn3.add(Conv1D(32, 3, padding="same", activation="relu", input_shape=(Xtr_o.shape[1],Xtr_o.shape[2])))
model_cnn3.add(MaxPooling1D(2, 2, padding='same'))
model_cnn3.add(Conv1D(64, 3, padding="same", activation="relu"))
model_cnn3.add(Flatten())
model_cnn3.add(Dense(64,name="hidden", activation="relu"))
model_cnn3.add(Dropout(0.5))
model_cnn3.add(Dense(5))
model_cnn3.add(Activation('softmax'))
model_cnn3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#设置模型LSTM1
model_lstm1 = Sequential()
model_lstm1.add(LSTM(32, activation="relu", input_shape=(Xtr_s.shape[1],Xtr_s.shape[2]), return_sequences=True))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(MaxPooling1D(2, 2, padding='same'))
model_lstm1.add(LSTM(64, activation="relu", return_sequences=True))
model_lstm1.add(Flatten())
model_lstm1.add(Dense(64,name="hidden", activation="relu"))
model_lstm1.add(Dropout(0.5))
model_lstm1.add(Dense(5))
model_lstm1.add(Activation('softmax'))
model_lstm1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#设置模型LSTM2
model_lstm2 = Sequential()
model_lstm2.add(LSTM(32, activation="relu", input_shape=(Xtr_t.shape[1],Xtr_t.shape[2]), return_sequences=True))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(MaxPooling1D(2, 2, padding='same'))
model_lstm2.add(LSTM(64, activation="relu", return_sequences=True))
model_lstm2.add(Flatten())
model_lstm2.add(Dense(64,name="hidden", activation="relu"))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(Dense(5))
model_lstm2.add(Activation('softmax'))
model_lstm2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#设置模型LSTM3
model_lstm3 = Sequential()
model_lstm3.add(LSTM(32, activation="relu", input_shape=(Xtr_o.shape[1],Xtr_o.shape[2]), return_sequences=True))
model_lstm3.add(Dropout(0.5))
model_lstm3.add(MaxPooling1D(2, 2, padding='same'))
model_lstm3.add(LSTM(64, activation="relu", return_sequences=True))
model_lstm3.add(Flatten())
model_lstm3.add(Dense(64,name="hidden", activation="relu"))
model_lstm3.add(Dropout(0.5))
model_lstm3.add(Dense(5))
model_lstm3.add(Activation('softmax'))
model_lstm3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

#生产时间特征核与空间特征核
XLtr = []
XLva = []

#第一组特征
#提取模型CNN1特征表达
model_cnn1.fit(
    Xtr_s, Ytr_ts,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xva_s, Yva_ts),
    verbose=1,
    callbacks=[reduce_lr, earlystop])
XLtr.append(extract_reps(Xtr_s, model_cnn1))
XLva.append(extract_reps(Xva_s, model_cnn1))
#提取模型LSTM1特征表达
model_lstm1.fit(
    Xtr_s, Ytr_ts,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xva_s, Yva_ts),
    verbose=1,
    callbacks=[reduce_lr, earlystop]
)
XLtr.append(extract_reps(Xtr_s, model_lstm1))
XLva.append(extract_reps(Xva_s, model_lstm1))


#第二组特征
#提取模型CNN2特征表达
model_cnn2.fit(
    Xtr_t, Ytr_ts,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xva_t, Yva_ts),
    verbose=1,
    callbacks=[reduce_lr, earlystop])
XLtr.append(extract_reps(Xtr_t, model_cnn2))
XLva.append(extract_reps(Xva_t, model_cnn2))
#提取模型LSTM2特征表达
model_lstm2.fit(
    Xtr_t, Ytr_ts,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xva_t, Yva_ts),
    verbose=1,
    callbacks=[reduce_lr, earlystop]
)
XLtr.append(extract_reps(Xtr_t, model_lstm2))
XLva.append(extract_reps(Xva_t, model_lstm2))


#第三组特征
#提取模型CNN3特征表达
model_cnn3.fit(
    Xtr_o, Ytr_ts,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xva_o, Yva_ts),
    verbose=1,
    callbacks=[reduce_lr, earlystop])
XLtr.append(extract_reps(Xtr_o, model_cnn3))
XLva.append(extract_reps(Xva_o, model_cnn3))
#提取模型LSTM3特征表达
model_lstm3.fit(
    Xtr_o, Ytr_ts,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xva_o, Yva_ts),
    verbose=1,
    callbacks=[reduce_lr, earlystop]
)
XLtr.append(extract_reps(Xtr_o, model_lstm3))
XLva.append(extract_reps(Xva_o, model_lstm3))


#根据特征表达产生对应的核矩阵
for X in XLtr:
    KLtr.append(X   @ X.T)
for Xva, X in zip(XLva, XLtr):
    KLva.append(Xva @ X.T)

KLtr.append(KLtr_1)
KLtr.append(KLtr_2)
KLtr.append(KLtr_3)
KLva.append(KLva_1)
KLva.append(KLva_2)
KLva.append(KLva_3)

#反One-Hot编码
Ytr = torch.Tensor(Ytr)
Yva = torch.Tensor(Yva)
_,Ytr = torch.max(Ytr,1)
_,Yva = torch.max(Yva,1)

#Boosting approach
iter = len(KLtr)
mkl = FHeuristic()
mkl.fit(KLtr[0:2], Ytr)
y_pred = mkl.predict(KLva[0:2])
temp_accuracy1 = accuracy_score(Yva, y_pred)
print((temp_accuracy1))
j = 2
for i in range(2, iter):
    mkl.fit(KLtr[0:i+1], Ytr)
    temp_y_pred = mkl.predict(KLva[0:i+1])
    temp_accuracy2 = accuracy_score(Yva, temp_y_pred)
    print((temp_accuracy2))
    if(temp_accuracy1 >= temp_accuracy2):
        KLtr.pop(j)
        KLva.pop(j)
        print('Drop the %d th kernel' %i)
    else:
        temp_accuracy1 = temp_accuracy2
        j=j+1
        y_pred = temp_y_pred
        print('Add the %d th kernel' %i)



'''
start_time = time.time()
mkl = FHeuristic()
#mkl = AverageMKL()
mkl.fit(KLtr, Ytr)       #train the classifier
end_time = time.time()
y_pred  = mkl.predict(KLva)            #predict the output class
y_score = mkl.decision_function(KLva)  #returns the projection on the distance vector
'''
accuracy = accuracy_score(Yva, y_pred)
#roc_auc = roc_auc_score(Yte, y_score)
print ('Accuracy score: %.4f' % accuracy)
#print ('training time: %.2f' % (end_time-start_time))
print(confusion_matrix(Yva, y_pred))
