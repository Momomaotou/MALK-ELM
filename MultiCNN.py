from MKLpy.preprocessing import \
    normalization, rescale_01, rescale, centering
from DataReading import nsl_data_read, cicids_data_read
from FeatureRead import nsl_data_read_feature
from keras import callbacks as callb
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D,Activation,Dropout,Reshape,LSTM
from MKLpy.algorithms import EasyMKL, AverageMKL, FHeuristic, PWMK
from keras.utils import to_categorical
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd

from sklearn.svm import SVC

#base_learner = SVC(C=100)

#读入数据

#NSL-KDD数据集
Xtr,Ytr = nsl_data_read_feature("data/KDDTrain+.txt", sampling_prop=0.05)
Xva,Yva = nsl_data_read_feature("data/KDDTest+.txt", sampling_prop=0.1)

#数据归一化，使用最大最小标准化方法
for i in range(0, len(Xtr)):
    Xtr[i] = MinMaxScaler().fit_transform(Xtr[i])
    Xtr[i] = pd.DataFrame(Xtr[i])
    Xva[i] = MinMaxScaler().fit_transform(Xva[i])
    Xva[i] = pd.DataFrame(Xva[i])
    Xtr[i] = Xtr[i].values.reshape(Xtr[i].shape[0], 1, Xtr[i].shape[1])
    Xva[i] = Xva[i].values.reshape(Xva[i].shape[0], 1, Xva[i].shape[1])
Ytr = to_categorical(Ytr)
Yva = to_categorical(Yva)

'''
mkl = FHeuristic()

print(type(KLtr))
print(type(Ytr))

mkl.fit(KLtr, Ytr)       #train the classifier
y_pred  = mkl.predict(KLte)            #predict the output class
y_score = mkl.decision_function(KLte)  #returns the projection on the distance vector

accuracy = accuracy_score(Yte, y_pred)
#roc_auc = roc_auc_score(Yte, y_score)
print ('Accuracy score: %.4f' % accuracy)
'''
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

model = []
for i in range(0, num_model):
    model.append(Sequential())
    model[i].add(Conv1D(32, 3, padding="same", activation="relu",input_shape=(Xtr[i].shape[1],Xtr[i].shape[2])))
    model[i].add(MaxPooling1D(2, 2, padding='same'))
    model[i].add(Conv1D(64, 3, padding="same", activation="relu"))
    model[i].add(Flatten())
    model[i].add(Dense(64,name="hidden", activation="relu"))
    model[i].add(Dropout(0.5))
    model[i].add(Dense(5))
    model[i].add(Activation('softmax'))
    model[i].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



final_model = Sequential()

final_model.add(Dense(256,activation="relu"))
final_model.add(Dropout(0.5))
final_model.add(Dense(5))
final_model.add(Activation('softmax'))
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

#数据分类训练并融合
#训练过程
for i in range(0, num_model):
    model[i].fit(
        Xtr[i], Ytr,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_data=(Xva[i], Yva),
        verbose=1,
        callbacks=[reduce_lr, earlystop])
'''
Xtr1 = extract_reps(Xtr[0],model[0])
Xtr2 = extract_reps(Xtr[1],model[1])
Xtr3 = extract_reps(Xtr[2],model[2])
Xtr4 = extract_reps(Xtr[3],model[3])

Fin_Xtr = torch.cat([Xtr1,Xtr2],dim=1)
Fin_Xtr = torch.cat([Fin_Xtr,Xtr3],dim=1)
Fin_Xtr = torch.cat([Fin_Xtr,Xtr4],dim=1)

final_model.fit(
        Fin_Xtr, Ytr,
        batch_size=batch_size,
        epochs=max_epochs,
        #validation_data=(, Yva),
        verbose=1,
        callbacks=[reduce_lr, earlystop])
'''
XLtr = []
XLva = []
for i in range(0, num_model):
    XLtr.append(extract_reps(Xtr[i],model[i]))
    XLva.append(extract_reps(Xva[i], model[i]))

KLtr = [X   @ X.T for X in XLtr]
KLva = [Xva @ X.T for Xva, X in zip(XLva, XLtr)]

Ytr = torch.Tensor(Ytr)
Yva = torch.Tensor(Yva)
_,Ytr = torch.max(Ytr,1)
_,Yva = torch.max(Yva,1)
print(type(Ytr))
print(Ytr)
print(Ytr.shape)
print(KLtr)

mkl = AverageMKL()
#mkl = FHeuristic()

mkl.fit(KLtr[0:2], Ytr)
y_preds = mkl.predict(KLva[0:2])
temp_accuracy1 = accuracy_score(Yva, y_preds)
print(temp_accuracy1)
j = 2
for i in range(2, num_model):
    mkl.fit(KLtr[0:i+1], Ytr)
    y_preds = mkl.predict(KLva[0:i+1])
    temp_accuracy2 = accuracy_score(Yva, y_preds)
    print(temp_accuracy2)
    if(temp_accuracy1 >= temp_accuracy2):
        KLtr.pop(j)
        KLva.pop(j)
        print('Drop the %d th kernel' %(i+1))
    else:
        temp_accuracy1 = temp_accuracy2
        j=j+1
        print('Add the %d th kernel' %(i+1))

# final evaluation
mkl.fit(KLtr, Ytr)
y_preds = mkl.predict(KLva)
accuracy = accuracy_score(Yva, y_preds)
#roc_auc = roc_auc_score(Yte, y_score)
print ('Accuracy score: %.4f' % accuracy)

