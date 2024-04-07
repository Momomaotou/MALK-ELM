from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D,Activation,Dropout,Reshape,LSTM
from keras import callbacks as callb
from sklearn.ensemble import AdaBoostClassifier
from keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

from MKLpy.utils.misc import identity_kernel


import numpy as np
from MKLpy.algorithms import EasyMKL, AverageMKL, FHeuristic, PWMK
from sklearn.model_selection import KFold
from MKLpy.preprocessing import \
    normalization, rescale_01, rescale, centering
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from DataReading import nsl_data_read


#读入数据
Xtr,Ytr = nsl_data_read("data/KDDTrain+.txt", 0.05)
Xva,Yva = nsl_data_read("data/KDDTest+.txt", 0.1)

print(Xtr.shape)
print(Xva.shape)
#数据归一化，使用最大最小标准化方法
Xtr = MinMaxScaler().fit_transform(Xtr)
Xva = MinMaxScaler().fit_transform(Xva)
Xtr = pd.DataFrame(Xtr)
Xva = pd.DataFrame(Xva)

Xtr = Xtr.values.reshape(Xtr.shape[0], 1, Xtr.shape[1])
Xva = Xva.values.reshape(Xva.shape[0], 1, Xva.shape[1])
Ytr = to_categorical(Ytr)
Yva = to_categorical(Yva)
'''
#数据格式转化为矩阵
Xtr = np.array(Xtr)
Ytr = np.array(Ytr)
Xva = np.array(Xva)
Yva = np.array(Yva)
'''
#num_classes = len(np.unique(Ytr))

'''
data = load_iris()  #let's try with iris dataset!
X,Y = data.data, data.target
num_classes = len(np.unique(Y))

Yh = to_categorical(Y)  #I need one-hotted labels for training the NN
Xtr, Xva, Ytr, Yva, Ytr_1h, Yva_1h = train_test_split(X, Y,
        Yh, random_state=42, shuffle=True, test_size=.3)
print(type(Ytr))
'''

#parameters of the network
learning_rate = 1e-5
batch_size    = 32
activation    = 'sigmoid'
num_hidden    = 4   #num hidden layers
num_neurons   = 64  #num neurons per layer
max_epochs    = 100

#model setting

model = Sequential()
#model.add(Flatten())
'''
for l in range(1, num_hidden+1):    #add hidden layers
    layer = Dense(num_neurons, activation=activation, name='hidden_%d' % l)
    model.add(layer)
classification_layer = Dense(num_classes, activation='softmax', name='output')
model.add(classification_layer)
'''

model.add(Conv1D(64, 3, padding="same", activation="relu",input_shape=(Xtr.shape[1],Xtr.shape[2])))
model.add(MaxPooling1D(2, 2, padding='same'))
model.add(Flatten())
model.add(Dense(64,name="hidden_1", activation="relu"))
for l in range(2, num_hidden):
    model.add(Reshape((1, 64)))
    model.add(Conv1D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(2, 2, padding='same'))
    model.add(Flatten())
    model.add(Dense(64,name="hidden_%d" % l, activation="relu"))
model.add(Reshape((1, 64)))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, name="hidden_%d" %num_hidden, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

#optional equipments
reduce_lr  = callb.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
earlystop  = callb.EarlyStopping(
    monitor='val_loss', patience=10, mode='min',verbose=1)

#compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    Xtr, Ytr,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(Xva, Yva),
    verbose=1,
    callbacks=[reduce_lr, earlystop])

#representations extraction and kernels definition
def extract_reps(X, net):
    ''' this function extracts intermediate representations
        developed by network for each input example in X.
    '''
    representations = []
    for l in range(1, num_hidden+1):
        layer_name = 'hidden_%d' % l
        partial_model = Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output)
        #rep_l contains the representation developed at the
        #layer l for each input examples
        rep_l = partial_model.predict(X).astype(np.double)
        rep_l = normalization(rep_l)    #we always prefer to normalize data
        representations.append(rep_l)
    return representations

# here, XL is a list containing matrices
# the i-th matrix contains the representations for all input examples
# developed at the i-th hidden layer
XLtr = extract_reps(Xtr, model)
XLva = extract_reps(Xva, model)

# now, we can create our kernels list
# in this specific case, we compute linear kernels
KLtr = [X   @ X.T for X in XLtr]
KLva = [Xva @ X.T for Xva, X in zip(XLva, XLtr)]
# have you seen the section *best practices* ?
# I just add the base input rerpesentation and an identity matrix
#KLtr += [Xtr @ Xtr.T, identity_kernel(len(Ytr))]
#KLva += [Xva @ Xtr.T, np.zeros(KLva[0].shape)]
#反One-Hot编码
Ytr = torch.Tensor(Ytr)
Yva = torch.Tensor(Yva)
_,Ytr = torch.max(Ytr,1)
_,Yva = torch.max(Yva,1)
print(type(Ytr))
print(Ytr)
print(Ytr.shape)
print(KLtr)

# MKL part
#mkl = EasyMKL().fit(KLtr, Ytr)
#mkl = AverageMKL().fit(KLtr, Ytr)
#mkl = FHeuristic().fit(KLtr, Ytr)
#y_preds = mkl.predict(KLva)


#Boosting approach
#cv = KFold(n_splits=5, shuffle=True, random_state=42)
#mkl = PWMK(delta=0, cv=cv)
mkl = AverageMKL()
#mkl = FHeuristic()
mkl.fit(KLtr[0:2], Ytr)
y_preds = mkl.predict(KLva[0:2])
temp_accuracy1 = accuracy_score(Yva, y_preds)
print(temp_accuracy1)
j = 2
for i in range(2, num_hidden):
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

print(KLtr[0].shape)
# final evaluation
mkl.fit(KLtr, Ytr)
y_preds = mkl.predict(KLva)
accuracy = accuracy_score(Yva, y_preds)
#roc_auc = roc_auc_score(Yte, y_score)
print ('Accuracy score: %.4f' % accuracy)