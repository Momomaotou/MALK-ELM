import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras import callbacks as callb
from keras.layers import Dense, Activation, Dropout, Conv1D, Flatten,MaxPooling1D
from sklearn.metrics import classification_report, confusion_matrix
import time
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from MKLpy.utils.misc import identity_kernel
from sklearn.model_selection import train_test_split
import numpy as np

from DataReading import nsl_data_read, NB15_data_read, New_data_read


Xtrain,Ytrain = nsl_data_read("data/KDDTrain+.txt",sampling_prop=1)
Xtest,Ytest = nsl_data_read("data/KDDTest+.txt",sampling_prop=1)
#Xtrain,Ytrain = NB15_data_read("data/UNSW-NB15/UNSW_NB15_testing-set.csv", 1)
#Xtest,Ytest = NB15_data_read("data/UNSW-NB15/UNSW_NB15_training-set.csv", 1)
#tempX,tempY = New_data_read('data\TempDataset.csv')
#Xtrain,Xtest,Ytrain,Ytest = train_test_split(tempX,tempY,test_size=0.4)

#数据归一化，使用最大最小标准化方法
Xtrain = MinMaxScaler().fit_transform(Xtrain)
Xtest = MinMaxScaler().fit_transform(Xtest)
Xtrain = pd.DataFrame(Xtrain)
Xtest = pd.DataFrame(Xtest)

#数据格式转化为矩阵
Xtrain = Xtrain.values.reshape(Xtrain.shape[0], 1, Xtrain.shape[1]).astype('float64')
Xtest = Xtest.values.reshape(Xtest.shape[0], 1, Xtest.shape[1]).astype('float64')
Ytrain = to_categorical(Ytrain)
Ytest = to_categorical(Ytest)
print(type(Xtrain))
print(type(Ytrain))
model = Sequential()
model.add(Conv1D(64, 3, padding="same", activation="relu",input_shape=(Xtrain.shape[1],Xtrain.shape[2])))
model.add(MaxPooling1D(2, 2, padding='same'))
model.add(Conv1D(128, 3, padding="same", activation="relu"))
model.add(MaxPooling1D(2, 2, padding='same'))
model.add(Flatten())
model.add(Dense(128,name="my_intermediate_layer", activation="relu"))
print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))
start = time.time()
model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
            )
model.fit(
            Xtrain,
            Ytrain,
            #validation_data=(Xtest, Ytest),
            epochs=10,
            batch_size=32,
            verbose=1
        )
print("--- %s seconds ---" % (time.time() - start))

loss, accuracy = model.evaluate(Xtest, Ytest, batch_size=32)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

temp = np.array(model.predict(Xtest))
y_pred = np.zeros_like(temp)
np.put_along_axis(y_pred, temp.argmax(1)[:, None], 1, axis=1)
y_pred = np.float64(y_pred)

print("\nAnomaly in Test: ", np.count_nonzero(Ytest, axis=0))
print("\nAnomaly in Prediction: ",np.count_nonzero(y_pred, axis=0))
print(confusion_matrix(Ytest.argmax(axis=1), y_pred.argmax(axis=1)))
