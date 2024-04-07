from MKLpy.algorithms import AverageMKL,EasyMKL,PWMK,FHeuristic,CKA,GRAM,RMKL,MEMO
from MKLpy import generators
from MKLpy.scheduler  import ReduceOnWorsening
from MKLpy.callbacks  import EarlyStopping,Monitor
from DataReading import nsl_data_read, NB15_data_read, New_data_read
from FeatureRead import nsl_data_read_feature
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.datasets import load_iris
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel as hpk
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.metrics.pairwise import polynomial_kernel as poly
from sklearn.model_selection import KFold
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
from KELM import KELM
import numpy as np
import pandas as pd
import time
import torch
from sklearn.svm import SVC
import os
import sys

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, 'venv\Lib\site-packages')


#base_learner = SVC(C=100)
base_learner = KELM()

#读入数据

#NSL-KDD数据集
#Xtr,Ytr = nsl_data_read("data/KDDTrain+.txt", sampling_prop=0.04)
#Xte,Yte = nsl_data_read("data/KDDTest+.txt", sampling_prop=0.2)
Xtr_l,Ytr = nsl_data_read_feature("data/KDDTrain+.txt", sampling_prop=0.04)
Xte_l,Yte = nsl_data_read_feature("data/KDDTest+.txt", sampling_prop=0.2)
Xtr = Xtr_l[0]#全部特征
Xte = Xte_l[0]
#Xtr,Ytr = NB15_data_read("data/UNSW-NB15/UNSW_NB15_training-set.csv", 0.05)
#Xte,Yte = NB15_data_read("data/UNSW-NB15/UNSW_NB15_testing-set.csv", 0.02)
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


#num_classes = len(np.unique(Ytr))
'''
#鸢尾花数据集
data = load_iris()  #let's try with iris dataset!
X,Y = data.data, data.target
num_classes = len(np.unique(Y))
Yh = to_categorical(Y)  #I need one-hotted labels for training the NN
Xtr, Xte, Ytr, Yte, Ytr_1h, Yva_1h = train_test_split(X, Y,
        Yh, random_state=42, shuffle=True, test_size=.3)
print(Xtr.shape)
print(Xte.shape)
'''
#KLtr = generators.HPK_generator(Xtr, degrees=range(1,4))
#KLte = generators.HPK_generator(Xte, degrees=range(1,4))
KLtr = [hpk(Xtr,     degree=d) for d in range(1,3)]
KLte = [hpk(Xte,Xtr, degree=d) for d in range(1,3)]
#KLtr = [poly(Xtr,     degree=d) for d in range(1,3)]
#KLte = [poly(Xte,Xtr, degree=d) for d in range(1,3)]
#KLtr = [rbf(Xtr,     gamma=g) for g in (0.1,0.01,0.001)]
#KLte = [rbf(Xte,Xtr, gamma=g) for g in (0.1,0.01,0.001)]



#mkl = AverageMKL(learner=base_learner)
#mkl = EasyMKL(lam=.1)
#cv = KFold(n_splits=5, shuffle=True, random_state=42)
#mkl = PWMK(delta=0, cv=cv)
mkl = FHeuristic()
#mkl = CKA()
'''
earlystop = EarlyStopping(
    KLte, Yte,      #validation data, KL is a validation kernels list
    patience=5,     #max number of acceptable negative steps
    cooldown=1,     #how ofter we run a measurement, 1 means every optimization step
    metric='roc_auc',   #the metric we monitor
)
#scheduler = ReduceOnWorsening()
'''
'''
#Boosting approach
iter = len(KLtr)
mkl = FHeuristic()
mkl.fit(KLtr[0:2], Ytr)
y_preds = mkl.predict(KLte[0:2])
temp_accuracy1 = accuracy_score(Yte, y_preds)
print(temp_accuracy1)
j = 2
for i in range(2, iter):
    mkl.fit(KLtr[0:i+1], Ytr)
    y_preds = mkl.predict(KLte[0:i+1])
    temp_accuracy2 = accuracy_score(Yte, y_preds)
    print(temp_accuracy2)
    if(temp_accuracy1 >= temp_accuracy2):
        KLtr.pop(j)
        KLte.pop(j)
        print('Drop the %d th kernel' %(i+1))
    else:
        temp_accuracy1 = temp_accuracy2
        j=j+1
        print('Add the %d th kernel' %(i+1))
'''

#AverageMKL simply computes the average of input kernels
#It looks bad but it is a really strong baseline in MKL ;)


start_time = time.time()
mkl.fit(KLtr, Ytr)       #train the classifier
end_time = time.time()
y_pred  = mkl.predict(KLte)            #predict the output class
y_score = mkl.decision_function(KLte)  #returns the projection on the distance vector

accuracy = accuracy_score(Yte, y_pred)

print ('Accuracy score: %.4f' % accuracy)
print ('training time: %.3f' % (end_time-start_time))

print(confusion_matrix(Yte, y_pred))