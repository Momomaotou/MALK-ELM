import numpy as np
import scipy.stats
from scipy.stats import pearsonr
from scipy.stats import entropy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.metrics import mutual_info_score as MIS
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from RobustmRMR import mRMR
def nsl_data_read_feature_2(filepath):
    data = pd.read_csv(filepath,header=None)
    #将第2、3、4列的字符型特征编码为数字特征
    encoder= LabelEncoder().fit(data[1])
    data1 = encoder.transform(data[1])
    data1 = pd.DataFrame(data1)
    data[1] = data1
    encoder= LabelEncoder().fit(data[2])
    data2 = encoder.transform(data[2])
    data2 = pd.DataFrame(data2)
    data[2] = data2
    encoder= LabelEncoder().fit(data[3])
    data3 = encoder.transform(data[3])
    data3 = pd.DataFrame(data3)
    data[3] = data3
    #整合攻击类型
    labels = data[41]
    labels.replace(['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop','mailbomb', 'apache2', 'processtable', 'udpstorm'], 'DoS', inplace=True)
    labels.replace(['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint'], 'Probe', inplace=True)
    labels.replace(['ftp_write',
            'guess_passwd',
            'imap',
            'multihop',
            'phf',
            'spy',
            'warezclient',
            'warezmaster',
            'sendmail',
            'named',
            'snmpgetattack',
            'snmpguess',
            'xlock',
            'xsnoop',
            'worm', ], 'R2L', inplace=True)
    labels.replace(['buffer_overflow',
            'loadmodule',
            'perl',
            'rootkit',
            'httptunnel',
            'ps',
            'sqlattack',
            'xterm'], 'U2R', inplace=True)
    #将标签列中的内容编码为数字型内容
    encoder= LabelEncoder().fit(labels)
    labels = encoder.transform(labels)
    labels = pd.DataFrame(labels)
    data[41] = labels
    temp_data = data
    #X = temp_data.drop(data.columns[[19, 20, 41, 42]], axis=1)
    temp_data2 = temp_data.drop(data.columns[[19, 20, 41, 42]], axis=1)
    X = temp_data2.iloc[:,[2, 3, 4, 6, 9, 10, 11, 13, 20, 22, 23, 26, 27, 28, 30, 32, 33, 34, 35, 36]]
    #X = temp_data2.iloc[:,[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 38]]
    Y = temp_data[41]
    return X,Y
def NB15_data_read_2(filepath):
    data = pd.read_csv(filepath)
    # 将第2、3、4列的字符型特征编码为数字特征
    encoder = LabelEncoder().fit(data['proto'])
    data1 = encoder.transform(data['proto'])
    data1 = pd.DataFrame(data1)
    data['proto'] = data1
    encoder = LabelEncoder().fit(data['service'])
    data2 = encoder.transform(data['service'])
    data2 = pd.DataFrame(data2)
    data['service'] = data2
    encoder = LabelEncoder().fit(data['state'])
    data3 = encoder.transform(data['state'])
    data3 = pd.DataFrame(data3)
    data['state'] = data3
    data = data.drop(['id'], axis=1)
    labels = data['attack_cat']
    #将标签列中的内容编码为数字型内容
    encoder= LabelEncoder().fit(labels)
    labels = encoder.transform(labels)
    labels = pd.DataFrame(labels)
    data['attack_cat'] = labels
    #将前42列赋值给X，即特征；将标签列赋值给Y
    X = data.drop(data.columns[[42,43]], axis=1)
    Y = data['attack_cat']
    my_list = list(range(42))
    X.columns = my_list
    #X = X.iloc[:,[1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 19, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
    X = X.iloc[:,[1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 19, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40]]
    return X,Y

def New_data_read_feature(filepath):
    data = pd.read_csv(filepath)
    encoder = LabelEncoder().fit(data['frame.protocols'])
    data1 = encoder.transform(data['frame.protocols'])
    data1 = pd.DataFrame(data1)
    data['frame.protocols'] = data1
    encoder = LabelEncoder().fit(data['ip.dst'])
    data2 = encoder.transform(data['ip.dst'])
    data2 = pd.DataFrame(data2)
    data['ip.dst'] = data2
    encoder = LabelEncoder().fit(data['ip.src'])
    data3 = encoder.transform(data['ip.src'])
    data3 = pd.DataFrame(data3)
    data['ip.src'] = data3
    data = data.drop(['frame.time_utc'],axis=1)
    my_list = list(range(17))
    data.columns = my_list
    X = data.iloc[:,[0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14]]
    #X = data.iloc[:,0:16]
    X=X.drop(13,axis=1)
    Y = data[16]

    return X,Y

x, y = nsl_data_read_feature_2("data/KDDTrain+.txt")
x2,y2 = NB15_data_read_2("data/UNSW-NB15/UNSW_NB15_training-set.csv")
x3,y3 = New_data_read_feature('data\TempDataset.csv')

#result = MIC(x2,y2, discrete_features='auto', n_neighbors=1)
#result = MIC(x3,y3, discrete_features='auto', n_neighbors=1)
x = MinMaxScaler().fit_transform(x)
x = pd.DataFrame(x)
x2 = MinMaxScaler().fit_transform(x2)
x2 = pd.DataFrame(x2)
x3 = MinMaxScaler().fit_transform(x3)
x3 = pd.DataFrame(x3)
#result = MIC(x,y, discrete_features='auto', n_neighbors=1)


for column in x.columns:
    x[column] += 0.001
f_num_1 = x.shape[1]

for column in x2.columns:
    x2[column] += 0.001
f_num_2 = x2.shape[1]

for column in x3.columns:
    x3[column] += 0.001
f_num_3 = x3.shape[1]

def feature_pivot_selection(data, c, f_num):
    pivot_index=[]
    for j in range(0,c):
        if j == 0:
            pivot = pd.DataFrame(data[3])
            print(pivot.shape)
            #temp_data = data.drop(index=0)
            pivot_index.append(3)
            #print(temp_data.shape)

        else:
            temp=[]
            print("计算第%d个枢轴" %(j+1) )
            for k in range(0,f_num):
                MIsum_of_feature = 0
                for l in range(0,j):
                    MIsum_of_feature += (MIS(data[pivot_index[l]], data[k]) / entropy(data[pivot_index[l]], data[k]))
                    #MIsum_of_feature += (MIS(data[pivot_index[l]], data[k]))
                temp.append(MIsum_of_feature)
            temp_index = np.argmin(temp)
            print(temp_index)
            pivot[j-1]=data[temp_index]
            #temp_data = temp_data.drop(index=temp_index)
            pivot_index.append(temp_index)
    return pivot_index

def feature_group_selection(data, pivot_index,f_num):
    #temp_data = data.drop(data.columns[pivot_index], axis=1)
    #temp_col_num = temp_data.shape[1]
    group_index = []
    for k in range(0, f_num):
        temp = []
        for l in range(0,len(pivot_index)):
            temp.append(MIS(data[pivot_index[l]], data[k]) / entropy(data[pivot_index[l]],data[k]))
            #temp.append(MIS(data[pivot_index[l]], data[k]))
        temp_index = np.argmax(temp)
        group_index.append(temp_index)
    return group_index
'''
X_new, selected_features = mRMR(x, y, 20)
print(X_new)
print(selected_features)
'''

pivot_index = feature_pivot_selection(x,2, f_num_1)
group_index = feature_group_selection(x, pivot_index, f_num_1)
print(pivot_index)
print(group_index)
print('元素索引：',[i for i, l in enumerate(group_index) if l == 0])
print('元素索引：',[i for i, l in enumerate(group_index) if l == 1])
print('元素索引：',[i for i, l in enumerate(group_index) if l == 2])

#result = MIS(x[0],x[2])
#print(result)

'''
X = [i for i in range(1,42)]
average = np.mean(result)
index = np.argwhere(result > average)
#print(result)
print(result2)
'''
'''
X = np.array(X)
plt.plot(X, result, linestyle='--', marker='o')
#plt.hlines(average, xmin = 0, xmax = 16, ls = '--', lw = 2,\
#           color = 'red', label = 'average')
#plt.grid(True)
plt.show()
#print("互信息相关系数:", metrics.mutual_info_score(x[1], x[3]))
'''