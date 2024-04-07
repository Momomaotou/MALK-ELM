import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DataReading import New_data_read
import glob
import numpy as np

def stratified_sampling(data, sampling_prop, class_num, label_column):
    temp_data=[]
    fin_data=pd.DataFrame()
    for i in range (0,class_num):
        temp_data.append(data[data[label_column] == i ])
        if i == 2 :
            sampling_size = int(len(temp_data[i]) * sampling_prop)
        else:
            sampling_size = int(len(temp_data[i]) * sampling_prop)
        samples = temp_data[i].sample(sampling_size)
        fin_data = pd.concat([fin_data, samples])
    return fin_data

def stratified_sampling_NB15(data, sampling_prop, class_num):
    temp_data=[]
    fin_data=pd.DataFrame()
    for i in range (0,class_num):
        temp_data.append(data[data['attack_cat'] == i ])
        sampling_size = int(len(temp_data[i]) * sampling_prop)
        samples = temp_data[i].sample(sampling_size)
        fin_data = pd.concat([fin_data, samples])
    return fin_data
def nsl_data_read_feature(filepath, sampling_prop):
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
    temp_data = stratified_sampling(data, sampling_prop=sampling_prop, class_num=5, label_column=41)
    Y = temp_data[41]
    #temp_data = temp_data.drop(data.columns[[19, 20, 41, 42]], axis=1)
    feature_1 = temp_data
    temp_data = temp_data.drop(data.columns[[19, 20, 41, 42]], axis=1)
    '''
    feature_2 = temp_data.iloc[:,0:22]
    feature_3 = temp_data.iloc[:,22:31]#时间特征
    feature_4 = temp_data.iloc[:,31:41]#空间特征
    '''
    #feature_5 = temp_data.iloc[:,[3, 6, 21, 22, 23, 24, 28, 34, 35, 36]]
    #feature_6 = temp_data.iloc[:,[0, 5, 8, 10, 12, 14, 15, 18]]
    #feature_7 = temp_data.iloc[:,[1, 2, 4, 7, 9, 11, 13, 16, 17, 19, 20, 25, 26, 27, 29, 30, 31, 32, 33, 37, 38]]
    temp_data2 = temp_data.iloc[:,[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 38]]
    feature_8 = temp_data2.iloc[:, [1, 4, 17, 18, 21, 26, 27, 28]]
    feature_9 = temp_data2.iloc[:, [3, 6, 8, 11, 14]]
    feature_10 = temp_data2.iloc[:, [0, 2, 5, 7, 9, 10, 12, 13, 15, 16, 19, 20, 22, 23, 24, 25, 29]]
    feature_11 = temp_data2.iloc[:,[1, 4, 5, 9, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29]]
    feature_12 = temp_data2.iloc[:, [0, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 23]]

    X = []
    X.append(feature_1)
    '''
    X.append(feature_2)
    X.append(feature_3)
    X.append(feature_4)
    '''
    #X.append(feature_5)
    #X.append(feature_6)
    #X.append(feature_7)
    X.append(feature_8)
    X.append(feature_9)
    X.append(feature_10)
    X.append(feature_11)
    X.append(feature_12)
    X.append(temp_data2)

    #X = temp_data[[2, 3, 4, 5, 11, 22, 24, 25, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38]]
    #Y = temp_data[41]
    return X, Y

def NB15_data_read_feature(filepath, sampling_prop):
    data = pd.read_csv(filepath)
    # 将第3、4、5列的字符型特征编码为数字特征
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
    temp_data = stratified_sampling_NB15(data, sampling_prop=sampling_prop, class_num=10)
    Y = temp_data['attack_cat']
    temp_data = temp_data.drop(data.columns[[42, 43]], axis=1)
    my_list = list(range(42))
    temp_data.columns = my_list
    feature_1 = temp_data
    temp_data2 = temp_data.iloc[:, [1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 19, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40]]
    feature_2 = temp_data2.iloc[:,[6, 8, 18, 19, 20, 21, 22, 23, 25, 26]]
    feature_3 = temp_data2.iloc[:,[1, 16, 17, 24]]
    feature_4 = temp_data2.iloc[:,[0, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15]]
    #
    X = []
    X.append(feature_1)
    X.append(feature_2)
    X.append(feature_3)
    X.append(feature_4)
    #X = temp_data[['dur','proto','service','dpkts','sbytes','dbytes','rate','dttl','sload','dload','sinpkt','dinpkt','sjit','tcprtt','synack','smean','dmean','ct_state_ttl','ct_src_dport_ltm','ct_dst_sport_ltm','ct_srv_dst']]

    return X,Y

def New_data_read_feature(filepath):
    tempX, tempY = New_data_read('data\TempDataset.csv')
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
    temp_data = data.drop(['frame.time_utc'],axis=1)
    my_list = list(range(13))
    temp_data.columns = my_list
    #feature_2 = data[['ip.src','tcp.srcport','ip.dst','tcp.dstport','frame.protocols','frame.time_delta','frame.time_relative','frame.time_delta_displayed']]
    #feature_1 = data[['ip.src','tcp.srcport','ip.dst','tcp.dstport','frame.protocols','eth.src.oui','eth.dst.oui','tcp.len','tcp.ack','tcp.analysis.bytes_in_flight','tcp.analysis.ack_rtt']]
    feature_1 = temp_data
    temp_data2 = temp_data.iloc[:, [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13]]
    feature_2 = temp_data2.iloc[:, [0, 3, 5, 8]]
    feature_3 = temp_data2.iloc[:, [1, 2, 7, 10]]
    feature_4 = temp_data2.iloc[:, [4, 6, 9, 11]]
    X = []
    X.append(feature_1)
    X.append(feature_2)
    X.append(feature_3)
    X.append(feature_4)
    Y = data['label']

    return X,Y