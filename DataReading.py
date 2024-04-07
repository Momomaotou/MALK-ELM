
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import glob
import numpy as np

def stratified_sampling(data, sampling_prop, class_num, label_column):
    temp_data=[]
    fin_data=pd.DataFrame()
    for i in range (0,class_num):
        temp_data.append(data[data[label_column] == i ])
        if i == 2:
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
def nsl_data_read(filepath, sampling_prop):
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
    #将前41列赋值给X，即特征；将标签列赋值给Y
    X = temp_data.drop(data.columns[[41,42]], axis=1)
    #X = data.drop(data.columns[[41, 42]], axis=1)
    Y = temp_data[41]
    #Y = data[41]
    return X,Y

def NB15_data_read(filepath, sampling_prop):
    # 获取文件夹中所有的CSV文件路径
    #csv_files = glob.glob('data/UNSW-NB15/*.csv')
    # 批量读取CSV文件
    #dataframes = []
    #for csv_file in csv_files:
    #    data = pd.read_csv(csv_file, header=None)
    #    data.drop(index=0, axis=0, inplace=True)
    #    dataframes.append(data)
    # 合并多个DataFrame
    #merged_data = pd.concat(dataframes)
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
    temp_data = stratified_sampling_NB15(data, sampling_prop=sampling_prop, class_num=10)
    #将前42列赋值给X，即特征；将标签列赋值给Y
    X = temp_data.drop(data.columns[[42,43]], axis=1)
    Y = temp_data['attack_cat']
    return X,Y

def New_data_read(filepath):
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

    X = data.drop(data.columns[[13,16]], axis=1)
    Y = data['label']

    return X,Y