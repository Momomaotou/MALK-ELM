import numpy as np
from sklearn import preprocessing


class Normalization():
    def __init__(self, data = None, datas = None):#data是一维数组，datas是二维数组
        self.data = data
        self.datas = datas
    def minMax1v(self):
        # 1、对于方差非常小的属性可以增强其稳定性；
        # 2、维持稀疏矩阵中为0的条目。
        nm = []
        for x in self.data:
            x = float(x - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
            nm.append(x)
        return nm
    def minMax2v(self):
        # 第一步求每个列中元素到最小值距离占该列最大值和最小值距离的比例，这实际上已经是将数据放缩到了[0, 1]区间上
        # 第二步将标准化的数据映射到给定的[min, max]区间
        min_max_scaler = preprocessing.MinMaxScaler()
        datas_minmax = min_max_scaler.fit_transform(self.datas)
        return datas_minmax
    def Z_score(self):
        nm = []
        for x in self.data:
            x = float(x - self.data.mean()) / self.data.std()
            nm.append(x)
        return nm
