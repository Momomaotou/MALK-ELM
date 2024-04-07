'''
from DataReading import NB15_data_read, nsl_data_read
from FeatureRead import NB15_data_read_feature
#X,Y = NB15_data_read("data/UNSW-NB15/UNSW_NB15_training-set.csv", 0.1)
X,Y = NB15_data_read_feature("data/UNSW-NB15/UNSW_NB15_training-set.csv", 0.1)
Xtr,Ytr = nsl_data_read("data/KDDTrain+.txt", sampling_prop=0.05)
print(X[0])
print(Y)
'''
import matplotlib.pyplot as plt
import numpy as np

#访问一个cmap，顺带可以访问它的.N属性获得它的颜色数目
cmap = plt.get_cmap("Dark2")
colorseries = cmap([0, 1, 2, 3, 4])

cmap.N			# 获得该cmap的颜色总数
colorseries		# 获得cmap的

#在访问出cmap之中的颜色序列colorseries之后，我们用这三种颜色进行绘图：
x = np.linspace(-1, 1, 80)
'''
y1 = (0.2 * x +1)**1
y2 = (0.2 * x +1)**2
y3 = (0.2 * x +1)**3
y4 = (0.2 * x +1)**4
y5 = (0.2 * x +1)**5
'''
y1 = np.exp(-2*(x-0.2)**2)
y2 = np.exp(-4*(x-0.2)**2)
y3 = np.exp(-6*(x-0.2)**2)
y4 = np.exp(-8*(x-0.2)**2)
y5 = np.exp(-10*(x-0.2)**2)
plt.figure()
plt.plot(x, y1, label='γ=2', color = colorseries[0])
plt.plot(x, y2, label='γ=4', color = colorseries[1])
plt.plot(x, y3, label='γ=6', color = colorseries[2])
plt.plot(x, y4, label='γ=8', color = colorseries[3])
plt.plot(x, y5, label='γ=10', color = colorseries[4])
plt.legend()
plt.grid(True)  # 显示网格线
plt.show()  # 显示图像