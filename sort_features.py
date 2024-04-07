
import numpy as np
'''
# 定义一维列表
one_dimensional = [0, 0, 1, 1, 2 ,2 ,3,3]


def convert_to_matrix(labels):
    unique_labels = list(set(labels))  # 获取所有不重复的标签

    num_rows = len(labels)  # 行数等于样本数量
    num_cols = len(unique_labels)  # 列数等于标签种类数量

    matrix = np.zeros((num_rows, num_cols), dtype=int)  # 创建全零矩阵
    print(matrix)
    for i in range(0, num_rows):

        matrix[i][labels[i]] = 1  # 根据索引位置设置相应元素为1

    return matrix

two_dimensional = convert_to_matrix(one_dimensional)
print(two_dimensional)
'''
X=[4, 6, 27, 23, 34, 3, 2, 26, 11, 35, 36, 32, 9, 22, 28, 20, 13, 10, 33, 30]
#X2 = [11, 36, 34, 41, 2, 9, 1, 37, 31, 33, 6, 28, 40, 10, 29, 3, 38, 26, 32, 19, 35, 22, 30, 39, 25, 5, 13, 14, 4, 27]
#X3 = [6, 4, 9, 10, 3, 1, 0, 2, 11, 13, 14, 8, 12]
#X.sort()
#X2.sort()
X.sort()
#print(X)
#print(X2)
print(X)

#Y=[2, 3, 4, 6, 9, 10, 11, 13, 20, 22, 23, 26, 27, 28, 30, 32, 33, 34, 35, 36]]
#Y2=[1, 2, 3, 4, 5, 6, 9, 10, 11, 13, 14, 19, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
Y3=[0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14]