import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class KELM(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=None, C=1):
        self.gamma = gamma
        self.C =C
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]

        self.H = X

        #self.beta = np.random.randn(self.n_hidden, y.shape[0]) # 随机初始化隐层权重
        #self.alpha = np.dot(np.linalg.inv(self.H + np.eye(X.shape[0])), y)
        #H = np.sqrt(2 / self.n_hidden) * np.cos(self.K @ self.beta) # TLU函数作为激活函数
        self.alpha = np.dot(np.linalg.inv(self.H + np.eye(X.shape[0])/self.C), y) # 输出层权重

    def predict(self, X):


        H_test = X
        y_pred = H_test @ self.alpha
        return y_pred

    def decision_function(self, X):

        return self.predict(X)