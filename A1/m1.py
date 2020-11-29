# Gender Classification based on MLP
# import necessary API
import cv2
import pandas
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

images = np.zeros(shape=(5000, 218, 178, 3))
labels = np.zeros(shape=(5000, 1))
test_size = 0.2

data = pandas.read_csv("../Datasets/celeba/labels.csv", header=None)
for n in range(5000):
    image = cv2.imread("../Datasets/celeba/img/%s.jpg" %n, cv2.IMREAD_COLOR)
    label = np.array(data.loc[n+1]).tolist()
    label = int(str(label[0]).split('\t')[-2])
    images[n] = image
    if label == -1: label = 0
    labels[n] = label

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=test_size,
                                                    random_state=123,
                                                    stratify=labels)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train[:100]
X_test = X_test[:100]
Y_train = Y_train[:100]
Y_test = Y_test[:100]

X_train = X_train.reshape(X_train.shape[0], 218*178*3)
X_test = X_test.reshape(X_test.shape[0], 218*178*3)
Y_train = Y_train.reshape(Y_train.shape[0],)
Y_test = Y_test.reshape(Y_test.shape[0],)

#(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# 把所有图片变为一维向量
#X_train = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # Xtr_rows 变成 50000 x 3072
#X_test = X_test.reshape(X_test.shape[0], 32 * 32 * 3) # Xte_rows 变成 10000 x 3072

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    #X是NXD的数组，其中每一行代表一个样本，Y是N行的一维数组，对应X的标签
    # 最近邻分类器就是简单的记住所有的数据
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    #X是NXD的数组，其中每一行代表一个图片样本
    #看一下测试数据有多少行
    num_test = X.shape[0]
    # 确认输出的结果类型符合输入的类型
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # 循环每一行，也就是每一个样本
    for i in range(num_test):
      # 找到和第i个测试图片距离最近的训练图片
      # 计算他们的L1距离
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # 拿到最小那个距离的索引
      Ypred[i] = self.ytr[min_index] # 预测样本的标签，其实就是跟他最近的训练数据样本的标签
    return Ypred


nn = NearestNeighbor() # 创建一个最近邻分类器的类，相当于初始化
nn.train(X_train, Y_train) # 把训练数据给模型，训练
Yte_predict = nn.predict(X_test) # 预测测试集的标签
# 算一下分类的准确率，这里取的是平均值
print('accuracy: %f' % ( np.mean(Yte_predict == Y_test) ))

"""
neigh = KNeighborsClassifier()
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)
score = metrics.accuracy_score(Y_test, Y_pred)
print(score)
"""