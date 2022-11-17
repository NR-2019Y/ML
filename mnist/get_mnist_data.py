import time
import numpy as np

# 下载数据
# https://pjreddie.com/media/files/mnist_train.csv
# https://pjreddie.com/media/files/mnist_test.csv

def get_sklearnstyle_mnist_data():
    dtrain = np.loadtxt("_data/mnist_train.csv", delimiter=',', dtype=np.int32)
    dtest  = np.loadtxt("_data/mnist_test.csv",  delimiter=',', dtype=np.int32)
    train_x = dtrain[:, 1:] / 255
    test_x  = dtest [:, 1:] / 255
    train_y = dtrain[:, 0]
    test_y  = dtest [:, 0]
    return train_x, test_x, train_y, test_y

tic = time.time()
train_x, test_x, train_y, test_y = get_sklearnstyle_mnist_data()
print("用时", time.time() - tic)
np.savez("_data/sklearnstyle_mnist_data", train_x, test_x, train_y, test_y)
