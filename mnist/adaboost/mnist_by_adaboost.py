import numpy as np
import sys, time
import my_adaboost
from sklearn.feature_selection import VarianceThreshold

def main(n_estimators):
    train_x, test_x, train_y, test_y = np.load("../_data/sklearnstyle_mnist_data.npz").values()
    fobj = VarianceThreshold()
    fobj.fit(train_x)
    train_x = fobj.transform(train_x)
    test_x = fobj.transform(test_x)
    for e in train_x, test_x, train_y, test_y:
        print(e.shape)
    print("开始训练", flush=True)
    tic = time.time()
    aobj = my_adaboost.AdaBoostMulticlassSolver(my_adaboost.SimpleClassifier(), n_estimators=n_estimators)
    aobj.fit(train_x, train_y)
    print("运行时间", time.time() - tic)
    print("训练集准确率", aobj.score(train_x, train_y))
    print("测试集准确率", aobj.score(test_x, test_y))

if __name__ == '__main__':
    n_estimators = int(sys.argv[1])
    main(n_estimators)
