import numpy as np


class DataIter(object):
    def __init__(self, X, y, *, batch_size):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        size = len(self.y)
        rand_perm = np.random.permutation(size)
        for i in range(0, size, self.batch_size):
            idx = rand_perm[i: i + self.batch_size]
            yield self.X[idx], self.y[idx]
