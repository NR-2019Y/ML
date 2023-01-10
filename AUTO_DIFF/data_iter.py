import numpy as np
import tensorflow as tf


class DataIter(object):
    def __init__(self, X, y, *, batch_size, shuffle=True):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if not self.shuffle:
            for i in range(0, len(self.y), self.batch_size):
                curr_slice = slice(i, i + self.batch_size)
                yield self.X[curr_slice], self.y[curr_slice]
        else:
            size = len(self.y)
            rand_perm = np.random.permutation(size)
            for i in range(0, size, self.batch_size):
                idx = rand_perm[i:i + self.batch_size]
                yield self.X[idx], self.y[idx]


class TensorIter(object):
    def __init__(self, X, y, *, batch_size, shuffle=True):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if not self.shuffle:
            for i in range(0, len(self.y), self.batch_size):
                curr_slice = slice(i, i + self.batch_size)
                yield self.X[curr_slice], self.y[curr_slice]
        else:
            size = len(self.y)
            rand_perm = np.random.permutation(size)
            for i in range(0, size, self.batch_size):
                idx = rand_perm[i: i + self.batch_size]
                yield tf.gather(self.X, idx, axis=0), tf.gather(self.y, idx, axis=0)
