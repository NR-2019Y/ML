import numpy as np
import os
import gzip


# references
# https://github.com/zalandoresearch/fashion-mnist

# download
# http://yann.lecun.com/exdb/mnist/

def load_mnist():
    cachefile = os.path.join(os.path.dirname(__file__), "mnist_data.npz")
    if os.path.exists(cachefile):
        return np.load(cachefile).values()
    train_x_file = os.path.join(os.path.dirname(__file__), 'train-images-idx3-ubyte.gz')
    with gzip.open(train_x_file, 'rb') as f:
        train_x = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape((-1, 784))
        train_x = (train_x / 255.0).astype(np.float32)
    test_x_file = os.path.join(os.path.dirname(__file__), 't10k-images-idx3-ubyte.gz')
    with gzip.open(test_x_file, 'rb') as f:
        test_x = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape((-1, 784))
        test_x = (test_x / 255.0).astype(np.float32)
    train_y_file = os.path.join(os.path.dirname(__file__), 'train-labels-idx1-ubyte.gz')
    with gzip.open(train_y_file, 'rb') as f:
        train_y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    test_y_file = os.path.join(os.path.dirname(__file__), 't10k-labels-idx1-ubyte.gz')
    with gzip.open(test_y_file, 'rb') as f:
        test_y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    np.savez(cachefile, train_x, test_x, train_y, test_y)
    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    load_mnist()
    # train_x, test_x, train_y, test_y = load_fashion_mnist()
    # for e in train_x, test_x, train_y, test_y:
    #     print(e.shape)
