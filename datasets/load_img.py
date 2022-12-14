import numpy as np
import os
import gzip

# references
# https://github.com/zalandoresearch/fashion-mnist
def _load_img(folder):
    cachefile = os.path.join(os.path.dirname(__file__), f"{folder}/{folder}_data.npz")
    if os.path.exists(cachefile):
        return np.load(cachefile).values()
    train_x_file = os.path.join(os.path.dirname(__file__), f'{folder}/train-images-idx3-ubyte.gz')
    with gzip.open(train_x_file, 'rb') as f:
        train_x = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape((-1, 784))
        train_x = (train_x / 255.0).astype(np.float64)
    test_x_file = os.path.join(os.path.dirname(__file__), f'{folder}/t10k-images-idx3-ubyte.gz')
    with gzip.open(test_x_file, 'rb') as f:
        test_x = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape((-1, 784))
        test_x = (test_x / 255.0).astype(np.float64)
    train_y_file = os.path.join(os.path.dirname(__file__), f'{folder}/train-labels-idx1-ubyte.gz')
    with gzip.open(train_y_file, 'rb') as f:
        train_y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    test_y_file = os.path.join(os.path.dirname(__file__), f'{folder}/t10k-labels-idx1-ubyte.gz')
    with gzip.open(test_y_file, 'rb') as f:
        test_y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    np.savez(cachefile, train_x, test_x, train_y, test_y)
    return train_x, test_x, train_y, test_y

load_mnist = lambda : _load_img('mnist')
load_fashion_mnist = lambda : _load_img('fashion_mnist')

if __name__ == '__main__':
    load_mnist()
    load_fashion_mnist()
