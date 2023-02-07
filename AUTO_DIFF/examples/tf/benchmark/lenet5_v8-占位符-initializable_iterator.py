import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, losses, metrics
from AUTO_DIFF.data_iter import DataIter
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(333)

all_gpu = tf.config.experimental.list_physical_devices('GPU')
if all_gpu:
    print('-' * 20, 'MODE:GPU', '-' * 20)
    for gpu in all_gpu:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print('-' * 20, 'MODE:CPU', '-' * 20)

model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(units=1024, activation='relu'),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=10)
])

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x[..., None].astype(np.float32) / 255.
test_x = test_x[..., None].astype(np.float32) / 255.
train_x = np.pad(train_x, [[0, 0], [2, 2], [2, 2], [0, 0]])
test_x = np.pad(test_x, [[0, 0], [2, 2], [2, 2], [0, 0]])
train_y = train_y.astype(np.int64)
test_y = test_y.astype(np.int64)


def create_dataset(X, y, batch_size, drop_remainder=False, shuffle=True, epochs=10):
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(4096)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder).repeat(epochs)
    return dataset


epochs = 10
train_batch_size, test_batch_size = 128, 1000
train_size = len(train_y)
test_size = len(test_y)
num_batch_each_epoch_train = math.ceil(train_size / train_batch_size)
num_batch_each_epoch_test = math.ceil(test_size / test_batch_size)

X_ = tf.compat.v1.placeholder(tf.float32, shape=(None,) + train_x.shape[1:])
Y_ = tf.compat.v1.placeholder(tf.int64, shape=(None,))
dtrain = create_dataset(X_, Y_, batch_size=train_batch_size, shuffle=True)
dtest = create_dataset(X_, Y_, batch_size=test_batch_size, shuffle=False)
train_iter = tf.compat.v1.data.make_initializable_iterator(dtrain)
test_iter = tf.compat.v1.data.make_initializable_iterator(dtest)
xi_train_, yi_train_ = train_iter.get_next()
xi_test_, yi_test_ = test_iter.get_next()

logits = model(X_)
loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
sum_loss_ = tf.reduce_sum(loss_)
nacc_ = tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(Y_, logits))
trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_)

tic = time.time()

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(train_iter.initializer, feed_dict={X_: train_x, Y_: train_y})
    sess.run(test_iter.initializer, feed_dict={X_: test_x, Y_: test_y})
    for epoch in range(1, epochs + 1):
        total_train_loss = 0.
        train_nacc = 0.
        ts = time.time()
        for _ in range(num_batch_each_epoch_train):
            xi, yi = sess.run([xi_train_, yi_train_])
            batch_sum_loss, batch_nacc, _ = sess.run([sum_loss_, nacc_, trainer], feed_dict={X_: xi, Y_: yi})
            total_train_loss += batch_sum_loss
            train_nacc += batch_nacc
        train_cost, train_acc = total_train_loss / train_size, train_nacc / train_size

        total_test_loss = 0.
        test_nacc = 0.
        for _ in range(num_batch_each_epoch_test):
            xi, yi = sess.run([xi_test_, yi_test_])
            batch_sum_loss, batch_nacc = sess.run([sum_loss_, nacc_], feed_dict={X_: xi, Y_: yi})
            total_test_loss += batch_sum_loss
            test_nacc += batch_nacc
        test_cost, test_acc = total_test_loss / test_size, test_nacc / test_size
        time_used = time.time() - ts
        print(f"[{epoch}]TIME:{time_used}s COST:{test_cost} ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")
print("用时", time.time() - tic)
# [1]TIME:19.072245597839355s COST:0.061966657066345215 ACC:0.9554666666666667 TEST_COST:0.061966657066345215 TEST_ACC:0.9796
# [2]TIME:8.160821437835693s COST:0.038955198287963864 ACC:0.9876333333333334 TEST_COST:0.038955198287963864 TEST_ACC:0.9879
# [3]TIME:8.147900819778442s COST:0.03309356827735901 ACC:0.9915833333333334 TEST_COST:0.03309356827735901 TEST_ACC:0.989
# [4]TIME:8.152174949645996s COST:0.02671728768348694 ACC:0.99335 TEST_COST:0.02671728768348694 TEST_ACC:0.9912
# [5]TIME:8.148587942123413s COST:0.029851652646064757 ACC:0.9953 TEST_COST:0.029851652646064757 TEST_ACC:0.9915
# [6]TIME:8.154478549957275s COST:0.03604084568023681 ACC:0.99555 TEST_COST:0.03604084568023681 TEST_ACC:0.9897
# [7]TIME:8.159964323043823s COST:0.036168484091758726 ACC:0.9966333333333334 TEST_COST:0.036168484091758726 TEST_ACC:0.9907
# [8]TIME:8.141690731048584s COST:0.0287416348695755 ACC:0.9961 TEST_COST:0.0287416348695755 TEST_ACC:0.992
# [9]TIME:8.152844905853271s COST:0.03287809472084045 ACC:0.9975 TEST_COST:0.03287809472084045 TEST_ACC:0.991
# [10]TIME:8.15078330039978s COST:0.027603231847286223 ACC:0.9978666666666667 TEST_COST:0.027603231847286223 TEST_ACC:0.9934
# 用时 94.11214447021484
