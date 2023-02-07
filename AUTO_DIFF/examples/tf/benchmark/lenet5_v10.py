import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, losses, metrics
from AUTO_DIFF.data_iter import DataIter

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

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x[..., None].astype(np.float32) / 255.
test_x = test_x[..., None].astype(np.float32) / 255.
train_x = np.pad(train_x, [[0, 0], [2, 2], [2, 2], [0, 0]])
test_x = np.pad(test_x, [[0, 0], [2, 2], [2, 2], [0, 0]])
train_y = train_y.astype(np.int64)
test_y = test_y.astype(np.int64)

train_iter = DataIter(train_x, train_y, batch_size=128, shuffle=True)
test_iter = DataIter(test_x, test_y, batch_size=1000, shuffle=False)


def create_model(x):
    Wc1 = tf.Variable(tf.random.normal([5, 5, 1, 32], stddev=0.01, mean=0., dtype=tf.float32))
    Wc2 = tf.Variable(tf.random.normal([5, 5, 1, 64], stddev=0.01, mean=0., dtype=tf.float32))
    Wf1 = tf.Variable(tf.random.normal([1600, 1024], stddev=0.01, mean=0., dtype=tf.float32))
    Wf2 = tf.Variable(tf.random.normal([1024, 128], stddev=0.01, mean=0., dtype=tf.float32))
    Wf3 = tf.Variable(tf.random.normal([128, 10], stddev=0.01, mean=0., dtype=tf.float32))
    bc1 = tf.Variable(tf.zeros([32], dtype=tf.float32))
    bc2 = tf.Variable(tf.zeros([64], dtype=tf.float32))
    bf1 = tf.Variable(tf.zeros([1024], dtype=tf.float32))
    bf2 = tf.Variable(tf.zeros([128], dtype=tf.float32))
    bf3 = tf.Variable(tf.zeros([10], dtype=tf.float32))
    x = tf.nn.conv2d(x, Wc1, strides=[1, 1, 1, 1], padding='VALID') + bc1
    x = tf.nn.relu(x)
    x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    x = tf.nn.conv2d(x, Wc2, strides=[1, 1, 1, 1], padding='VALID') + bc2
    x = tf.nn.relu(x)
    x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    x = tf.reshape(x, [-1, 1600])
    x = tf.nn.relu(tf.matmul(x, Wf1) + bf1)
    x = tf.nn.relu(tf.matmul(x, Wf2) + bf2)
    return tf.matmul(x, Wf3) + bf3


X_ = tf.compat.v1.placeholder(tf.float32, shape=(None,) + train_x.shape[1:])
Y_ = tf.compat.v1.placeholder(tf.int64, shape=(None,))
logits = create_model(X_)
loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
sum_loss_ = tf.reduce_sum(loss_)
nacc_ = tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(Y_, logits))
trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_)

tic = time.time()
epochs = 10
train_size = len(train_y)
test_size = len(test_y)
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(1, epochs + 1):
        total_train_loss = 0.
        train_nacc = 0.
        ts = time.time()
        for xi, yi in train_iter:
            batch_sum_loss, batch_nacc, _ = sess.run([sum_loss_, nacc_, trainer], feed_dict={X_: xi, Y_: yi})
            total_train_loss += batch_sum_loss
            train_nacc += batch_nacc
        train_cost, train_acc = total_train_loss / train_size, train_nacc / train_size

        total_test_loss = 0.
        test_nacc = 0.
        for xi, yi in test_iter:
            batch_sum_loss, batch_nacc = sess.run([sum_loss_, nacc_], feed_dict={X_: xi, Y_: yi})
            total_test_loss += batch_sum_loss
            test_nacc += batch_nacc
        test_cost, test_acc = total_test_loss / test_size, test_nacc / test_size
        time_used = time.time() - ts
        print(f"[{epoch}]TIME:{time_used}s COST:{test_cost} ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")
print("用时", time.time() - tic)
# [1]TIME:16.819828033447266s COST:0.10916547927856446 ACC:0.8291 TEST_COST:0.10916547927856446 TEST_ACC:0.9664
# [2]TIME:6.766838073730469s COST:0.0535492341041565 ACC:0.97095 TEST_COST:0.0535492341041565 TEST_ACC:0.9822
# [3]TIME:6.753689289093018s COST:0.053626239013671875 ACC:0.9794333333333334 TEST_COST:0.053626239013671875 TEST_ACC:0.9826
# [4]TIME:6.762122869491577s COST:0.05639031257629395 ACC:0.9856166666666667 TEST_COST:0.05639031257629395 TEST_ACC:0.9821
# [5]TIME:6.760713577270508s COST:0.04220517301559448 ACC:0.988 TEST_COST:0.04220517301559448 TEST_ACC:0.9861
# [6]TIME:6.757901191711426s COST:0.02982727909088135 ACC:0.9907166666666667 TEST_COST:0.02982727909088135 TEST_ACC:0.9906
# [7]TIME:6.739657402038574s COST:0.028390232944488527 ACC:0.99135 TEST_COST:0.028390232944488527 TEST_ACC:0.9912
# [8]TIME:6.751420259475708s COST:0.02903502473831177 ACC:0.99235 TEST_COST:0.02903502473831177 TEST_ACC:0.991
# [9]TIME:6.772443771362305s COST:0.035664323806762696 ACC:0.9935833333333334 TEST_COST:0.035664323806762696 TEST_ACC:0.9878
# [10]TIME:6.759258985519409s COST:0.03061532096862793 ACC:0.9948833333333333 TEST_COST:0.03061532096862793 TEST_ACC:0.9911
# 用时 78.88835072517395
