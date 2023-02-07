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
X_ = tf.compat.v1.placeholder(tf.float32, shape=(None,) + train_x.shape[1:])
Y_ = tf.compat.v1.placeholder(tf.int64, shape=(None,))
logits = model(X_, training=True)
loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
sum_loss_ = tf.reduce_sum(loss_)
nacc_ = tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(Y_, logits))
trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_)

logits_test_ = model(X_, training=False)
sum_loss_test_ = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_test_, labels=Y_))
nacc_test_ = tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(Y_, logits_test_))

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
            batch_sum_loss, batch_nacc = sess.run([sum_loss_test_, nacc_test_], feed_dict={X_: xi, Y_: yi})
            total_test_loss += batch_sum_loss
            test_nacc += batch_nacc
        test_cost, test_acc = total_test_loss / test_size, test_nacc / test_size
        time_used = time.time() - ts
        print(f"[{epoch}]TIME:{time_used}s COST:{test_cost} ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")
print("用时", time.time() - tic)
# [1]TIME:18.098488569259644s COST:0.04362227792739868 ACC:0.95775 TEST_COST:0.04362227792739868 TEST_ACC:0.9868
# [2]TIME:7.223858594894409s COST:0.040379850673675534 ACC:0.9882 TEST_COST:0.040379850673675534 TEST_ACC:0.9873
# [3]TIME:7.225771903991699s COST:0.03251103157997132 ACC:0.9916 TEST_COST:0.03251103157997132 TEST_ACC:0.9905
# [4]TIME:7.2238969802856445s COST:0.025633438968658447 ACC:0.9933 TEST_COST:0.025633438968658447 TEST_ACC:0.9917
# [5]TIME:7.216560125350952s COST:0.027479250407218932 ACC:0.99515 TEST_COST:0.027479250407218932 TEST_ACC:0.9921
# [6]TIME:7.269716262817383s COST:0.03254400491714478 ACC:0.9962666666666666 TEST_COST:0.03254400491714478 TEST_ACC:0.9896
# [7]TIME:7.210285186767578s COST:0.03375932512283325 ACC:0.9964333333333333 TEST_COST:0.03375932512283325 TEST_ACC:0.9914
# [8]TIME:7.221844434738159s COST:0.027918221056461336 ACC:0.9966833333333334 TEST_COST:0.027918221056461336 TEST_ACC:0.9935
# [9]TIME:7.218725681304932s COST:0.05138747732639313 ACC:0.9968333333333333 TEST_COST:0.05138747732639313 TEST_ACC:0.9878
# [10]TIME:7.193302869796753s COST:0.035941382169723514 ACC:0.9973666666666666 TEST_COST:0.035941382169723514 TEST_ACC:0.9913
# 用时 84.41417336463928
