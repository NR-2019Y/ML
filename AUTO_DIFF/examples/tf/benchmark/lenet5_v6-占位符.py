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
logits = model(X_)
loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
sum_loss_ = tf.reduce_sum(loss_)
nacc_ = tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(Y_, logits))
trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_)

# 也可以
# grads_ = tf.gradients(loss_, model.trainable_variables)
# trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).apply_gradients(zip(grads_, model.trainable_variables))


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
# [1]TIME:17.906850576400757s COST:0.045967582416534425 ACC:0.95615 TEST_COST:0.045967582416534425 TEST_ACC:0.9837
# [2]TIME:7.172150135040283s COST:0.03343165073394776 ACC:0.9868833333333333 TEST_COST:0.03343165073394776 TEST_ACC:0.9896
# [3]TIME:7.181892156600952s COST:0.04373206071853638 ACC:0.9914333333333334 TEST_COST:0.04373206071853638 TEST_ACC:0.9858
# [4]TIME:7.169977426528931s COST:0.024167806243896483 ACC:0.9932166666666666 TEST_COST:0.024167806243896483 TEST_ACC:0.9924
# [5]TIME:7.1593663692474365s COST:0.0457813512802124 ACC:0.9947666666666667 TEST_COST:0.0457813512802124 TEST_ACC:0.9857
# [6]TIME:7.162168264389038s COST:0.022690448129177095 ACC:0.9956166666666667 TEST_COST:0.022690448129177095 TEST_ACC:0.9928
# [7]TIME:7.151243448257446s COST:0.028439832091331482 ACC:0.9968166666666667 TEST_COST:0.028439832091331482 TEST_ACC:0.9911
# [8]TIME:7.176565647125244s COST:0.03769120074510574 ACC:0.9964333333333333 TEST_COST:0.03769120074510574 TEST_ACC:0.9903
# [9]TIME:7.15494441986084s COST:0.030222787272930145 ACC:0.9968666666666667 TEST_COST:0.030222787272930145 TEST_ACC:0.9922
# [10]TIME:7.1563379764556885s COST:0.03748782169818878 ACC:0.9973333333333333 TEST_COST:0.03748782169818878 TEST_ACC:0.9914
# 用时 83.72683095932007
