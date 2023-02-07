import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, losses, metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

train_iter = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_iter = train_iter.shuffle(buffer_size=4096).batch(batch_size=128)
test_iter = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size=1000)


class Model:
    def __init__(self):
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
        self.params = Wc1, Wc2, Wf1, Wf2, Wf3, bc1, bc2, bf1, bf2, bf3

    @tf.function
    def __call__(self, x):
        Wc1, Wc2, Wf1, Wf2, Wf3, bc1, bc2, bf1, bf2, bf3 = self.params
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


model = Model()
tic = time.time()
epochs = 10
train_size = len(train_y)
test_size = len(test_y)
optimizer = optimizers.Adam(learning_rate=0.001)
for epoch in range(1, epochs + 1):
    total_train_loss = 0.
    train_nacc = 0.
    ts = time.time()
    for xi, yi in train_iter:
        with tf.GradientTape() as tape:
            logits = model(xi)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=yi)
        grads = tape.gradient(loss, model.params)
        optimizer.apply_gradients(zip(grads, model.params))
        total_train_loss += tf.reduce_sum(loss)  # .numpy()
        train_nacc += tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(yi, logits))  # .numpy()
    train_cost, train_acc = total_train_loss / train_size, train_nacc / train_size

    total_test_loss = 0.
    test_nacc = 0.
    for xi, yi in test_iter:
        logits = model(xi)
        total_test_loss += tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=yi)
        )  # .numpy()
        test_nacc += tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(yi, logits))  # .numpy()

    test_cost, test_acc = total_test_loss / test_size, test_nacc / test_size
    time_used = time.time() - ts
    print(f"[{epoch}]TIME:{time_used}s COST:{test_cost} ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")
print("用时", time.time() - tic)
# @tf.function修饰__call__
# [1]TIME:20.87550401687622s COST:0.08986172080039978 ACC:0.8472166657447815 TEST_COST:0.08986172080039978 TEST_ACC:0.972000002861023
# [2]TIME:7.763920307159424s COST:0.052771471440792084 ACC:0.9741666913032532 TEST_COST:0.052771471440792084 TEST_ACC:0.9828000068664551
# [3]TIME:7.782207489013672s COST:0.044536348432302475 ACC:0.9817500114440918 TEST_COST:0.044536348432302475 TEST_ACC:0.9855999946594238
# [4]TIME:7.777348518371582s COST:0.03278813511133194 ACC:0.985883355140686 TEST_COST:0.03278813511133194 TEST_ACC:0.9890000224113464
# [5]TIME:7.766017913818359s COST:0.0327024981379509 ACC:0.9887833595275879 TEST_COST:0.0327024981379509 TEST_ACC:0.9898999929428101
# [6]TIME:7.763012647628784s COST:0.0306173637509346 ACC:0.9908833503723145 TEST_COST:0.0306173637509346 TEST_ACC:0.9898999929428101
# [7]TIME:7.784641981124878s COST:0.03374514356255531 ACC:0.9920833110809326 TEST_COST:0.03374514356255531 TEST_ACC:0.989300012588501
# [8]TIME:7.768442869186401s COST:0.031753093004226685 ACC:0.993483304977417 TEST_COST:0.031753093004226685 TEST_ACC:0.9908000230789185
# [9]TIME:7.784359455108643s COST:0.027213403955101967 ACC:0.9944666624069214 TEST_COST:0.027213403955101967 TEST_ACC:0.991599977016449
# [10]TIME:7.770740270614624s COST:0.034733936190605164 ACC:0.9944666624069214 TEST_COST:0.034733936190605164 TEST_ACC:0.989799976348877
# 用时 90.84566235542297

# 没有@tf.function修饰__call__
# [1]TIME:21.04878568649292s COST:0.08929816633462906 ACC:0.84743332862854 TEST_COST:0.08929816633462906 TEST_ACC:0.972100019454956
# [2]TIME:8.369320154190063s COST:0.05218712240457535 ACC:0.9740333557128906 TEST_COST:0.05218712240457535 TEST_ACC:0.9833999872207642
# [3]TIME:8.36272144317627s COST:0.04491066187620163 ACC:0.9815166592597961 TEST_COST:0.04491066187620163 TEST_ACC:0.9848999977111816
# [4]TIME:8.333733558654785s COST:0.03316323086619377 ACC:0.9858666658401489 TEST_COST:0.03316323086619377 TEST_ACC:0.989300012588501
# [5]TIME:8.33917236328125s COST:0.036643341183662415 ACC:0.9887666702270508 TEST_COST:0.036643341183662415 TEST_ACC:0.9884999990463257
# [6]TIME:8.354501485824585s COST:0.030340563505887985 ACC:0.9904500246047974 TEST_COST:0.030340563505887985 TEST_ACC:0.9901999831199646
# [7]TIME:8.343841791152954s COST:0.030202819034457207 ACC:0.9918666481971741 TEST_COST:0.030202819034457207 TEST_ACC:0.9908999800682068
# [8]TIME:8.400402069091797s COST:0.03065330535173416 ACC:0.993066668510437 TEST_COST:0.03065330535173416 TEST_ACC:0.9909999966621399
# [9]TIME:8.359248638153076s COST:0.02843637391924858 ACC:0.9947333335876465 TEST_COST:0.02843637391924858 TEST_ACC:0.9916999936103821
# [10]TIME:8.347809314727783s COST:0.03134008124470711 ACC:0.9949833154678345 TEST_COST:0.03134008124470711 TEST_ACC:0.9907000064849854
# 用时 96.26363325119019
