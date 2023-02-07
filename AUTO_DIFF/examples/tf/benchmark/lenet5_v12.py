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
model.build(input_shape=(None,) + train_x.shape[1:])
params = model.trainable_variables
model = tf.function(model)

tic = time.time()
epochs = 10
train_size = len(train_y)
test_size = len(test_y)
optimizer = optimizers.Adam(learning_rate=0.001)
for epoch in range(1, epochs + 1):
    ts = time.time()
    total_train_loss = 0.
    train_nacc = 0.
    for xi, yi in train_iter:
        with tf.GradientTape() as tape:
            logits = model(xi, training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=yi)
        grads = tape.gradient(loss, params)
        optimizer.apply_gradients(zip(grads, params))
        total_train_loss += tf.reduce_sum(loss)
        train_nacc += tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(yi, logits))
    train_cost, train_acc = total_train_loss / train_size, train_nacc / train_size

    total_test_loss = 0.
    test_nacc = 0.
    for xi, yi in test_iter:
        logits = model(xi, training=False)
        total_test_loss += tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=yi)
        )
        test_nacc += tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(yi, logits))

    test_cost, test_acc = total_test_loss / test_size, test_nacc / test_size
    time_used = time.time() - ts
    print(f"[{epoch}]TIME:{time_used}s COST:{test_cost} ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")
print("用时", time.time() - tic)
# [1]TIME:23.008384227752686s COST:0.0389426127076149 ACC:0.9586166739463806 TEST_COST:0.0389426127076149 TEST_ACC:0.9869999885559082
# [2]TIME:8.299229145050049s COST:0.031934816390275955 ACC:0.9884166717529297 TEST_COST:0.031934816390275955 TEST_ACC:0.989300012588501
# [3]TIME:8.303453207015991s COST:0.030502019450068474 ACC:0.9923666715621948 TEST_COST:0.030502019450068474 TEST_ACC:0.9904000163078308
# [4]TIME:8.295071601867676s COST:0.02807709388434887 ACC:0.9942333102226257 TEST_COST:0.02807709388434887 TEST_ACC:0.9912999868392944
# [5]TIME:8.311877489089966s COST:0.030119391158223152 ACC:0.9954166412353516 TEST_COST:0.030119391158223152 TEST_ACC:0.989799976348877
# [6]TIME:8.31123971939087s COST:0.0418061800301075 ACC:0.9958000183105469 TEST_COST:0.0418061800301075 TEST_ACC:0.9882000088691711
# [7]TIME:8.298534631729126s COST:0.026116907596588135 ACC:0.9968666434288025 TEST_COST:0.026116907596588135 TEST_ACC:0.9923999905586243
# [8]TIME:8.298739433288574s COST:0.03489198163151741 ACC:0.9969666600227356 TEST_COST:0.03489198163151741 TEST_ACC:0.9916999936103821
# [9]TIME:8.30128264427185s COST:0.02938525378704071 ACC:0.9973833560943604 TEST_COST:0.02938525378704071 TEST_ACC:0.9911999702453613
# [10]TIME:8.298724174499512s COST:0.04033897817134857 ACC:0.9973833560943604 TEST_COST:0.04033897817134857 TEST_ACC:0.9902999997138977
# 用时 97.7837598323822
