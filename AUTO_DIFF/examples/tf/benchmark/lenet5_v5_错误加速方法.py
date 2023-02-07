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

tf.config.experimental_run_functions_eagerly(True)


@tf.function
def train():
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
                logits = model(xi, training=True)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=yi)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
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


train()
# [1]TIME:22.484792709350586s COST:0.036519404500722885 ACC:0.9586499929428101 TEST_COST:0.036519404500722885 TEST_ACC:0.9883999824523926
# [2]TIME:10.464116096496582s COST:0.03386634960770607 ACC:0.988349974155426 TEST_COST:0.03386634960770607 TEST_ACC:0.9894999861717224
# [3]TIME:10.4430570602417s COST:0.02629818022251129 ACC:0.9918666481971741 TEST_COST:0.02629818022251129 TEST_ACC:0.9918000102043152
# [4]TIME:10.43070363998413s COST:0.029514184221625328 ACC:0.9940833449363708 TEST_COST:0.029514184221625328 TEST_ACC:0.991100013256073
# [5]TIME:10.432788848876953s COST:0.027550216764211655 ACC:0.9952333569526672 TEST_COST:0.027550216764211655 TEST_ACC:0.9911999702453613
# [6]TIME:10.445068120956421s COST:0.03417324647307396 ACC:0.9963499903678894 TEST_COST:0.03417324647307396 TEST_ACC:0.989300012588501
# [7]TIME:10.427319288253784s COST:0.02877364121377468 ACC:0.996649980545044 TEST_COST:0.02877364121377468 TEST_ACC:0.9918000102043152
# [8]TIME:10.430953979492188s COST:0.032897163182497025 ACC:0.9962000250816345 TEST_COST:0.032897163182497025 TEST_ACC:0.9907000064849854
# [9]TIME:10.451010465621948s COST:0.040055662393569946 ACC:0.9975333213806152 TEST_COST:0.040055662393569946 TEST_ACC:0.9904000163078308
# [10]TIME:10.454001665115356s COST:0.03642953559756279 ACC:0.997783362865448 TEST_COST:0.03642953559756279 TEST_ACC:0.9908999800682068
# 用时 116.51461100578308

