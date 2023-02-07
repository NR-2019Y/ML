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
        total_train_loss += tf.reduce_sum(loss)  # .numpy()
        train_nacc += tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(yi, logits))  # .numpy()
    train_cost, train_acc = total_train_loss / train_size, train_nacc / train_size

    total_test_loss = 0.
    test_nacc = 0.
    for xi, yi in test_iter:
        logits = model(xi, training=False)
        total_test_loss += tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=yi)
        )  # .numpy()
        test_nacc += tf.reduce_sum(tf.metrics.sparse_categorical_accuracy(yi, logits))  # .numpy()

    test_cost, test_acc = total_test_loss / test_size, test_nacc / test_size
    time_used = time.time() - ts
    print(f"[{epoch}]TIME:{time_used}s COST:{test_cost} ACC:{train_acc} TEST_COST:{test_cost} TEST_ACC:{test_acc}")
print("用时", time.time() - tic)
# [1]TIME:22.7285053730011s COST:0.03857175260782242 ACC:0.9587000012397766 TEST_COST:0.03857175260782242 TEST_ACC:0.9872000217437744
# [2]TIME:10.450164079666138s COST:0.03516824170947075 ACC:0.9885833263397217 TEST_COST:0.03516824170947075 TEST_ACC:0.9890999794006348
# [3]TIME:10.453494787216187s COST:0.02812795154750347 ACC:0.9922666549682617 TEST_COST:0.02812795154750347 TEST_ACC:0.9901999831199646
# [4]TIME:10.453661918640137s COST:0.031423136591911316 ACC:0.9938666820526123 TEST_COST:0.031423136591911316 TEST_ACC:0.9898999929428101
# [5]TIME:10.463224411010742s COST:0.02919466607272625 ACC:0.9952666759490967 TEST_COST:0.02919466607272625 TEST_ACC:0.9909999966621399
# [6]TIME:10.45029354095459s COST:0.030406177043914795 ACC:0.9957000017166138 TEST_COST:0.030406177043914795 TEST_ACC:0.9907000064849854
# [7]TIME:10.439327478408813s COST:0.028955083340406418 ACC:0.9965500235557556 TEST_COST:0.028955083340406418 TEST_ACC:0.9914000034332275
# [8]TIME:10.446352481842041s COST:0.03272898122668266 ACC:0.9973000288009644 TEST_COST:0.03272898122668266 TEST_ACC:0.9918000102043152
# [9]TIME:10.443070888519287s COST:0.036023180931806564 ACC:0.9979666471481323 TEST_COST:0.036023180931806564 TEST_ACC:0.9919000267982483
# [10]TIME:10.449196815490723s COST:0.05419763922691345 ACC:0.9971333146095276 TEST_COST:0.05419763922691345 TEST_ACC:0.9871000051498413
# 用时 116.83509469032288


# 含 .numpy()
# [1]TIME:24.569766521453857s COST:0.038806995964050295 ACC:0.9585333333333333 TEST_COST:0.038806995964050295 TEST_ACC:0.9873
# [2]TIME:12.409549474716187s COST:0.03878648881912231 ACC:0.9884 TEST_COST:0.03878648881912231 TEST_ACC:0.9884
# [3]TIME:12.390197038650513s COST:0.026759686279296875 ACC:0.9917333333333334 TEST_COST:0.026759686279296875 TEST_ACC:0.9918
# [4]TIME:12.373218297958374s COST:0.03611686601638794 ACC:0.9942 TEST_COST:0.03611686601638794 TEST_ACC:0.9889
# [5]TIME:12.386597871780396s COST:0.028482612252235414 ACC:0.9949333333333333 TEST_COST:0.028482612252235414 TEST_ACC:0.9914
# [6]TIME:12.394531965255737s COST:0.027579640007019044 ACC:0.9956333333333334 TEST_COST:0.027579640007019044 TEST_ACC:0.9911
# [7]TIME:12.408935546875s COST:0.032372799175977705 ACC:0.9972833333333333 TEST_COST:0.032372799175977705 TEST_ACC:0.9916
# [8]TIME:12.368473768234253s COST:0.02810882544517517 ACC:0.99675 TEST_COST:0.02810882544517517 TEST_ACC:0.9912
# [9]TIME:12.334931373596191s COST:0.031202509737014772 ACC:0.9969833333333333 TEST_COST:0.031202509737014772 TEST_ACC:0.9914
# [10]TIME:12.369096994400024s COST:0.03595182899236679 ACC:0.9975833333333334 TEST_COST:0.03595182899236679 TEST_ACC:0.9917
# 用时 136.0071783065796
