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
model.compile(optimizers.Adam(learning_rate=0.001),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True, name='LOSS'),
              metrics=metrics.sparse_categorical_accuracy)

tic = time.time()
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=128, validation_batch_size=1000, epochs=10)
print("用时", time.time() - tic)
# 469/469 [==============================] - 14s 29ms/step - loss: 0.1436 - sparse_categorical_accuracy: 0.9554 - val_loss: 0.0354 - val_sparse_categorical_accuracy: 0.9887
# Epoch 2/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0391 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0269 - val_sparse_categorical_accuracy: 0.9910
# Epoch 3/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0273 - sparse_categorical_accuracy: 0.9915 - val_loss: 0.0243 - val_sparse_categorical_accuracy: 0.9916
# Epoch 4/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0210 - sparse_categorical_accuracy: 0.9933 - val_loss: 0.0292 - val_sparse_categorical_accuracy: 0.9896
# Epoch 5/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0159 - sparse_categorical_accuracy: 0.9948 - val_loss: 0.0319 - val_sparse_categorical_accuracy: 0.9886
# Epoch 6/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0125 - sparse_categorical_accuracy: 0.9959 - val_loss: 0.0355 - val_sparse_categorical_accuracy: 0.9893
# Epoch 7/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0106 - sparse_categorical_accuracy: 0.9967 - val_loss: 0.0292 - val_sparse_categorical_accuracy: 0.9920
# Epoch 8/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0089 - sparse_categorical_accuracy: 0.9972 - val_loss: 0.0259 - val_sparse_categorical_accuracy: 0.9925
# Epoch 9/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0082 - sparse_categorical_accuracy: 0.9974 - val_loss: 0.0303 - val_sparse_categorical_accuracy: 0.9922
# Epoch 10/10
# 469/469 [==============================] - 8s 17ms/step - loss: 0.0082 - sparse_categorical_accuracy: 0.9974 - val_loss: 0.0252 - val_sparse_categorical_accuracy: 0.9929
# 用时 93.7184476852417
