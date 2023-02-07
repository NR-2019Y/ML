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
train_x = train_x[:, None, ...].astype(np.float32) / 255.
test_x = test_x[:, None, ...].astype(np.float32) / 255.
train_x = np.pad(train_x, [[0, 0], [0, 0], [2, 2], [2, 2]])
test_x = np.pad(test_x, [[0, 0], [0, 0], [2, 2], [2, 2]])
train_y = train_y.astype(np.int64)
test_y = test_y.astype(np.int64)

train_iter = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_iter = train_iter.shuffle(buffer_size=4096).batch(batch_size=128)
test_iter = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size=1000)

model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', data_format='channels_first'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'),
    layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', data_format='channels_first'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'),
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
model.fit(train_iter, validation_data=test_iter, epochs=10)
print("用时", time.time() - tic)
# 469/469 [==============================] - 14s 29ms/step - loss: 0.1402 - sparse_categorical_accuracy: 0.9574 - val_loss: 0.0391 - val_sparse_categorical_accuracy: 0.9869
# Epoch 2/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0379 - sparse_categorical_accuracy: 0.9886 - val_loss: 0.0288 - val_sparse_categorical_accuracy: 0.9902
# Epoch 3/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0262 - sparse_categorical_accuracy: 0.9919 - val_loss: 0.0301 - val_sparse_categorical_accuracy: 0.9901
# Epoch 4/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0187 - sparse_categorical_accuracy: 0.9942 - val_loss: 0.0273 - val_sparse_categorical_accuracy: 0.9916
# Epoch 5/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0140 - sparse_categorical_accuracy: 0.9956 - val_loss: 0.0291 - val_sparse_categorical_accuracy: 0.9909
# Epoch 6/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0114 - sparse_categorical_accuracy: 0.9962 - val_loss: 0.0296 - val_sparse_categorical_accuracy: 0.9912
# Epoch 7/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0112 - sparse_categorical_accuracy: 0.9960 - val_loss: 0.0288 - val_sparse_categorical_accuracy: 0.9916
# Epoch 8/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0100 - sparse_categorical_accuracy: 0.9969 - val_loss: 0.0289 - val_sparse_categorical_accuracy: 0.9919
# Epoch 9/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0075 - sparse_categorical_accuracy: 0.9974 - val_loss: 0.0409 - val_sparse_categorical_accuracy: 0.9886
# Epoch 10/10
# 469/469 [==============================] - 8s 18ms/step - loss: 0.0074 - sparse_categorical_accuracy: 0.9977 - val_loss: 0.0324 - val_sparse_categorical_accuracy: 0.9918
# 用时 96.35957312583923
