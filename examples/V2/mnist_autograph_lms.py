# Copyright 2019, 2020. IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow_large_model_support import LMS

num_classes = 1000 # 10
epochs = 1
use_lms = True

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_train_samples = x_train.shape[0]
num_test_samples = x_test.shape[0]
batch_size = num_train_samples # 64
steps_per_epoch = num_train_samples # num_train_samples / batch_size
evaluation_steps = num_test_samples # num_test_samples / batch_size
x_train = x_train.reshape(num_train_samples, 784)
x_test = x_test.reshape(num_test_samples, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# convert train adn test data to tf.tensor
x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)

# define model
model = Sequential()
model.add(Dense(5120, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dense(5120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
              metrics=['accuracy'])

# execute the model with/without TFLMS
lms=None
model_fit = tf.autograph.to_graph(model.fit)
callbacks = []
if use_lms:
  lms = LMS(swapout_threshold=1, swapin_ahead=1, swapin_groupby=0)
  lms.batch_size = batch_size
  callbacks.append(lms)

history = model_fit(model, x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1,
                    validation_data=(x_test, y_test),
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=evaluation_steps,
                    callbacks=callbacks)
score = model.evaluate(x_test, y_test, verbose=0, steps=evaluation_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
