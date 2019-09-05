#!/bin/bash

run_tests() {
  echo "================= Test 1: train, save, load ================================="
  python mnist_cnn_keras_trainsave.py $1
  python mnist_cnn_keras_loadtrain.py $1

  echo "================= Test 2: load last save, predict ================================="
  python mnist_cnn_keras_loadpredict.py $1

  echo "================= Test 3: train, save weights, load, train ================================="
  python mnist_cnn_keras_trainsaveweights.py $1
  python mnist_cnn_keras_loadweights_train.py $1

  echo "================= Test 4: load previous save weights, predict ================================="
  python mnist_cnn_keras_loadweights_predict.py $1

  echo "================= Test 5: train, save weights in h5 format, load, train ================================="
  python mnist_cnn_keras_trainsaveweights_h5.py $1
  python mnist_cnn_keras_loadweights_h5_train.py $1

  echo "================= Test 6: load previous save weights in h5 format, predict ================================="
  python mnist_cnn_keras_loadweights_h5_predict.py $1

  echo "================= Test 7: train, save without saving optimizer state, load, predict ================================="
  python mnist_cnn_keras_trainsave_no_optimizer.py $1
  python mnist_cnn_keras_loadpredict.py $1

  echo "================= Test 8: train, save checkpoint with tf.train.Checkpoint, restore checkpoint, train ================================="
  python mnist_cnn_keras_train_save_checkpoint.py $1
  python mnist_cnn_keras_load_checkpoint_train.py $1

  echo "================= Test 9: load tf.train.Checkpoint, predict ================================="
  python mnist_cnn_keras_load_checkpoint_predict.py $1

}

set -x
echo "Testing TensorFlow Keras"
run_tests tf.keras

echo "Testing Keras Team Keras"
run_tests ktk
