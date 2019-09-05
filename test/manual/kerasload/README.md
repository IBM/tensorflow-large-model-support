# Keras save / load testing

## Background
Keras' save model functionality does not save the nodes that TFLMS adds to
the graph. It saves layer definitions. The save model functionality also
saves the weights, the optimizer weights, and the optimizer configuration/state.

During load, the weights are set back on the instantiated model using a
Session. This finalizes the model and prevents TFLMS from adding nodes after
the load. An API change was added to TensorFlow Keras in PowerAI to allow
the load_model method to take a list of Keras callbacks. This same change
has been proposed upstream. The change calls the set_model method on the
callbacks after the load routine instantiates the model but before the
weights and optimizer weights and settings are applied.

In addition to the save model method, Keras provides a method to save just
the weights and can save them in two different formats: the TensorFlow
checkpoint format or the Keras .h5 format.

This has ramifications to Keras save and load and user scripts that are
saving and loading models or weights.

When these save and load options are combined with the various use cases it
provides many paths to test for LMS. For instance:

train > save > load_model > resume training
train > save > inference / evaluate
train > save weights > instantiate model with code > inference / evaluate
train > save weights > instantiate model with code > resume training

In practice we have seen that the loading of the weights will freeze the graph
and prevent LMS modifications from taking effect. With low amounts of swapping
it has been observed that loading the weights does not freeze all of the
graph from modification. For this reason the test case specify the
maximum amount of swapping.

## Description of test files

The Keras_ResNet50 files provide a means to test the fit_generator path but
are not being actively maintained. The more exhaustive test cases are provided
by the mnist model.

The tests consist of two parts. First train and save followed by loading the
saved file and doing either more training or inference.

Here the save - load test combinations that should be run:

| train-save                                  | load-other action               |
|---------------------------------------------|---------------------------------|
| mnist_cnn_keras_trainsave.py                |  mnist_cnn_keras_loadtrain.py   |
| mnist_cnn_keras_trainsave.py                |  mnist_cnn_keras_loadpredict.py |
| mnist_cnn_keras_trainsaveweights.py         |  mnist_cnn_keras_loadweights_train.py |
| mnist_cnn_keras_trainsaveweights.py         |  mnist_cnn_keras_loadweights_predict.py |
| mnist_cnn_keras_trainsaveweights_h5.py      |  mnist_cnn_keras_loadweights_h5_train.py |
| mnist_cnn_keras_trainsaveweights_h5.py      |  mnist_cnn_keras_loadweights_h5_predict.py |
| mnist_cnn_keras_trainsave_no_optimizer.py   |  mnist_cnn_keras_loadpredict.py |
| mnist_cnn_keras_train_save_checkpoint.py    | mnist_cnn_keras_load_checkpoint_train.py |
| mnist_cnn_keras_train_save_checkpoint.py    | mnist_cnn_keras_load_checkpoint_predict.py |

The `run_save_load.sh` script runs the 9 combinations in order and
run the 7 tests with both TensorFlow Keras and Keras team Keras.

Checking for error conditions:
The .py files should run without error. Moreover, the output of the load* files
should show LMS modifying the graph. The output of the load* files should NOT
contain warning/error messages about graph modifications not taking effect. An
example of such a warning message is: `W tensorflow/c/c_api.cc:769] Operation '{name:'loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape' id:205 op device:{} def:{{{node loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32](lms/swapin_dense_1_BiasAdd_0:0, loss/dense_1_loss/softmax_cross_entropy_with_logits/concat)}}' was changed by updating input tensor after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.`

Such an error message means that the LMS modifications will not take effect and some or
no swapping will be done. This is an error condition for LMS and must be addressed.
