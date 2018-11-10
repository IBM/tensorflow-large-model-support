# TFLMS: Graph Editing Library for Large Model Support (LMS) in TensorFlow

This library provides an approach to training large models that cannot be fit into GPU memory.
It takes a computational graph defined by users, and automatically adds swap-in and swap-out nodes for transferring tensors from GPUs to the host and vice versa.
The computational graph is statically modified. Hence, it needs to be done before a session actually starts.

## How to use
Enabling LMS for a model depends on how users write their training. The
following guidelines cover three ways to train:
- [Session](https://www.tensorflow.org/programmers_guide/graphs)-based training
- [Estimator](https://www.tensorflow.org/programmers_guide/estimators)-based training
- [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)-based training

### [Session](https://www.tensorflow.org/programmers_guide/graphs)-based training
```python
from tensorflow_large_model_support import LMS
lms_obj = LMS()
lms_obj.run(tf.get_default_graph())
```
The above lines must be put before starting a training session, for example:
```python
# Import and run the graph modification before running a session:
from tensorflow_large_model_support import LMS
lms_obj = LMS()
lms_obj.run(tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```
For a working example of LMS integration with Session based training see:
`examples/mnist_deep_lms.py`
which is an LMS enabled version of `https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py`.

### [Estimator](https://www.tensorflow.org/programmers_guide/estimators)-based training
#### Step 1: Import and initialize LMS
```python
from tensorflow_large_model_support import LMS
lms_hook = LMS()
```
#### Step 2: Add the LMS object into Estimator's hook list
```python
mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000
      hooks=[logging_hook, lms_hook])
```

For a working example of LMS integration with Estimator based training see:
`examples/cnn_mnist_lms.py`
which is an LMS enabled version of `https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py`.

### [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)-based training
#### Step 1: Import and initialize LMS
```python
from tensorflow_large_model_support import LMS
lms_callback = LMS()
```
#### Step 2: Add the LMS object to the callback list on the Keras
`fit` or `fit_generator` function.
```python
model.fit_generator(generator=training_gen, callbacks=[lms_callback])
```

For a working example of LMS integration with [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)-based training see:
`examples/mnist_cnn_keras.py`
which is an LMS enabled version of `https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py`.


### Parameters for LMS
_swapout_threshold_: the smaller `swapout_threshold` is, the more tensors are swapped out to the host memory. Default `-1` (auto mode).

_swapin_ahead_: the larger `swapin_ahead` is, the earlier a tensor is swapped in to the GPU memory from the host memory. Default `-1` (auto mode).

_swapin_groupby_: multiple swap-in operations of the same tensor will be grouped or fused into one swap-in operation for better performance if they are *close* to each other (the distance betweem them is within `swapin_groupby`). Default `-1` (auto mode).

_sync_mode_: whether do synchronization between data transfer and kernel computation or not. Four modes: `0` turn off. `1` sync for only swap-out operations. `2` sync for only swap-in operations. `3` sync for both swap-out and swap-in operations. Default `0`.

_serialization_: serialize operations at the same level in the topological sort. This option accepts a list of Python slicing string in which each slicing represents level indices in the topological sort. E.g. [1, 3:5, 7] means levels 1, 3, 4, 5 and 7 are serialized. Default `[]` (turn off).

_debug_ :: Debug mode for LMS. Default `False`.

_debug_level_ :: Debug level for LMS (1 or 2). Default `1`.

### AutoTune
If parameters `swapout_threshold`, `swapin_ahead`, `swapin_groupby` are set to the default values, we will enable AutoTune to automatically find suitable values for them. However, if AutoTune does not have enough information to do auto-tuning, such as a lack of mini-batch size (since users use a Placeholder to feed data), it would raise an error and users should provide the mini-batch size for it as follows:

```python
from tensorflow_large_model_support import LMS
lms_callback = LMS()
lms.batch_size = 32
lms.run(tf.get_default_graph())
```

### Performance Tuning LMS
(To be added)

### TensorFlow and LMS

TensorFlow version >= 1.8 has a mechanism for memory optimization. Though the
mechanism totally works well with this LMS module, it is recommended to switch
its mode to `SCHEDULING_HEURISTICS` to allow training as large a model as
possible. This can be done via the following snippet code:
```python
config = tf.ConfigProto()
config.graph_options.rewrite_options.memory_optimization = \
	rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
```
