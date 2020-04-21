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
lms_obj.run()
```
The above lines must be put before starting a training session, for example:
```python
# Import and run the graph modification before running a session:
from tensorflow_large_model_support import LMS
lms_obj = LMS()
lms_obj.run()

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

An working example of LMS integration with Keras based training is:
`examples/Keras_ResNet50.py`.


### Parameters for the LMS constructor
_swapout_threshold_: the smaller `swapout_threshold` is, the more tensors are swapped out to the host memory. Default `-1` (auto mode).

_swapin_ahead_: the larger `swapin_ahead` is, the earlier a tensor is swapped in to the GPU memory from the host memory. Default `-1` (auto mode).

_swapin_groupby_: multiple swap-in operations of the same tensor will be grouped or fused into one swap-in operation for better performance if they are *close* to each other (the distance betweem them is within `swapin_groupby`). Default `-1` (auto mode).

_sync_mode_: whether to do synchronization between data transfer and kernel computation or not. Four modes: `0` turn off. `1` sync for only swap-out operations. `2` sync for only swap-in operations. `3` sync for both swap-out and swap-in operations. Default `0`.

_serialization_: serialize operations at the same level in the topological sort. This option accepts a list of Python slicing string in which each slicing represents level indices in the topological sort. E.g. [1, '3:5', 7] means levels 1, 3, 4, 5 and 7 are serialized. Default `[]` (turn off).

_serialization_by_size_: serialize operations in levels of the topological sort, if the cumulative memory consumption of the level is greater than `serialization_by_size` GiB. Default `0` (turn off).

_debug_ :: Debug mode for LMS. Default `False`.

_debug_level_ :: Debug level for LMS (1 or 2). Default `1`.

### Parameters for the method `run`
_graph_: a computational graph that will be modified by LMS. Default `tensorflow.get_default_graph()`.
_keras_: whether the computational graph is a Keras model or not. Default `False`.

### Auto tuning
If parameters `swapout_threshold`, `swapin_ahead`, `swapin_groupby` are set to
the default values, we will enable auto tuning to automatically find suitable
values for them. Auto tuning requires the mini batch size to correctly
calculate memory usage. Some models and some methods of invoking the model
training do not allow LMS to know the batch size. When auto tuning cannot
discover the batch size it will raise an error and the batch size should be
specified manually as follows:

```python
from tensorflow_large_model_support import LMS
lms = LMS()
lms.batch_size = 32
lms.run()
```

### Performance Tuning LMS
(To be added)

### Example of TensorFlow Large Model Support with Large Data and Tuning

The Keras model example `examples/Keras_ResNet50.py` allows the user to
increase the input data size to cause out of memory situations and then
easily experiment with TFLMS tuning options to train with larger data.
See the comment header in the example for more information.


### Using LMS with saved models
Both TensorFlow and Keras have various ways to save models. Some of these
methods save the model or graph definition and some methods save only the
weights. Whether you need to enable large model support on the loaded model
depends on several factors: if you are loading the model for further training
or loading the model for further inferencing, as well as how the model was
saved.

If TensorFlow MetaGraphs or SavedModels are saved after LMS has added swapping
nodes to the model, the loaded model will contain swapping nodes. If only the
model weights are saved, and are restored onto a model that is built using
code, then the model will only have LMS swapping nodes if LMS is re-run on the
model.

Keras models saved with `tf.keras.models.save_model` do not have LMS swapping
nodes in them. If swapping is required in the loaded model, LMS should be
passed to the load `tf.keras.models.load_model`
API. For example:
```python
from tensorflow_large_model_support import LMS
lms_callback = LMS()
model = tf.keras.models.load_model(filepath, callbacks=[lms_callback])
```

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

## Citations

### Scientific papers:

Tung D. Le, Haruki Imai, Yasushi Negishi, and Kiyokuni Kawachiya. 2019. [Automatic GPU memory management for large neural models in TensorFlow](https://dl.acm.org/doi/abs/10.1145/3315573.3329984). In Proceedings of the 2019 ACM SIGPLAN International Symposium on Memory Management (ISMM 2019), 1–13.
