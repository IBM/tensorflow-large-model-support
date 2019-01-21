# TensorFlow Large Model Support: Graph Editing Library for Large Model Support (LMS) in TensorFlow

This library provides an approach to training large models that cannot be fit into GPU memory.
It takes a computational graph defined by users, and automatically adds swap-in and swap-out nodes for transferring tensors from GPUs to the host and vice versa.
The computational graph is statically modified. Hence, it needs to be done before a session actually starts.

## Install
TensorFlow Large Model Support can be installed as a pip module named
`tensoflow-large-model-support`. To install
TensorFlow Large Model Support, run the following commands to clone this repository
and install the module:
```sh
git clone https://github.com/IBM/tensorflow-large-model-support.git
pip install ./tensorflow-large-model-support
```

## How to use
TensorFlow Large Model Support needs to know some information about user-defined models.
There is one requirement for a user-defined model: it must have scopes for the optimizers/solvers.

Enabling LMS for a model depends on how users write their training. The
following guidelines cover three ways to train:
- [Session](https://www.tensorflow.org/programmers_guide/graphs)-based training
- [Estimator](https://www.tensorflow.org/programmers_guide/estimators)-based training
- [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)-based training

### [Session](https://www.tensorflow.org/programmers_guide/graphs)-based training
#### Step 1: define optimizer/solver scopes
```python
with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```
#### Step 2: define an LMS object and run it
```python
from tensorflow_large_model_support import LMS
lms_obj = LMS({'adam_optimizer'})
lms_obj.run(graph=tf.get_default_graph())
```
The above lines must be put before starting a training session, for example:
- Before inserting LMS code
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```
- After inserting LMS code
```python
from tensorflow_large_model_support import LMS
lms_obj = LMS({'adam_optimizer'})
lms_obj.run(graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```
For a working example of LMS integration with Session based training see:
`examples/mnist_deep_lms.py`
which is an LMS enabled version of `https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py`.

### [Estimator](https://www.tensorflow.org/programmers_guide/estimators)-based training
#### Step 1: define optimizer/solver scopes
```python
with tf.name_scope('adam_optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
        loss=loss,
	global_step=tf.train.get_global_step())
```
#### Step 2: define a LMSSessionRunHook
```python
# Hook for Large Model Support
from tensorflow_large_model_support import LMSSessionRunHook
# LMSSessionRunHook and LMS share the same set of parameters. Here we just
# use the default keyword arguments.
lms_hook = LMSSessionRunHook({'adam_optimizer'})
```
#### Step 3: add the LMSSessionRunHook into the Estimator's hook list
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

**NOTE:** The use of the LMSKerasCallback requires TensorFlow 1.12 or later.

#### Step 1: Define a LMSKerasCallback.
```python
from tensorflow_large_model_support import LMSKerasCallback
# LMSKerasCallback and LMS share a set of keyword arguments. Here we just
# use the default options.
lms_callback = LMSKerasCallback()
```
#### Step 2: pass the callback to the Keras `fit` or `fit_generator` function.
```python
model.fit_generator(generator=training_gen, callbacks=[lms_callback])
```
For a working example of LMS integration with Keras based training see:
`examples/Keras_ResNet50.py`.

### Scaling tips

If scaling to multiple GPUs is achieved by building the model
in a multi-tower fashion with a tower for each GPU as described in the [TensorFlow documentation](https://www.tensorflow.org/guide/using_gpu),
you must also set and tune the `TF_CUDA_HOST_MEM_LIMIT_IN_MB` environment variable.

TensorFlow sets a limit on the amount of memory that will be allocated on
the CUDA host (CPU) side. The limit is often not high enough to act as a tensor
swap space for multiple GPUs. Failure to set this limit higher will result
in out of memory errors like this: `Allocator (cuda_host_bfc) ran out of memory trying to allocate`.
Note the `cuda_host_bfc` allocator is mentioned rather than a GPU allocator.

A good rule of thumb would be to start with a value that is 4 times the memory
capacity of the GPUs times the number of GPUs that will be used.  For example,
if you have four 16 GB GPUs in a system and will use all four in a training run,
`TF_CUDA_HOST_MEM_LIMIT_IN_MB` should be set to 262144 and adjust from there
as needed. (4 x 16384 (16GB as MB) x 4 GPUs) = 262144 MB.


### Parameters for Parameters for LMS/LMSSessionRunHook/LMSKerasCallback
#### Required parameters
_graph_ :: the graph we will modify for LMS. This should be the graph of user-defined neural network. (not required in LMSSessionRunHook or LMSKerasCallback)

_optimizer_scopes_ :: scopes for the optimizers/solvers. (Not required in LMSKerasCallback)

#### Optional parameters
_starting_scope_ :: Tensors that are reachable from the operations in this scope will be swapped for LMS. Set this to the scope of the first layer if we would like to modify the whole graph. Default `None`.

_starting_op_names_ :: Tensors that are reachable from the operations with these names will be swapped for LMS. Default `None`.

_excl_scopes_ :: a set of scopes for operations whose tensors will not be swapped out to the host. Default `empty`.

_incl_scopes_ :: a set of scopes for operations whose tensors will be swapped out to the host. Default `empty`.

_excl_types_ :: a set of types for operations whose tensors will not be swapped out to the host. Default `empty`.

_incl_types_ :: a set of types for operations whose tensors will be swapped out to the host. Default `empty`.

_n_tensors_ :: The number of tensors for LMS, counting from the `starting_scope`. To turn off LMS, set `n_tensors` to `0`. Default `-1` (all reachable tensors will be swapped for LMS).

_lb_ :: Lowerbound value for LMS. A tensor will be swapped in during the backward phase at least `lb` nodes before it in the graph. Default `1`.

_ub_ :: Upperbound value for LMS. Default `10000`.

_fuse_swapins_ :: Fuse "close" swap-in operations into one operation. This may improve the performance. Default `False`.

_ctrld_strategy_ :: Two strategies to find control dependency ops for	swapin ops: `chain_rule` and `direct_order`. `chain_rule` strategy starts from a forward operation, goes forward and finds a corresponding backward operation to be a control dependency operation. `direct_order` strategy directly gets a backward ops in the topological order to be a control dependency operation. Both strategies depend on `lb` and `ub` to choose a control dependency operation. While the `direct_order` is more exact than `chain_rule` in relation to `lb` and `ub`, it experimentally often results in smaller maximum batch size than `chain_rule`. Default `chain_rule`.

_swap_branches_ :: If True, LMS will swap tensors in branches in the forward phase. Default `False`.

_branch_threshold_ :: If `swap_branches` is enabled and the topological-sort distance between the consuming operation and generating operation of a tensor is greater than `branch_threshold`, then swap the tensor. Default `0`.

_debug_ :: Debug mode for LMS. Default `False`.

_debug_level_ :: Debug level for LMS (1 or 2). Default `1`.


### Performance Tuning LMS

Once you have enabled LMS graph modification in your code you will want to find
the combination of tuning parameters that gives the fastest training time and
best accuracy with your model. The goal of the performance tuning is to swap
out enough tensors to allow your training to run without hitting out of memory
errors, while not swapping too many such that the extra swapping communication
overhead degrades performance.

The two tuning parameters you should focus on are `n_tensors` and `lb`.
Since `n_tensors` controls the number of tensors that will be swapped, the
higher this is set, the lower the peak GPU memory usage will be. The `lb`
controls how soon the tensor is swapped back in before use. A low value of `lb`
can make the training on the GPU pause and wait while the swap in finishes.
This will degrade performance. A higher value of `lb` can allow the tensor
swap in to finish before it's needed and allow training to run without pause.
The downside to swapping in too early is that more tensors will be in GPU
memory at any point in time, resulting in higher peak GPU memory usage.

The tuning thus becomes finding the correct balance between `n_tensors` and `lb`
that provides the best performance for given model.  To start the performance
tuning it's suggested that `n_tensors` be set to -1 which will swap all
reachable tensors. The `lb` should be set to the default 1, which is the latest
possible swap in. If `tf.logging` verbosity is set to `tf.logging.INFO`, LMS
will output a log statement with a count of the number of tensors swapped.
It is useful to run with `n_tensors=-1` for the first run to find this maximum
value and then adjust it downward. If your model has branches like some UNet
models do, you will likely want to set `swap_branches=True` and tune the branch
threshold as well. While tuning the parameters, it is often useful to surface
the LMS parameters in the training script as command line parameters or
configuration file properties.

By default LMS will analyze your graph to find the starting operations to use
for finding tensor swap candidates. You can bypass this analysis by placing your
starting operations in a named scope and providing the scope on the
`starting_scope` parameter, or by providing the names of the starting operations
on the `starting_op_names` parameter. This can speed up repeated runs of LMS
during tuning. Furthermore, you can enable `debug=True` and `debug_level=1`
and LMS will print out the name and type of the starting operations it
finds. These names could be passed in on the `starting_op_names` parameter on
subsequent runs.

It is recommended that you start with tuning training on a single GPU before
enabling your code for multi-GPU with DDL.

### TensorFlow Grappler and TensorFlow Large Model Support

TensorFlow has a mechanism for memory optimization. Though the mechanism can
coexist with this module, it is recommended to switch its mode to
`SCHEDULING_HEURISTICS` to allow training as large a model as possible. This
can be done via the following snippet code:
```python
config = tf.ConfigProto()
config.graph_options.rewrite_options.memory_optimization = \
	rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
```

### Example of TensorFlow Large Model Support with Large Data and Tuning

The Keras model example `examples/Keras_ResNet50.py` allows the user to
increase the input data size to cause out of memory situations and then
easily experiment with TFLMS tuning options to train with larger data.
See the comment header in the example for more information.


## Citations

### Scientific papers:

Tung D. Le, Haruki Imai, Yasushi Negishi, Kiyokuni Kawachiya. [TFLMS: Large Model Support in TensorFlow by Graph Rewriting](https://arxiv.org/abs/1807.02037). eprint arXiv:1807.02037. 07/2018.

### Case studies:

Samuel Matzek. TensorFlow Large Model Support Case Study with 3D Image Segmentation. [https://developer.ibm.com/linuxonpower/2018/07/27/tensorflow-large-model-support-case-study-3d-image-segmentation/](https://developer.ibm.com/linuxonpower/2018/07/27/tensorflow-large-model-support-case-study-3d-image-segmentation/). IBM Developer. 07/2018.
