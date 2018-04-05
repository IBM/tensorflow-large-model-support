# TFLMS: Graph Editing Library for Large Model Support (LMS) in Tensorflow

This library provides an approach to training large models that cannot be fit into GPU memory.
It takes a computational graph defined by users, and automatically adds swap-in and swap-out nodes for transferring tensors from GPUs to the host and vice versa.
The computational graph is statically modified. Hence, it needs to be done before a session actually starts.

## Installation
```
python setup.py install
```

## How to use
TFLMS needs to know some information about user-defined models.
There are two requirements for a user-defined model:
- it must have a placeholder operation for the model's input, or a scope for the first layer (or any layer from which users want to start swapping tensors).
- it must have scopes for the optimizers/solvers.

Enabling LMS for a model depends on how users write their training. Followings are guidelines for two ways: [Session](https://www.tensorflow.org/programmers_guide/graphs)-based training and [Estimator](https://www.tensorflow.org/programmers_guide/estimators)-based training.
### [Session](https://www.tensorflow.org/programmers_guide/graphs)-based training
Assume that the user-defined model has a placeholder for input data
```python
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
```
#### Step 1: define optimizer/solver scopes
```python
with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```
#### Step 2: define an LMS object and run it
```python
from lms import LMS
lms_obj = LMS(optimizer_scopes={'adam_optimizer'})
lms_obj.run(graph=tf.get_default_graph())
```
The above lines must be put before starting a traning session, for example:
- Before inserting LMS code
```python
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
# other lines for creating the mode are omitted
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```
- After inserting LMS code
```python
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
# other lines for creating the mode are omitted

from lms import LMS
lms_obj = LMS(optimizer_scopes={'adam_optimizer'})
lms_obj.run(graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```
For more information, see [mnist_deep_lms.py](examples/mnist_deep_lms.py).
### [Estimator](https://www.tensorflow.org/programmers_guide/estimators)-based training
Assume that the user-defined model DOESN'T HAVE a placeholder for input data. In this case, users need to define a scope for the first layer (or any layer from which users want to start swapping tensors):
```python
with tf.name_scope('conv1'):
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
```
#### Step 1: define optimizer/solver scopes
```python
with tf.name_scope('adam_optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
        loss=loss,
	global_step=tf.train.get_global_step())
```
#### Step 2: define an LMSHook (LMSHook and LMS share the same set of parameters)
```python
# Hook for Large Model Support
from lms import LMSHook
lms_hook = LMSHook(optimizer_scopes={'adam_optimizer'},
	starting_scope='conv1')
```
#### Step 3: add the LMSHook into Estimator's hook list
```python
mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000
      hooks=[logging_hook, lms_hook])
```
For more information, see [cnn_mnist_lms.py](examples/cnn_mnist_lms.py).
### Parameters for LMS/LMSHook
#### Required parameters
_graph_ :: the graph we will modify for LMS. This should be the graph of user-defined neural network. (not required in LMSHook)

_optimizer_scopes_ :: scopes for the optimizers/solvers.

#### Optional parameters
_starting_scope_ :: Tensors that are reachable from the operations in this scope will be swapped for LMS. Set this to the scope of the first layer if we would like to modify the whole graph. Default `None`.

_excl_scopes_ :: a set of scopes for operations whose tensors will not be swapped out to the host. Default `empty`.

_incl_scopes_ :: a set of scopes for operations whose tensors will be swapped out to the host. Default `empty`.

_excl_types_ :: a set of types for operations whose tensors will not be swapped out to the host. Default `empty`.

_incl_types_ :: a set of types for operations whose tensors will be swapped out to the host. Default `empty`.

_n_tensors_ :: The number of tensors for LMS, counting from the `starting_scope`. To turn off LMS, set `n_tensors` to `0`. Default `-1` (all reachable tensors will be swapped for LMS).

_lb_ :: Lowerbound value for LMS. A tensor will be swapped in during the backward phase at least `lb` nodes before it in the graph. Default `1`.

_ub_ :: Upperbound value for LMS. Default `10000`.

_fuse_swapins_ :: Fuse "close" swap-in operations into one operation. This may improve the performance. Default `False`.

_debug_ :: Debug mode for LMS. Default `False`.

_debug_level_ :: Debug level for LMS (1 or 2). Default `1`.
