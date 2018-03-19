# TFLMS: Graph Editing Library for Large Model Support (LMS) in Tensorflow

This library provides an approach to training large models that cannot be fit into GPU memory.
It takes a computational graph defined by users, and automatically adds swap-in and swap-out nodes for transferring tensors from GPUs to the host and vice versa.
The computational graph is statically modified. Hence, it needs to be done before a session actually starts.

## Installation
```
python setup.py install
```

## How to use
### Step 1: define optimizer/solver scope and starting scope
TFLMS needs to know some information about user-defined models.
There are two must-have parameters:
- a scope for the optimizer/solver
- a scope from which TFLMS starts to discover candidates (tensors) for LMS.

User should define them as follows:
- For optimizer/solver
```python
with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```
- For the starting scope
```python
with tf.name_scope('conv1'):
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
```

### Step 2: define a LMS object and run it
Define a LMS object for the graph we want to edit and run it to actually modify the graph.
```python
from lms import LMS
lms_obj = LMS(graph=tf.get_default_graph(),
	optimizer_scope='adam_optimizer',
	starting_scope='conv1')
lms_obj.run()
```
The above lines must be put before starting a traning session, for example:
- before
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```
- after inserting LMS code
```python
from lms import LMS
lms_obj = LMS(graph=tf.get_default_graph(),
	optimizer_scope='adam_optimizer',
	starting_scope='conv1')
lms_obj.run()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

### Parameters for LMS
#### Required parameters
_graph_ :: the graph we will modify for LMS. This should be the graph of user-defined neural network.

_optimizer_scope_ :: the scope for the optimizer.

_starting_scope_ :: Tensors that are reachable from the operations in this scope will be swapped for LMS. Set this to the scope of the first layer if we would like to modify the whole graph.

#### Optional parameters
_excl_scopes_ :: a set of scopes for operations whose tensors will not be swapped out to the host.

_incl_scopes_ :: a set of scopes for operations whose tensors will be swapped out to the host.

_excl_types_ :: a set of types for operations whose tensors will not be swapped out to the host.

_incl_types_ :: a set of types for operations whose tensors will be swapped out to the host.

_lb_ :: Lowerbound value for LMS. A tensor will be swapped in during the backward phase at least `lb` nodes before it in the graph. Default `1`.

_ub_ :: Upperbound value for LMS. Default `10000`.

_n_tensors_ :: The number of tensors for LMS, counting from the `starting_scope`. To turn off LMS, set `n_tensors` to `0`. If not set, all reachable tensors will be swapped for LMS (Default).

_debug_ :: Debug mode for LMS. Default `False`.

_debug_level_ :: Debug level for LMS (1 or 2). Default `1`.

#### Optional/Experimental parameters
_ssg_n_tensors_ :: The number of tensors that will be placed on a second storage, counting from the `starting_scope`. to turn off SSG, set `ssg_n_tensors` to `0` (Default).

_ssg_id_ :: The GPU device ID that will be used as a second storage for LMS. This is only effective if `ssg_n_tensors` is not `0`. Default `1`.

_ssg_as_buffer :: Use the second storage just as a buffer. Data are finally forwarded to the host via the second storage. Default `False`.

_fuse_swapins_ :: Fuse "close" swap-in operations into one. This may improve the performance. Default `False`
