# TFLMS: Graph Editing Library for Large Model Support (LMS) in Tensorflow

This library provides an approach to training large models that cannot be fit into GPU memory.
It takes a computational graph defined by users, and automatically adds swap-in and swap-out nodes for transferring tensors from GPUs to the host and vice versa.
The computational graph is statically modified. Hence, it needs to be done before a session actually starts.

## Installation
```
python setup.py install
```

## How to use
### Step 1: define scopes for optimizer and the first layer
TFLMS needs to know some information about user-defined models.
There are two must-have parameters: a scope for the optimizer and a scope for the first layer in the model.

User should define them as follows:
- For optimizer
```python
with tf.name_scope('adam_optimizer'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```
- For the first layer in the model
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
	first_layer='conv1')
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
	first_layer='conv1')
lms_obj.run()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	batch = mnist.train.next_batch(50)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

### Parameters for LMS
#### Required parameters
_graph_ :: the graph we want to modify for LMS.

_optimizer_scope_ :: the scope for the optimizer.

_first_layer_ :: the first layer of the model.

#### Optional parameters
_excl_scopes_ :: a set of scopes for tensors we don't want to swap them out to the host.

_lb_ :: Lowerbound value for LMS. A tensor will be swapped in during the backward phase at least `lb` nodes before it in the graph. Default `1`.

_ub_ :: Upperbound value for LMS. Default `10000`.

_n_tensors_ :: The number of tensors for LMS, counting from the beginning of the model. "0" means taking all tensors in the graph for LMS. Default `0`.

_ssg_n_tensors_ :: The number of tensors that will be placed on a second storage, counting from the beginning of the model. "0" means "not use" the second storage. Default `0`.

_ssg_id_ :: The GPU device ID that will be used as a second storage for LMS. Default `1`.

_ssg_as_buffer :: Use the second storage just as a buffer. Data are then forwarded to the host. Default `False`.

_debug_ :: Debug mode for LMS. Default `False`.

_debug_level_ :: Debug level for LMS (1 or 2). Default `1`.
