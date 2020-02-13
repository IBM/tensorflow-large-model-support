# TensorFlow Large Model Support
TensorFlow Large Model Support (TFLMS) is a feature in the TensorFlow provided
by [IBM Watson Machine Learning Community Edition](https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/) (WML CE) that allows the
successful training of deep learning models that would otherwise exhaust GPU
memory and abort with "out-of-memory" errors. LMS manages this
oversubscription of GPU memory by temporarily swapping tensors to host memory
when they are not needed.

One or more elements of a deep learning model can lead to GPU memory exhaustion.

These include:

 * Model depth and complexity
 * Base data size (for example, high-resolution images)
 * Batch size

Traditionally, the solution to this problem has been to modify the model until
it fits in GPU memory. This approach, however, can negatively impact
accuracy – especially if concessions are made by reducing data
fidelity or model complexity.

With LMS, deep learning models can scale significantly beyond what was
previously possible and, ultimately, generate more accurate results.

# Installing TensorFlow Large Model Support

TFLMS is built into the `tensorflow-gpu` conda package so it is installed by
default when you install the GPU enabled TensorFlow from WML CE.
The support is currently available in the [WML CE Early Access conda channel](https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/).
For more information on this channel, how to add channels, and install
frameworks see [this post](https://developer.ibm.com/linuxonpower/2020/02/10/using-the-watson-machine-learning-community-edition-auxiliary-channels/).

# How to enable TFLMS

The TFLMS functionality is disabled by default in TensorFlow and needs to be
enabled before your model creates tensors. In most cases, enabling TFLMS is
as simple as calling the enablement API at the start of your program:

```python
import tensorflow as tf
tf.config.experimental.set_lms_enabled(True)
```

In TensorFlow 2 some models use sessions and session configurations are
created either explicitly in model code or implicitly within TensorFlow APIs.

## Using TensorFlow Estimators
TensorFlow Estimators use sessions for training and will implicitly create
a default session configuration if one is not specified. To enable TFLMS
the ConfigProto settings need to be updated with the LMS setting.

```python
# Create a session config if necessary, or add to the existing session config
session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.experimental.lms_enabled = True

# Create a run config if necessary, or add the session_config to the existing
# run config.
run_config = tf.estimator.RunConfig(# ... other RunConfig parameters,
                                    session_config=session_config)
# Pass the RunConfig to the Estimator
estimator = tf.estimator.Estimator( # .. other Estimator parameters,
                                   config=run_config)
```

## TensorFlow Keras directly setting session
If a TensorFlow Keras model is used in with v1 compatibility mode	in
TensorFlow 2, and TensorFlow 2 behavior is disabled using:
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
then the Session configuration must be set to enable LMS.

```python
session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.experimental.lms_enabled = True
sess = tf.Session(config=session_config)
tf.keras.backend.set_session(sess)
```

### TensorFlow 1.x models using Sessions
If a standard sessions-based TensorFlow 1.x model is used with v1
compatibility mode in TensorFlow 2 using:
```python
import tensorflow.compat.v1 as tf
```
then the Session configuration must be set to enable LMS.

```python
session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.experimental.lms_enabled = True
sess = tf.Session(config=session_config)
```

# Examples
The ManyModel.py example, found in the [TensorFlow LMS examples](examples/),
uses synthetic random images with multiple models provided by
TensorFlow Keras applications to allow users a fast hands-on experience with
LMS. The example allows users to change the image size, explore auto-tuning,
and manually set the LMS tunable parameters on many variants of the
ResNet, DenseNet, MobileNet, Inception, NASNet, and Xception models.
Advanced users can also use the command line parameters to enable CUDA
profiling that can be used with the NVIDIA Visual Profiler to profile
and visualize the tensor swapping.

# Usage tips

## Increase the system memory (GPU host) memory allocation
TensorFlow sets a limit on the amount of memory that will be allocated on the
GPU host (CPU) side. The limit is often not high enough to act as a tensor swap
space when swapping a large amount of data or when using multiple GPUs without
the use of Horovod. The limit can be adjusted by setting the
`TF_GPU_HOST_MEM_LIMIT_IN_MB` environment variable. Failure to set this limit
higher will result in out of memory errors such as: Allocator (gpu_host_bfc)
ran out of memory trying to allocate. Note the gpu_host_bfc allocator is
mentioned rather than a GPU allocator.

The value for `TF_GPU_HOST_MEM_LIMIT_IN_MB` should be several times the size
of the memory of the GPUs being used by the TensorFlow process. For example,
if a single 32GB GPU is being used then the `TF_GPU_HOST_MEM_LIMIT_IN_MB`
should be set several times greater than 32GB.

If Horovod distribution is being used, it will create one process per GPU. In
this the `TF_GPU_HOST_MEM_LIMIT_IN_MB` limit should be set several times greater
than the memory of one of the GPUs.

If other GPU distrubtion mechanisms are used, then the
`TF_GPU_HOST_MEM_LIMIT_IN_MB` limit should be set to several times the sum of
the memory of all the GPUs being used.

## Use NUMA pinning for single GPU use
If you are utilizing a single GPU it is recommended to use NUMA pinning to pin
the process to the CPU and memory that is on the same system socket as the
GPU being used. Pinning the process allows the fastest connection paths between
system memory and GPU memory, which reduces the training or inferencing time.
WML CE includes the numactl utility that can be used to do this pinning. It
can be installed with the `conda install numactl` command. The following
example shows how to specify a single GPU to be used and how to pin the
process to use the CPU cores and memory that are on the same socket
as the specified GPU:

```sh
export CUDA_VISIBLE_DEVICES=0
numactl --cpunodebind=0 --membind=0 python train.py
```

## Use Horovod when using more than one GPU
It is recommended to use Horovod distribution when using more than one GPU
because Horovod creates a separate process per GPU and automatically sets the
process have socket affinity with the GPU which allows the fastest
connection paths between system memory and GPU memory, which reduces the
training or inferencing time.

# Memory defragmentation
When using very large tensors or during the course of a very long training
operation, the model's memory allocation and usage pattern may lead to
fragmented GPU memory and out of memory errors. When this occurs there is
enough free memory in the GPU for the next allocation, but it is in
non-contiguous blocks. In these cases, the process will fail and output a
message like this:

```
Enough free memory to satisfy the allocation request exists but it is fragmented.
Enabling Large Model Support defragmentation may avoid this failure.
```

TFLMS is capable of defragmenting sections of GPU memory to gather a
contiguous block large enough for the request. This feature waits for current
GPU computation to finish and then relocates active tensors to coalesce
contiguous free memory blocks.

Even with the GPU computation cleared, the moving of active tensors carries
a risk of introducing NaN errors or other instability into the model. Despite
this risk it has performed well in multi-week training runs with very large
tensors and defragmentation called frequently.

Due to the possible risk of instability the Large Model Support defragmentation
is disabled by default and can be enabled along with LMS with this the `tf.config.experimental.set_lms_defrag_enabled(True)` API or the  
`config.gpu_options.experimental.lms_defrag_enabled=True` ConfigProto setting.

# Model memory usage analysis with allocator statistics
TFLMS adds several APIs to obtain GPU memory allocator statistics such as
the number of allocations, the peak memory usage, the amount
of memory swapped, and more. For more information on the statistics APIs
and examples of their usage see the [TensorFlow LMS examples](examples/).

# Building TensorFlow from source with TensorFlow Large Model Support
The [patches](patches/) directory contains git patch of for the TFLMS code.
The file names correspond to tag levels in the
[TensorFlow source](https://github.com/tensorflow/tensorflow/). To build
TensorFlow from source with TensorFlow Large Model Support, check out the
specific TensorFlow git tag and then apply the corresponding TensorFlow Large
Model Support patch file.

For example:
```sh
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git pull --tags
git checkout v2.1.0
git apply /tensorflow-large-model-support/patches/tensorflow_v2.1.0_large_model_support.patch
```

# Contribution guidelines

If you want to contribute to TensorFlow Large Model Support please read the
[contribution guidelines](CONTRIBUTING.md).

# Previous implementations of TensorFlow Large Model Support
## TFLMSv1
The TFLMSv1 implementation was installed as a separate module from TensorFlow
and performed static graph modifications on the model's graph to introduce
swapping nodes. This implementation was included in the tensorflow.contrib
module path as a technology preview in IBM PowerAI 1.5.4 and earlier releases.
The implementation source resides in the [tflmsv1](https://github.com/IBM/tensorflow-large-model-support/tree/tflmsv1) branch of this repository.
## TFLMSv2
The TFLMSv2 implementation was installed as a separate conda module from
TensorFlow and performed static graph modifications on the model's graph to
introduce swapping nodes and other graph optimizations. This implementation
was included in IBM Watson Machine Learning Community Edition 1.6.x versions.
The implementation of this version is not open source.
