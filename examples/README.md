# TensorFlow Large Model Support Examples

This directory contains examples for using the TensorFlow
Large Model Support (TFLMS).

## Adjustable image resolution ResNet, DenseNet, and other models
The [ManyModel.py](ManyModel.py) file uses the various models from
`tf.keras.applications` to demonstrate TensorFlow Large Model
Support (TFLMS) in a Keras model that cannot fit in GPU memory when
using larger resolution data. It provides a convenient way to test out the
capabilities of TFLMS with various flavors of ResNet, DenseNet, Inception,
MobileNet, NASNet, and Xception models. Command line parameters allow
the user to change the size of the input image data, enable or disable
TFLMS, and log memory allocator statistics.

The ManyModel.py example can be run by adding the `examples` directory to
the PYTHONPATH and running like as shown:

```bash
cd examples
export PYTHONPATH=`pwd`
python ManyModel.py -h
```

## Memory Allocator statistics
TensorFlow Large Model Support provides APIs to retrieve statistics from
TensorFlow's GPU memory allocator. These statistics provide a means to
do deeper analysis of a model's memory usage, including how often TFLMS
reclaims memory and how many bytes of memory are being reclaimed.

The [callbacks module](callbacks.py) provides a working example of how the APIs
can be used in used to log per-iteration and aggregate memory statistics. The
`LMSStatsLogger` Keras callback in this module is used by the ManyModel
example to demonstrate how the statistics APIs can used in model training.

For more information see the [allocator statistics documentation](AllocatorStats.md).
