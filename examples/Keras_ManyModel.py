# Copyright 2018, 2019. IBM All Rights Reserved.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This example uses a variety of models from keras_applications to demonstrate
# how to enable TensorFlow Large Model Support (TFLMS) in a Keras model that
# cannot fit in GPU memory when using larger resolution data. To simplify
# the running of the model with different higher resolution images,
# the code uses a random image generator to generate synthetic image data.
#
# This model provides a convenient way to test out the capabilities
# of TFLMS. Command line parameters allow the user to change the size of
# the input image data, enable or disable TFLMS, set TFLMS tunables,
# and enable or disable CUDA profiling to ease collection of profile data.
#
# This example uses randomly generated synthetic image data. The random
# generation is intentionally not using GPU operations to avoid impacting
# the memory usage profile of the model. Since random generated images
# are being used this code should not be used as an example on how to
# correctly and efficiently pipeline data using datasets, etc.
#
# Invocation examples:
# Run without LMS:
#   python Keras_ManyModel.py --image_size 2300 --model resnet50
# Run with LMS:
#   python Keras_ManyModel.py --image_size 3900 --lms --model resnet50


import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
#from distutils.version import LooseVersion
import keras_applications
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow_large_model_support import LMS
import ctypes
_cudart = ctypes.CDLL('libcudart.so')

tf.logging.set_verbosity(tf.logging.INFO)

model_choices = {'resnet50': tf.keras.applications.ResNet50,
                 'resnet101': tf.keras.applications.ResNet101,
                 'resnet152': tf.keras.applications.ResNet152,
                 'resnet50v2': tf.keras.applications.ResNet50V2,
                 'resnet101v2': tf.keras.applications.ResNet101V2,
                 'resnet152v2': tf.keras.applications.ResNet152V2,
                 'densenet121': tf.keras.applications.DenseNet121,
                 'densenet169': tf.keras.applications.DenseNet169,
                 'densenet201': tf.keras.applications.DenseNet201,
                 'inception': tf.keras.applications.InceptionV3,
                 'inceptionresnet': tf.keras.applications.InceptionResNetV2,
                 'mobilenet': tf.keras.applications.MobileNet,
                 'mobilenetv2': tf.keras.applications.MobileNetV2,
                 'nasnetlarge': tf.keras.applications.NASNetLarge,
                 'nasnetmobile': tf.keras.applications.NASNetMobile,
                 'xception': tf.keras.applications.Xception}

# Import the distribution module if invoked with distribution.
dist_mod = None
if "DDL_OPTIONS" in os.environ:
  import ddl
  dist_mod = ddl


class CudaProfileCallback(Callback):
    def __init__(self, profile_epoch, profile_batch_start, profile_batch_end):
        self._epoch = profile_epoch - 1
        self._start = profile_batch_start
        self._end = profile_batch_end
        self.epoch_keeper = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_keeper = epoch
    def on_batch_begin(self, batch, logs=None):
        if batch == self._start and self.epoch_keeper == self._epoch:
            print('Starting cuda profiler')
            _cudart.cudaProfilerStart()
        if batch == self._end and self.epoch_keeper == self._epoch:
            print('Stopping cuda profiler')
            _cudart.cudaProfilerStop()

def random_image_generator(batch_size, num_classes, input_shape):
    # This generator yields batches of randomly generated images and categories.
    # The random generation parts came from
    # https://github.com/tensorflow/tensorflow/blob/v1.12.0/tensorflow/python/keras/testing_utils.py#L29

    # These templates and random_data generations take a long time for large dimensions and should
    # really be in the while loop below to have
    # better random images being generated for every yield. They were moved out of the while loop
    # to speed up the generator since the intent of this example is to show resolutions with
    # and without TFLMS versus a good data spread and loss / accuracy numbers.
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    random_data = np.random.normal(loc=0, scale=1., size=input_shape)
    while True:
        y = np.random.randint(0, num_classes, size=(batch_size,))
        x = np.zeros((batch_size,) + input_shape, dtype=np.float32)
        for i in range(batch_size):
            x[i] = templates[y[i]] + random_data
        x_array = np.array(x)
        y_array = tf.keras.utils.to_categorical(y, num_classes)
        yield(x_array, y_array)

def get_callbacks(args):
    callbacks = []

    if args.nvprof:
        callbacks.append(CudaProfileCallback(args.nvprof_epoch,
                                             args.nvprof_start,
                                             args.nvprof_stop))
    if "DDL_OPTIONS" in os.environ:
        callbacks.append(dist_mod.DDLCallback())
        callbacks.append(dist_mod.DDLGlobalVariablesCallback())

    # Enable TFLMS
    if args.lms:
        handle_mem_ratio(args)
        # Specifying this starting name, from previous runs of LMS,
        # speeds up graph analysis time.
        serialization = []
        if args.serialization > 0:
            serialization.append('%s:' % args.serialization)
        lms = LMS(swapout_threshold=args.swapout_threshold,
                  swapin_groupby=args.swapin_groupby,
                  swapin_ahead=args.swapin_ahead,
                  sync_mode=args.sync_mode,
                  serialization=serialization,
                  serialization_by_size=args.serialization_by_size)
        lms.batch_size = args.batch_size
        callbacks.append(lms)

    return callbacks

def handle_mem_ratio(args):
    mem_ratio = os.environ.get('TF_LMS_SIMULATOR_MEM_RATIO')
    if not mem_ratio:
      os.environ['TF_LMS_SIMULATOR_MEM_RATIO'] = '0.8'
    mem_ratio = float(os.environ.get('TF_LMS_SIMULATOR_MEM_RATIO',
                                1.0))
    if (args.swapout_threshold < 0 or args.swapin_groupby < 0 or
        args.swapin_ahead < 0) and mem_ratio > 0.8:
            print('WARNING: The environment variable, '
                  'TF_LMS_SIMULATOR_MEM_RATIO is set higher than 0.8. '
                  'The operations used by this model have higher '
                  'GPU memory overhead than their input and output tensor '
                  'sizes use. It is recommended that '
                  'TF_LMS_SIMULATOR_MEM_RATIO be set at 0.8 or lower to avoid '
                  'out of memory issues with auto tune chosen values.')

def run_model(args):
    image_dim = args.image_size

    if args.channels_last:
        K.set_image_data_format('channels_last')
        input_shape = (image_dim, image_dim, 3)
    else:
        K.set_image_data_format('channels_first')
        input_shape = (3, image_dim, image_dim)

    num_classes = 15
    batch_size = args.batch_size
    model_class = model_choices.get(args.model)
    model = model_class(weights=None, include_top=True, input_shape=input_shape,
                        classes=num_classes)

    if args.tensors_on_oom:
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        run_metadata = tf.RunMetadata()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      options=run_options, run_metadata=run_metadata)
    else:
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    random_generator = random_image_generator(batch_size, num_classes,
                                              input_shape)
    steps_per_epoch = args.steps
    if dist_mod:
        steps_per_epoch = steps_per_epoch // dist_mod.size()

    verbose = 0 if dist_mod and dist_mod.rank() != 0 else 1
    model.fit_generator(random_generator, steps_per_epoch=steps_per_epoch,
                        epochs=args.epochs, callbacks=get_callbacks(args),
                        verbose=verbose)


def main(model=None):
    parser = argparse.ArgumentParser()
    if model is None:
        parser.add_argument('--model', choices=list(model_choices.keys()),
                            default='resnet50',
                            type=lambda s : s.lower(),
                            help='Keras model to run training against. '
                                 'The model names are case insensitive. '
                                 '(Default resnet50)')
    parser.add_argument("--epochs", type=int,
                        default=1,
                        help='Number of epochs to run. (Default 1)')
    parser.add_argument("--steps", type=int,
                        default=10,
                        help='Number of steps per epoch. (Default 10)')
    parser.add_argument("--image_size", type=int,
                        default=500,
                        help='Dimension of one side of the square image '
                             'to be generated. (Default 500)')
    
    parser.add_argument("--batch_size", type=int,
                        default=1,
                        help='Batch size. (Default 1)')

    # LMS parameters
    lms_group = parser.add_mutually_exclusive_group(required=False)
    lms_group.add_argument('--lms', dest='lms', action='store_true',
                           help='Enable TFLMS')
    lms_group.add_argument('--no-lms', dest='lms', action='store_false',
                           help='Disable TFLMS (Default)')
    parser.set_defaults(lms=False)
    parser.add_argument("--swapout_threshold", type=int, default=-1,
                        help='The TFLMS swapout_threshold parameter. See the '
                             'TFLMS documentation for more information. '
                             'Default `-1` (auto mode).')
    parser.add_argument("--swapin_groupby", type=int, default=-1,
                        help='The TFLMS swapin_groupby parameter. See the '
                             'TFLMS documentation for more information. '
                             'Default `-1` (auto mode).')
    parser.add_argument("--swapin_ahead", type=int, default=-1,
                        help='The TFLMS swapin_ahead parameter. See the '
                             'TFLMS documentation for more information. '
                             'Default `-1` (auto mode).')
    parser.add_argument("--serialization", type=int, default=-1,
                        help='The layer to start serialization on. This '
                             'number will be passed to the LMS serialization '
                             'parameter as the start of a slice like this: '
                             '[\'parameter:\']. See the TFLMS documentation '
                             'for more information. Default -1, no '
                             'serialization.')
    parser.add_argument("--sync_mode", type=int, default=0,
                        help='Sync mode of TFLMS. See the TFLMS documentation '
                             'for more information')
    parser.add_argument("--serialization_by_size", type=float, default=0,
                        help='Serialize operations in levels of the '
                             'topological sort, if the cumulative memory '
                             'consumption of the level is greater than '
                             'serialization_by_size. The size unit is GiB. '
                             'Default 0 (turn off).')
    # nvprof parameters
    nvprof_group = parser.add_mutually_exclusive_group(required=False)
    nvprof_group.add_argument('--nvprof', dest='nvprof', action='store_true',
                              help='Enable CUDA profilng for nvprof profiling.')
    nvprof_group.add_argument('--no-nvprof', dest='nvprof',
                              action='store_false',
                              help='Disable CUDA profilng for nvprof '
                                   'profiling. (Default)')
    parser.set_defaults(nvprof=False)

    parser.add_argument("--nvprof_epoch", type=int,
                        default=1,
                        help='The epoch in which to run CUDA profiling. '
                             '(Default 1)')
    parser.add_argument("--nvprof_start", type=int,
                        default=4,
                        help='The batch in which to start CUDA profiling. '
                             '(Default 4)')
    parser.add_argument("--nvprof_stop", type=int,
                        default=9,
                        help='The batch in which to stop CUDA profiling. '
                             '(Default 9)')

    # channels first/last parameter
    ch_fl_group = parser.add_mutually_exclusive_group(required=False)
    ch_fl_group.add_argument('--channels_last', dest='channels_last',
                             action='store_true',
                             help='Create the model and images with '
                                  'channels last.')
    ch_fl_group.add_argument('--no-channels_last', dest='channels_last',
                             action='store_false',
                             help='Create the model and images with '
                                  'channels first. (Default)')
    parser.set_defaults(channels_last=False)

    # Show tensors in GPU memory on OOM
    tensors_on_oom = parser.add_mutually_exclusive_group(required=False)
    tensors_on_oom.add_argument('--show_tensors_on_oom', dest='tensors_on_oom',
                                action='store_true',
                                help='Show tensors in GPU on out of memory '
                                     'errors.')
    tensors_on_oom.add_argument('--no-show_tensors_on_oom',
                                dest='tensors_on_oom',
                                action='store_false',
                                help='Do not show tensors in GPU on out of '
                                     'memory errors. (Default)')
    parser.set_defaults(tensors_on_oom=False)
    args = parser.parse_args()
    if model is not None:
        args.model = model
    run_model(args)

if __name__ == "__main__":
    main()
