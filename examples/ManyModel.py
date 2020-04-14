# Copyright 2018, 2020. IBM All Rights Reserved.
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
# how to enable Large Model Support (LMS) for TensorFlow 2 in a Keras model that
# cannot fit in GPU memory when using larger resolution data. To simplify
# the running of the model with different higher resolution images,
# the code uses a random image generator to generate synthetic image data.
#
# This model provides a convenient way to test out the capabilities
# of LMS. Command line parameters allow the user to test different models,
# change the size of the input image data, enable or disable LMS,
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
#   python ManyModel.py --image_size 2000
# Run with LMS:
#   python ManyModel.py --image_size 4000 --lms


import argparse
import tensorflow as tf

import numpy as np
import os
from tensorflow.keras import backend as K
from callbacks import CudaProfileCallback, LMSStatsLogger, LMSStatsAverage

# import Horovod if invoked with distribution
hvd = None
if "OMPI_COMM_WORLD_RANK" in os.environ:
    import horovod.tensorflow.keras as hvd
    hvd.init()

#tf.logging.set_verbosity(tf.logging.INFO)
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



def generate_stats_name(model, root):
    # Generates the name of the output stats file.
    # If Horovod distribution is enabled, the node and GPU ID are appended to the end
    return ('%s_%s%s%s.csv' %
           (model, root,
           ('_%s' % os.environ["HOSTNAME"] if hvd else ""),
           (('_gpu%s' % hvd.local_rank()) if hvd else "")))


def get_callbacks(args):
    callbacks = []

    if hvd:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

    if args.nvprof:
        callbacks.append(CudaProfileCallback(args.nvprof_epoch,
                                             args.nvprof_start,
                                             args.nvprof_stop))

    if args.lms_stats:
        stats_filename = os.path.join(args.output_dir,
                                      generate_stats_name(args.model, "lms_stats"))
        callbacks.append(LMSStatsLogger(stats_filename))

    if args.lms_stats_average:
        stats_filename = os.path.join(args.output_dir,
                                      generate_stats_name(args.model, "lms_stats_average"))
        lms = LMSStatsAverage(stats_filename,
                              args.image_size,
                              batch_size=args.batch_size,
                              start_batch=args.lms_stats_warmup_steps)
        callbacks.append(lms)

    return callbacks

def run_model(args):
    if args.lms:
        tf.config.experimental.set_lms_enabled(True)

    image_dim = args.image_size
    opt = tf.keras.optimizers.RMSprop()
    if hvd:
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)
        tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


        # Horovod: add Horovod DistributedOptimizer.
        opt = hvd.DistributedOptimizer(opt)
        steps_per_epoch = max(1, args.steps // hvd.size())
        experimental_run_tf_function = False
    else:
        steps_per_epoch = args.steps
        experimental_run_tf_function = True

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

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  experimental_run_tf_function=experimental_run_tf_function)

    random_generator = random_image_generator(batch_size, num_classes,
                                              input_shape)


    model.fit(random_generator, steps_per_epoch=steps_per_epoch,
              verbose=1 if not hvd or hvd.rank() == 0 else 0,
              epochs=args.epochs, callbacks=get_callbacks(args))


def main():
    parser = argparse.ArgumentParser()
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
                           help='Enable LMS')
    lms_group.add_argument('--no-lms', dest='lms', action='store_false',
                           help='Disable LMS (Default)')
    parser.set_defaults(lms=False)

    lms_stats = parser.add_mutually_exclusive_group(required=False)
    lms_stats.add_argument('--lms_stats', dest='lms_stats', action='store_true',
                           help='Log LMS per-step stats to a file named '
                                '<modelName>_lms_stats.csv')
    lms_stats.add_argument('--no-lms_stats', dest='lms_stats',
                           action='store_false',
                           help='Disable logging LMS per-step stats (Default)')
    parser.set_defaults(lms_stats=False)

    lms_stats_average = parser.add_mutually_exclusive_group(required=False)
    lms_stats_average.add_argument('--lms_stats_average',
         dest='lms_stats_average',
         action='store_true',
         help='Log LMS average stats to a file named '
              '<modelName>_lms_stats_average.csv')
    lms_stats_average.add_argument('--no-lms_stats_average',
        dest='lms_stats_average', action='store_false',
        help='Disable logging LMS average stats (Default)')
    parser.set_defaults(lms_stats_average=False)

    parser.add_argument('--lms_stats_warmup_steps',
                        default=5,
                        help='The number of steps to train before starting '
                             'LMS statistics recording. (Default 5)',
                        type=int)
    parser.add_argument('--output_dir',
                        default='./model_outputs',
                        help='The directory to write output files to.',)


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

    args = parser.parse_args()
    if args.lms_stats or args.lms_stats_average:
        if not os.path.exists(args.output_dir):
            if hvd:
                if hvd.local_rank() == 0:
                    os.makedirs(args.output_dir, exist_ok=True)
            else:
                os.makedirs(args.output_dir, exist_ok=True)
    run_model(args)

if __name__ == "__main__":
    main()
