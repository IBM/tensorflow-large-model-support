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

# This model uses the ResNet50 from keras_applications to demonstrate
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
# This model uses randomly generated synthetic image data. The random
# generation is intentionally not using GPU operations to avoid impacting
# the memory usage profile of the model. Since random generated images
# are being used this code should not be used as an example on how to
# correctly and efficiently pipeline data using datasets, etc.
#
# Invocation examples:
# Run without LMS:
#   python Keras_ResNet50.py --image_size 2300
# Run with LMS:
#   python Keras_ResNet50.py --image_size 3900 --lms
import Keras_ManyModel as k
if __name__ == "__main__":
    k.main('resnet50')
