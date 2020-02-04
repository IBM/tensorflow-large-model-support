# Copyright 2019, 2020. IBM All Rights Reserved.
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
import csv
import ctypes
import os
import time
import tensorflow as tf
import statistics

from tensorflow.keras.callbacks import Callback
from tensorflow.estimator import SessionRunHook

_cudart = ctypes.CDLL('libcudart.so')
nvtx=  ctypes.CDLL("libnvToolsExt.so")
nvtx.nvtxMarkA.restype = None

STATS_KEYS = ['time', 'allocs', 'reclaim_ones',
              'reclaim_alls', 'defrags', 'gib_reclaimed', 'gib_defragged']

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
        nvtx.nvtxRangePushA(ctypes.c_char_p("Iteration".encode("ascii")))
    def on_batch_end(self, batch, logs=None):
        ret = nvtx.nvtxRangePop()


class LMSStats():

    def __init__(self, gpu_id=0):
        self._gpu_id = gpu_id

        self._start_stats = {key:0 for key in STATS_KEYS}
        self._end_stats = self._start_stats.copy()
        self._cumulative_stats = self._start_stats.copy()
        self._num_steps = 0
        self._step_times = []

    def _get_stats(self):
        stats = {}
        stats['time'] = time.time()
        stats['allocs'] = tf.experimental.get_num_allocs(self._gpu_id)
        stats['reclaim_ones'] = tf.experimental.get_num_single_reclaims(self._gpu_id)
        stats['reclaim_alls'] = tf.experimental.get_num_full_reclaims(self._gpu_id)
        stats['defrags'] = tf.experimental.get_num_defragmentations(self._gpu_id)
        stats['gib_reclaimed'] = tf.experimental.get_bytes_reclaimed(self._gpu_id) / 1073741824.0
        stats['gib_defragged'] = tf.experimental.get_bytes_defragged(self._gpu_id) / 1073741824.0
        return stats

    def step_begin(self):
        self._start_stats = self._get_stats()
        return self._start_stats.copy()

    def step_end(self):
        self._num_steps += 1
        self._end_stats = self._get_stats()
        step_diff = self.get_last_step_difference()
        self._step_times.append(step_diff['time'])
        for key in STATS_KEYS:
            self._cumulative_stats[key] += step_diff[key]
        self._cumulative_stats['num_steps'] = self._num_steps
        return self._end_stats.copy()

    def get_last_step_difference(self):
        return {k: self._end_stats[k]-self._start_stats[k] for k in STATS_KEYS}

    def get_cumulative_stats(self):
        return self._cumulative_stats.copy()

    def get_average_stats(self):
        s = self._num_steps * 1.0
        average =  {k: v/s for (k,v) in self._cumulative_stats.items()}
        average['num_steps'] = self._num_steps
        return average

    def get_median_time(self):
        return statistics.median(self._step_times)

# writes the stats from the last call to step_end to the log file
def write_step_stats(logfile, step_type, epoch, step_num, step_stats):
        row = [step_type, epoch, step_num]
        row.append(step_stats['time'])
        row.append(step_stats['allocs'])
        row.append(step_stats['reclaim_ones'])
        row.append(step_stats['reclaim_alls'])
        row.append(step_stats['defrags'])
        row.append(step_stats['gib_reclaimed'])
        row.append(step_stats['gib_defragged'])
        with open(logfile, 'a+', newline='') as csvfile:
            statswriter = csv.writer(csvfile)
            statswriter.writerow(row)


def write_step_log_header(logfile):
    with open(logfile, 'w', newline='') as csvfile:
        statswriter = csv.writer(csvfile)
        statswriter.writerow(['step type', 'epoch', 'step',
                              'duration', 'allocs', 'reclaimOnes',
                              'reclaimAlls', 'defrags',
                              'GiB reclaimed', 'GiB defragged'])


class LMSStatsLogger(Callback):
    def __init__(self, logfile, gpu_id=0):
        self._epoch=0
        self._logfile = logfile
        self._lms_stats = LMSStats(gpu_id=gpu_id)

    def set_params(self, params):
        write_step_log_header(self._logfile)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        self._lms_stats.step_begin()

    def on_test_batch_begin(self, batch, logs=None):
        self._lms_stats.step_begin()

    def on_train_batch_end(self, batch, logs=None):
        self._lms_stats.step_end()
        step_diff = self._lms_stats.get_last_step_difference()
        write_step_stats(self._logfile, 't', self._epoch, batch, step_diff)

    def on_test_batch_end(self, batch, logs=None):
        self._lms_stats.step_end()
        step_diff = self._lms_stats.get_last_step_difference()
        write_step_stats(self._logfile, 'v', self._epoch, batch, step_diff)


class LMSStatsTrainingStepsAverage(Callback):
    def __init__(self, gpu_id=0):
        self._epoch=0
        self._lms_stats = LMSStats(gpu_id=gpu_id)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        # Do not record the first step of the first epoch as it contains
        # TensorFlow startup overhead
        if (batch == 0) and (self._epoch == 0):
            return
        self._lms_stats.step_begin()

    def on_train_batch_end(self, batch, logs=None):
        # Do not record the first step of the first epoch as it contains
        # TensorFlow startup overhead
        if (batch == 0) and (self._epoch == 0):
            return
        self._lms_stats.step_end()

    def on_train_end(self, logs=None):
        print('LMS statistics averages:', self._lms_stats.get_average_stats())


class LMSStatsLoggerRunHook(SessionRunHook):
    def __init__(self, logfile, gpu_id=0):
        self._logfile = logfile
        self._lms_stats = LMSStats(gpu_id=gpu_id)
        self._step = 0

    # Estimator SessionRunHook methods
    def begin(self):
        write_step_log_header(self._logfile)

    def before_run(self, run_context):
        self._lms_stats.step_begin()
        self._step += 1

    def after_run(self, run_context, run_values):
        self._lms_stats.step_end()
        step_diff = self._lms_stats.get_last_step_difference()
        write_step_stats(self._logfile, 't', 0, self._step, step_diff)


class LMSStatsAverage(Callback):
    def __init__(self, logfile, image_size, image_dimensions=2, gpu_id=0,
                 batch_size=1, start_epoch=0, start_batch=2):
        self._epoch=0
        self._logfile = logfile
        self._lms_stats = LMSStats(gpu_id=gpu_id)
        self._start_epoch = start_epoch
        self._start_batch = start_batch
        self._num_dims = image_dimensions
        self._dim = image_size
        self._batch_size = batch_size

    def _should_record(self, batch):
        if (batch >= self._start_batch) and (self._epoch >= self._start_epoch):
            return True
        return False

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_begin(self, batch, logs=None):
        if not self._should_record(batch):
            return
        self._lms_stats.step_begin()

    def on_train_batch_end(self, batch, logs=None):
        if not self._should_record(batch):
            return
        self._lms_stats.step_end()

    def on_train_end(self, logs=None):
        stats_dict = self._lms_stats.get_average_stats()
        rate_field = 'megapixels/sec'
        if self._num_dims == 3:
            rate_field = 'megavoxels/sec'

        duration = stats_dict['time']
        rate = ((self._batch_size * (self._dim ** self._num_dims)) / duration ) / 1000000.0

        duration = self._lms_stats.get_median_time()
        median_rate = ((self._batch_size * (self._dim ** self._num_dims)) / duration ) / 1000000.0
        median_rate_field = 'median '+rate_field
        # Put these columns first, with the rest of the stats in a sorted
        # order.
        fieldnames = ['image_size', rate_field, median_rate_field, 'gib_reclaimed']
        dictkeys = list(stats_dict)
        dictkeys.remove('gib_reclaimed')
        fieldnames.extend(sorted(dictkeys))

        stats_dict['image_size'] = self._dim
        stats_dict[rate_field] = rate
        stats_dict[median_rate_field] = median_rate

        write_header = not os.path.exists(self._logfile)
        with open(self._logfile, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(stats_dict)
