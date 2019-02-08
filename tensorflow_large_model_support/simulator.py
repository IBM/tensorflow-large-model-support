# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2018. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

"""Simulator
"""

import os

import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops import memory_stats_ops

from tensorflow_large_model_support import util as ut


class Simulator(object):
    """A simulator to simulate how LMS works.
    It is used to predict whether LMS works with given LMS parameters.
    Operations that do not consume GPU memory will be ignored.
    """
    def __init__(self, lms, ratio=0.9, swapout_delay=1, debug_level=1,
                 plot=False):
        self._lms = lms
        self._ratio = ratio
        self._debug_level = debug_level
        self._plot = plot
        # swapout ops would take time to transfer data from device to host.
        # this variable is used to simulate how slow the transfer is.
        # TODO: this variable should depend on the tensor size.
        self._swapout_delay = swapout_delay
        # trainable variables
        self._trainable_vars = None
        # memory limitation
        self._max_mem = 0

        # the following variables need to be reset before each play()
        # memory to store tensors
        self._mem = {}
        # used memory at a given time
        self._used_mem = 0
        # traces of used memory
        self._mem_traces = []
        # each tensor has a ref_count showing how many ops will consume it
        self._ref_counts = {}
        self._dead_ref_counts = set()
        # simulated ops
        self._simulated_ops = set()
        # keep tensors that were swapped out
        self._swapouts = {}
        # keep tensors that were swapped in
        self._swapins = set()
        # keep control input ops for swapins
        self._ctrl_inputs = {}

        # aliases, variables
        self._graph = self._lms._graph
        self._topo_sort = self._lms._topo_sort
        self._lms_dir = self._lms._lms_dir
        self._distance = self._lms._distance
        self._batch_size = self._lms._batch_size

        # aliases, methods
        self._groupby = self._lms._groupby
        self._get_level = self._lms._get_level
        self._get_ops_by_level = self._lms._get_ops_by_level
        self._get_earliest_op = self._lms._get_earliest_op
        self._get_latest_op = self._lms._get_latest_op
        self._search_by_level = self._lms._search_by_level
        self._filter_by_source_ops = self._lms._filter_by_source_ops
        self._filter_by_dest_ops = self._lms._filter_by_dest_ops
        self._filter_by_swapout_threshold = self._lms._filter_by_swapout_threshold
        self._is_valid_op = self._lms._is_valid_op

        self._initialize()

    def _initialize(self):
        """Initialize some variables
        """
        with tf.Session(graph=tf.Graph()) as sess:
            self._max_mem = sess.run(memory_stats_ops.BytesLimit())
        # exclude memories for variables
        learning_params_size = 0
        self._trainable_vars = tf.trainable_variables()
        for v in self._trainable_vars:
            var_size = self._ts_size(v)
            self._max_mem -= var_size
            learning_params_size += var_size
        # use only `ratio` percent of the available memory
        self._max_mem *= self._ratio
        self._log_info("Available memory for simulation: {} GiB".format(
            round(self._max_mem/1024/1024/1024, 2)) +
                       " (memory ratio: {})".format(self._ratio), 0)

    def _reset(self):
        """Reset memory to the initial state.
        """
        self._mem = {}
        self._used_mem = 0
        self._mem_traces = []
        self._ref_counts = {}
        self._dead_ref_counts = set()
        self._simulated_ops = set()
        self._swapouts = {}
        self._swapins = set()
        self._ctrl_inputs = {}

    # @ut.measure_time
    def play(self, threshold, ahead, groupby, sync_mode):
        """Check whether LMS works with parameters `threshold` and `ahead`.

        Return:
          True if successfully. Otherwise, False.
        """
        self._reset()

        trainable_vars = [v.op.name for v in self._trainable_vars]
        # start simulating
        passed = True
        for k in range(0, self._topo_sort.size):
            self._log_info("Simulate level {}".format(k))

            self._gc()  # collect swapout tensors

            k_ops = self._get_ops_by_level(k)

            for op in k_ops:
                self._log_info("[{}] execute op {}".format(k, op.name))
                in_tensors = set(op.inputs)
                out_tensors = set(op.outputs)
                op_name = op.name

                if not self._is_valid_op(op):
                    continue

                # do not simulate ops relating to variables
                if op_name in trainable_vars:
                    continue

                # allocate memory for inputs
                swapin_tensors = {}
                for ts in in_tensors:
                    ts_name = ts.name
                    # whether this tensor was produced by an ignorable op
                    if not self._is_valid_op(ts.op):
                        continue
                    # whether this tensor was swapped in?
                    found, sin = self._is_swapin_tensor(ts_name, op_name)
                    if found:
                        swapin_tensors[ts_name] = sin
                        continue
                    else:
                        if ts_name not in self._mem:
                            ts_size = self._ts_size(ts)
                            lifetime = len(ts.consumers())
                            ok = self._allocate(ts_name, ts_size, lifetime)
                            if not ok:
                                passed = False
                                if not self._plot:
                                    return passed

                # allocate memory for outputs
                for ts in out_tensors:
                    ts_name = ts.name
                    consumers = ts.consumers()
                    n_consumers = len(consumers)
                    for cop in consumers:
                        if not self._is_valid_op(cop):
                            n_consumers -= 1
                    if n_consumers == 0:
                        continue    # no ops consuming `ts`

                    ts_size = self._ts_size(ts)
                    ok = self._allocate(ts_name, ts_size, n_consumers)
                    if not ok:
                        passed = False
                        if not self._plot:
                            return passed

                # simulate execution
                for ts in in_tensors:
                    # check if the tensor was produced by an ignorable op
                    if not self._is_valid_op(ts.op):
                        continue
                    # check if the tensor is a swapin tensor
                    ts_name = ts.name
                    if ts_name in swapin_tensors:
                        s = swapin_tensors[ts_name]
                    else:
                        s = ts_name
                    self._update_ref_counts(s, -1)

                # simulate swapping out/in tensors
                if op not in self._filter_by_source_ops({op}):
                    continue
                for ts in out_tensors:
                    self._add_swapout_swapin(
                        k, op, ts, threshold, ahead, groupby, sync_mode)

                # finish simulating this ops
                self._simulated_ops.add(op)

                # collect swapout tensors before triggerring other swapins.
                # only collect swapout tensors once the latest ops that
                # use them were simulated.
                for ts in {x for x in self._swapouts
                           if (x in self._mem and self._ref_counts[x] > 0
                               and self._swapouts[x] in self._simulated_ops)}:
                    self._update_ref_counts(ts, -1)

                # garbage collection
                self._gc()  # collect input and swapout tensors

                # trigger other swapins
                ok = self._trigger_swapins(k, op)
                if not ok:
                    passed = False
                    if not self._plot:
                        return passed

            self._log_info("[{}] available memory {}".format(
                k, self._get_free_mem()))

        self._gc()
        self._log_info("Swapped out {} tensors".format(len(self._swapouts)))

        if passed:
            self._generate_diagram(threshold, ahead, groupby, sync_mode)
            self._log_info("Found a parameter set: " +
                           "swapout_threshold {}".format(threshold) +
                           ", swapin_ahead {}".format(ahead) +
                           ", swapin_groupby {}".format(groupby) +
                           ", sync_mode {}".format(sync_mode), 0)
        else:
            if self._plot:
                self._generate_diagram(threshold, ahead, groupby, sync_mode)

        return passed

    def _trigger_swapins(self, k, op):
        # allocate memory for swapin tensors that are triggered by
        # this operation
        passed = True
        name_size_lifetimes = self._ctrl_inputs[op] \
                              if op in self._ctrl_inputs else set()
        for nsl in name_size_lifetimes:
            ts_name, ts_size, lifetime = nsl
            self._log_info("[{}] swapped in {}".format(k, ts_name))
            ok = self._allocate(ts_name, ts_size, lifetime)
            if not ok:
                passed = False
                if not self._plot:
                    return passed
        return passed

    def _add_swapout_swapin(
            self, k, op, ts,
            threshold, ahead, groupby, sync_mode):
        # swap out tensors
        ndims = ts.shape.ndims
        if ndims is None or ndims <= 1:
            return
        ts_name = ts.name

        # Maybe all consumers are ignorable ops,
        # so there is no need to swap out this tensor
        if ts_name not in self._ref_counts:
            return

        ts_size = self._ts_size(ts)
        # filter by threshold
        dest_ops = self._filter_by_swapout_threshold(
            op, ts, set(), threshold)
        # filter by dest operations
        dest_ops = self._filter_by_dest_ops(dest_ops)
        if not dest_ops:
            return
        # swapout tensors will be collected later
        latest_op = self._get_latest_op({
            x for x in set(ts.consumers()) - dest_ops})
        if latest_op is None:
            latest_op = op
        self._swapouts[ts_name] = latest_op
        self._ref_counts[ts_name] -= len(dest_ops)
        if (not self._is_swapout_sync(sync_mode) and
            self._ref_counts[ts_name] > 0):
            if (self._topo_sort.size - self._get_level(latest_op) > \
                self._swapout_delay):
                self._ref_counts[ts_name] += self._swapout_delay
        self._update_ref_counts(ts_name, 0)
        self._log_info("[{}] swapped out {}".format(k, ts_name))
        # add swapin ops
        dests_grp = self._groupby(dest_ops, groupby)
        for dests in dests_grp:
            # create a new tensor to simulate swapin
            s = self._create_swapin_name(ts, dests)
            ts_info = (s, ts_size, len(dests))
            # put the tensor into the swapins queue
            dest = self._get_earliest_op(dests)
            ctrl_op = None
            if not self._is_swapin_sync(sync_mode):
                ctrl_op, _ = self._search_by_level(op, dest, ahead)
            if ctrl_op is None:  # swapin sync mode
                ctrl_op = self._get_latest_op({
                    op for op in ut.fanins(dest)})
            if ctrl_op in self._ctrl_inputs:
                self._ctrl_inputs[ctrl_op] |= {ts_info}
            else:
                self._ctrl_inputs[ctrl_op] = {ts_info}
            self._swapins.add(s)

    def _allocate(self, ts_name, ts_size, lifetime):
        """Allocate memory for tensor `ts`.

        Return:
          True if successfully. Otherwise, False.
        """
        succeed = True
        if ts_size >= self._get_free_mem():
            succeed = False
            # out of memory
            self._log_info(
                "OOM tensor {}, size {}, used {}, free {}".format(
                    ts_name, ts_size,
                    self._used_mem,
                    self._get_free_mem()))
            if not self._plot:
                return succeed
        # allocate
        self._used_mem += ts_size
        self._mem[ts_name] = ts_size
        self._update_ref_counts(ts_name, lifetime, override=True)
        self._mem_traces.append(self._used_mem)
        self._log_info(
            "allocated {} bytes for {}, used {}, free {}".format(
                ts_size, ts_name,
                self._used_mem,
                self._get_free_mem()))
        return succeed

    def _release(self, ts_name):
        """Release memory taken by tensor `ts`.
        """
        ts_size = self._mem[ts_name]
        del self._mem[ts_name]
        self._delete_ref_counts(ts_name)
        self._used_mem -= ts_size
        self._mem_traces.append(self._used_mem)
        self._log_info(
            "released {} bytes taken by {}, used {}, free {}".format(
                ts_size, ts_name,
                self._used_mem,
                self._get_free_mem()))

    def _gc(self):
        """Simulate TensorFlow garbage collection.
        A tensor will be released from mem if its `ref_count` becomes zero.
        """
        while len(self._dead_ref_counts) > 0:
            ts_name = next(iter(self._dead_ref_counts))
            self._release(ts_name)

    def _get_free_mem(self):
        """Return the available memory
        """
        return self._max_mem - self._used_mem

    def _generate_diagram(self, threshold, ahead, groupby, sync_mode):
        """Generate a diagram of memory consumption.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        label_str = ""
        if threshold == self._topo_sort.size:
            label_str = "No LMS"
        else:
            label_str = "LMS(swapout_threshold: {}".format(threshold) + \
                        ", swapin_ahead: {}".format(ahead) + \
                        ", swapin_groupby: {}".format(groupby) + \
                        ", sync_mode: {})".format(sync_mode)
        plt.plot([m/(1e9) for m in self._mem_traces], label=label_str)
        plt.title("Simulation of memory consumption")
        plt.xlabel("Allocation/Deallocation steps")
        plt.ylabel("GigaBytes")
        plt.legend(loc='upper left', fontsize='x-small')
        plt.grid(True)
        if not os.path.exists(self._lms_dir):
            os.makedirs(self._lms_dir)
        if threshold == self._topo_sort.size:
            label_str = "{}/nolms_simulator_mem_traces".format(self._lms_dir)
        else:
            label_str = "{}/tflms_simulator_mem_traces".format(self._lms_dir) + \
                        "_swapout_threshold{}".format(threshold) + \
                        "_swapin_ahead{}".format(ahead) + \
                        "_swapin_groupby{}".format(groupby) + \
                        "_sync_mode{}".format(sync_mode)
        plt.savefig(label_str + ".pdf", format='pdf')
        plt.close()

    def _ts_size(self, ts):
        return ut.get_tensor_size(ts, self._batch_size)

    def _create_swapin_name(self, ts, dests):
        """Create a name for a swap-in tensor.
        """
        s = ts.name + "_" + "_".join(x.name for x in dests)
        return s

    def _is_swapin_tensor(self, ts_name, op_name):
        """Check if the tensor is a swapin tensor AND already swapped in.
        """
        for s in self._swapins:
            if ts_name in s and op_name in s and s in self._mem:
                return (True, s)
        return (False, ts_name)

    def _is_swapin_sync(self, sync_mode):
        return sync_mode in {2, 3}

    def _is_swapout_sync(self, sync_mode):
        return sync_mode in {1, 3}

    def _update_ref_counts(self, ts_name, value, override=False):
        if override:
            self._ref_counts[ts_name] = value
        else:
            self._ref_counts[ts_name] += value

        if self._ref_counts[ts_name] == 0:
            self._dead_ref_counts.add(ts_name)
        else:
            if ts_name in self._dead_ref_counts:
                self._dead_ref_counts.remove(ts_name)

    def _delete_ref_counts(self, ts_name):
        del self._ref_counts[ts_name]
        self._dead_ref_counts.remove(ts_name)

    def _log_info(self, msg, level=-1, offset=0):
        if level >= 0:
            self._lms._log_info("[Simulator] " + msg, level, offset)
        else:
            self._lms._log_info(
                "[Simulator] " + msg, self._debug_level, offset)
    
    @property
    def plot(self):
        return self._plot

    @plot.setter
    def plot(self, val):
        self._plot = val
