import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.contrib.graph_editor import util

import time
import queue as Queue
import lms.topos as topos


class LMS(object):
    def __init__(self, graph, optimizer_scope=None, excl_scopes=set(),
                 first_layer=None,
                 lb=1, ub=10000,
                 n_tensors=0,
                 n_cpu_threads=1,  # experimental feature
                 ssg_n_tensors=0,  # the number of tensors for second storage
                 ssg_id="1",
                 ssg_as_buffer=False,
                 fuse_swapins=False,
                 debug=False,
                 debug_level=1):
        if optimizer_scope is None:
            print("set the optimizer scope")
            return
        if first_layer is None:
            print("set the first layer name")
            return

        self.graph = graph
        self.optimizer_scope = optimizer_scope
        self.excl_scopes = excl_scopes
        self.first_layer = first_layer
        self.lb = lb  # lowerbound
        self.ub = ub  # upperbound
        self.n_tensors = n_tensors
        self.fuse_swapins = fuse_swapins

        # Operations with these types will be ignored
        self.atomic_types = ['Const', 'Mul', 'Add',
                             'Identity', 'Assign', 'VariableV2',
                             'Reshape', 'Shape', 'ShapeN']

        self.excl_ops = set()
        self.grad_ops = set()
        self.topo_sort = None
        self.debug = debug
        self.debug_level = debug_level

        self.ingpu_count = 0
        self.ssg_n_tensors = ssg_n_tensors
        self.ssg_id = ssg_id
        self.ssg_as_buffer = ssg_as_buffer
        self.incpu_count = 0
        self.n_cpu_threads = n_cpu_threads

        # for roundrobin scheduling
        self.currentSSG = False

    def find_ctrld_ops(self, fw_op, src_op, lower_b, upper_b):
        '''Find a control dependency operation using chain rules.
        '''
        if lower_b == 0:
            return (None, -1)

        result_ops = set()

        # offset ordering
        fw_order = self.topo_sort.get_order(fw_op)
        src_order = self.topo_sort.get_order(src_op)
        if src_order < 0:
            return (None, -1)
        range_ub = src_order - lower_b
        range_lb = max([src_order - upper_b, fw_order]) + 1

        self.log_info("Finding ctrld_op for {}, order {}, in range: [{}-{}]".format(
            src_op.name, src_order, range_lb, range_ub - 1), 1)

        ctrld_order = -1
        for i in reversed(range(range_lb, range_ub)):
            candidates = self.topo_sort.get_ops(i)
            if candidates:
                result_ops |= candidates
                ctrld_order = i
                break

        if result_ops:
            ctrld_op = next(iter(result_ops))
            return (ctrld_op, ctrld_order)
        else:
            return (None, -1)

    def run(self):
        self.log_info("Editing model for LMS")
        start_time = time.time()

        seed_ops = ge.filter_ops_from_regex(
            ge.make_list_of_op(self.graph), "^{}".format(self.first_layer))

        reachable_ops = set()
        for seed_op in seed_ops:
            reachable_ops |= set(ge.get_forward_walk_ops(seed_op))

        # gradient ops
        self.grad_ops = set(ge.filter_ops_from_regex(
            ge.make_list_of_op(self.graph), "^{}".format(
                self.optimizer_scope)))

        self.fw_reachable_ops = reachable_ops - self.grad_ops

        # exclusive ops
        for scope in self.excl_scopes:
            self.excl_ops |= set(ge.get_name_scope_ops(reachable_ops, scope))
        # atomic ops
        atomic_ops = {op for op in self.fw_reachable_ops
                      if op.type in self.atomic_types}
        self.excl_ops |= atomic_ops

        # build a topological sort
        self.topo_sort = topos.TOPOS(seed_ops, self.graph, self.grad_ops)
        self.topo_sort.build()

        self.do_action(seed_ops)

        self.log_info("Editing model for LMS, took: {} ms".format(
            (time.time()-start_time)/1000))
        if (self.ingpu_count > 0):
            self.log_info(
                "{} tensors will be swapped out(in) to(from) GPU device {}".format(
                    self.ingpu_count, self.ssg_id))
            if self.ssg_as_buffer:
                self.log_info(
                    "The GPU device {} is just used as a buffer.".format(
                        self.ssg_id))
        self.log_info(
            "{} tensors will be swapped out(in) to(from) the host".format(
                self.incpu_count))

    def do_action(self, src_ops):  # BFS
        open_set = Queue.Queue()
        closed_set = set()

        for op in src_ops:
            open_set.put(op)

        while not open_set.empty():
            src_op = open_set.get()

            # get next ops before the graph is changed
            next_ops = set()
            for t in src_op.outputs:
                frontier_ops = set(util.get_consuming_ops(t))
                next_ops |= frontier_ops - self.grad_ops

            # do action for src_op
            self.insert_swnodes(src_op)

            for op in next_ops:
                if op in closed_set:
                    continue
                if op not in open_set.queue:
                    open_set.put(op)

            closed_set.add(src_op)

    def insert_swnodes(self, src_op):
        self.log_info("Operation: {}".format(src_op), 2)

        # bypass exclusive ops
        if src_op in self.excl_ops:
            return

        for t in src_op.outputs:
            if self.n_tensors > 0:
                if self.ingpu_count > 0:
                    if (self.ingpu_count + self.incpu_count) >= self.n_tensors:
                        return  # swap enough
                else:
                    if (self.incpu_count) >= self.n_tensors:
                        return

            frontier_ops = set(util.get_consuming_ops(t))
            self.log_info("my frontier ops: {}".format(frontier_ops), 2)

            bw_frontier_ops = frontier_ops & self.grad_ops
            self.log_info("my bw frontier ops: {}".format(bw_frontier_ops), 2)

            if not bw_frontier_ops:
                continue

            # swap-out node
            swap_out_sgv = None
            src_out_idx = None
            # create swap-out node only if the current op has gradient or branch
            if bw_frontier_ops:  # TODO: check branch also
                ts0 = None
                with tf.device(self.get_ext_device(True)):
                    src_sgv = ge.sgv(src_op, graph=self.graph)
                    sample_op = next(iter(bw_frontier_ops))
                    ts = ge.filter_ts_from_regex(sample_op, src_op.name)
                    ts0 = ts[0]

                    swap_out = tf.identity(
                        ts0,
                        name='swap_out_' + src_op.name.replace('/', '_'))
                    swap_out_sgv = ge.sgv(swap_out.op, graph=self.graph)

                    # get output index
                    src_out_idx = src_sgv.output_index(ts0)

                    # Connect: src-node -> swap-out
                    connect_sgv(src_sgv, swap_out_sgv,
                                remap_outputs=True, idx=src_out_idx)
                    self.excl_ops.add(swap_out.op)
                    self.log_info("Tensor {} will be placed on {}".format(
                        ts0.name, self.get_ext_device()), 1)

                if self.ssg_as_buffer and ("GPU" in self.get_ext_device()):
                    with tf.device("/cpu:0"):
                        swap_out0 = tf.identity(
                            ts0,
                            name='swap_out0_' + src_op.name.replace('/', '_'))
                        swap_out0_sgv = ge.sgv(swap_out0.op, graph=self.graph)
                        connect_sgv(swap_out_sgv, swap_out0_sgv)
                        self.excl_ops.add(swap_out0.op)
                        swap_out_sgv = swap_out0_sgv

                # swap_in nodes
                # TODO: swap_in nodes for branches
                if self.fuse_swapins:
                    fuse_bw_frontier_ops = {
                        op for op in bw_frontier_ops
                        if self.topo_sort.get_order(op) > 0}
                    if fuse_bw_frontier_ops:
                        dev = "/cpu:0" if self.ssg_as_buffer else self.get_ext_device()
                        with tf.device(dev):
                            swap_in = tf.identity(
                                ts0,
                                name='swap_in_' + src_op.name.replace('/', '_'))
                            swap_in_sgv = ge.sgv(swap_in.op, graph=self.graph)

                        # Connect: swap_out -> swap_in
                        connect_sgv(swap_out_sgv, swap_in_sgv)
                        self.excl_ops.add(swap_in.op)

                        # reuse swap_in tensors
                        for op in fuse_bw_frontier_ops:
                            op_sgv = ge.sgv(op, graph=self.graph)
                            ts = ge.filter_ts_from_regex(op, src_op.name)
                            # Connect: swap_in -> dest
                            input_idx = op_sgv.input_index(ts[0])
                            connect_sgv(swap_in_sgv, op_sgv,
                                        remap_inputs=True, idx=input_idx)

                            self.log_info("{} reuses tensor {}".format(
                                op.name, ts[0].name), 1)

                        # control dependency -> swap_in
                        min_order = self.topo_sort.size + 1
                        earliest_op = None
                        for op in fuse_bw_frontier_ops:
                            order = self.topo_sort.get_order(op)
                            min_order = min(min_order, order)
                            earliest_op = op
                        if earliest_op:
                            self.add_ctrld(src_op, earliest_op, swap_in.op,
                                           self.lb, self.ub)
                        bw_frontier_ops -= fuse_bw_frontier_ops

                for dest_op in bw_frontier_ops:
                    dest_sgv = ge.sgv(dest_op, graph=self.graph)
                    ts = ge.filter_ts_from_regex(dest_op, src_op.name)

                    dev = "/cpu:0" if self.ssg_as_buffer else self.get_ext_device()
                    with tf.device(dev):
                        swap_in = tf.identity(
                            ts[0],
                            name='swap_in_' + dest_op.name.replace('/', '_'))
                        swap_in_sgv = ge.sgv(swap_in.op, graph=self.graph)

                    # Connect: swap_out -> swap_in
                    connect_sgv(swap_out_sgv, swap_in_sgv)

                    # Connect: swap_in -> dest
                    input_idx = dest_sgv.input_index(ts[0])
                    connect_sgv(swap_in_sgv, dest_sgv,
                                remap_inputs=True, idx=input_idx)
                    self.excl_ops.add(swap_in.op)
                    self.log_info("Consuming op {} swaps in {}".format(
                        dest_op.name, ts[0].name), 1)

                    # control dependency -> swap_in
                    self.add_ctrld(src_op, dest_op, swap_in.op,
                                   self.lb, self.ub)

    def get_ext_device(self, update=False):
        return self.roundrobin_ext_device(update)

    def roundrobin_ext_device(self, update=False):
        # choose GPU device first
        if self.currentSSG:  # previous one is GPU memory, now use host memory
            if update:
                self.incpu_count = self.incpu_count + 1
                self.currentSSG = False
                return "/cpu:{}".format(self.incpu_count % self.n_cpu_threads)
            else:
                # get only
                return "/device:GPU:{}".format(self.ssg_id)
        else:  # previous one is host memory, now use GPU or host memory
            if update:
                # if having slots for GPU, use GPU
                if (self.ingpu_count < self.ssg_n_tensors):
                    self.ingpu_count = self.ingpu_count + 1
                    self.currentSSG = True
                    return "/device:GPU:{}".format(self.ssg_id)
                else:  # otherwise, use host
                    self.incpu_count = self.incpu_count + 1
                    self.currentSSG = False
                    return "/cpu:{}".format(
                        self.incpu_count % self.n_cpu_threads)
            else:
                # get only
                return "/cpu:{}".format(self.incpu_count % self.n_cpu_threads)

    def add_ctrld(self, fw_op, bw_op, swapin_op, lb, ub):
        re = self.find_ctrld_ops(fw_op, bw_op, lb, ub)
        if re[0]:
            ge.add_control_inputs(swapin_op, re[0])
            self.log_info(
                "Control dependency op {},  order: {}".format(
                    re[0].name, re[1]), 1)

    def log_info(self, message, level=0):
        if level == 0 or (self.debug and self.debug_level >= level):
            print("[LMS][{}] {}".format(level, message))


def connect_sgv(src_sgv, dest_sgv,
                remap_inputs=False, remap_outputs=False,
                idx=None, disconnect_first=False):
    if remap_outputs:
        src_sgv = src_sgv.remap_outputs([idx])
    if remap_inputs:
        dest_sgv = dest_sgv.remap_inputs([idx])

    ge.connect(src_sgv, dest_sgv, disconnect_first)
