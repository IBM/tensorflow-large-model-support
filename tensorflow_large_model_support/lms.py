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
"""LMS
"""
import tensorflow as tf

import time
from tensorflow_large_model_support import topos
from tensorflow_large_model_support import util as ut


class LMS(tf.keras.callbacks.Callback, tf.train.SessionRunHook):
    """LMS class for Large Model Support (LMS).

    The `LMS` object statically modifies a model by swapping its tensors
    to the host so that the model can be trained with the limited memory
    of GPUs.

    Tensors those are generated by forward operations and consumed by
    backward operations are candidates for swapping. The `LMS` object will
    automatically find these tensors.

    Swapping is done by cutting the link between two operations whose
    topological-sort distance between them is greater than a given
    `threshold`, then replacing the link by inserting `identity`
    operations on the host. In theory, this procedure does not have any
    effect on the training convergence as well as inference task.
    """
    def __init__(self, graph=None,
                 swapout_threshold=-1,
                 swapin_groupby=0,
                 swapin_ahead=-1,
                 sync_mode=0,
                 serialization=[],
                 debug=False,
                 debug_level=1,
                 cpu_device="/cpu:0"):
        """Create an LMS object to edit the graph for supporting large model.

        Args:
          graph: the graph we will modify for LMS. This should be the graph of
            user-defined neural network.
          swapout_threshold: if the topological-sort distance between the
            consuming operation and generating operation of a tensor is
            greater (>) than `swapout_threshold`, then trigger swapping the
            tensor. Default `-1` (auto mode).
          swapin_groupby: consuming operations whose distances among them are
            within `swapin_groupby` share the same swap-in operation.
            Default `0`.
          swapin_ahead: lower-bound value for LMS. A tensor will be swapped in
            during the backward phase at least `swapin_ahead` nodes before it
            in the graph. Default `-1` (auto mode).
          sync_mode: whether do synchronization between data transfer and
            kernel computation or not. Four modes: `0` turn off. `1` sync for
            only swap-out ops. `2` sync for only swap-in ops. `3` sync for both
            swap-out and swap-in ops. Default `0`.
          serialization: serialize operations at the same level in the
            topological sort. This option accepts a list of Python slicing
            string in which each slicing represents level indices in the
            topological sort. E.g. [1, 3:5, 7] means levels 1, 3, 4, 5 and 7
            are serialized. Default `[]` (turn off).
          debug: debug mode for LMS. Default `False`.
          debug_level: debug level for LMS (1 or 2). Default `1`.
          cpu_device: the device we would like swap tensors to.
        """
        self._graph = graph
        self._swapout_threshold = swapout_threshold
        self._swapin_groupby = swapin_groupby
        self._swapin_ahead = swapin_ahead
        if sync_mode not in {0, 1, 2, 3}:
            raise ValueError('Invalid value for sync_mode')
        else:
            self._sync_mode = sync_mode
        self._serialization = serialization
        self._cpu_device = cpu_device
        self._debug = debug
        self._debug_level = debug_level

        # optional parameters
        self._excl_output_by_scopes = set()
        self._excl_output_by_types = set()
        self._excl_input_by_scopes = set()
        self._excl_input_by_types = set()
        self._incl_output_by_scopes = set()
        self._incl_output_by_types = set()
        self._incl_input_by_scopes = set()
        self._incl_input_by_types = set()
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/graph_util_impl.py
        self._variable_ops = {
            "Assign",
            "AssignAdd",
            "AssignSub",
            "Queue",
            "ScatterAdd",
            "ScatterSub",
            "ScatterUpdate",
            "TruncatedNormal",
            "Variable",
            "VariableV2",
        }
        # variable ops: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/kernels/variable_ops.cc
        self._unused_types = {
            # input data
            'Const', 'Identity', 'Read',
            'Placeholder', 'PlaceholderWithDefault',
            # variable ops
            'VarHandleOp', 'VarIsInitializedOp', 'VariableShape',
            'ReadVariableOp', 'AssignVariableOp',
            'AssignAddVariableOp', 'AssignSubVariableOp'
            'ResourceGather', 'ResourceScatterAdd',
            'ResourceScatterSub', 'ResourceScatterMul',
            'ResourceScatterDiv', 'ResourceScatterMin',
            'ResourceScatterMax', 'ResourceScatterUpdate',
            'ResourceScatterNdUpdate', 'ResourceScatterNdAdd',
            # data filling
            'Fill', 'Range', 'RandomUniform'}

        # a topological sort
        self._topo_sort = None
        # store information to be used to adding control dependencies
        self._swap_ops = []  # [(src_op, swapout_op, swapin_op, dest_op)]
        # control outputs topology
        self._control_outputs = None
        self._version = None

    @property
    def excl_output_by_scopes(self):
        return self._excl_output_by_scopes

    @excl_output_by_scopes.setter
    def excl_output_by_scopes(self, val):
        self._excl_output_by_scopes = val

    @property
    def excl_output_by_types(self):
        return self._excl_output_by_types

    @excl_output_by_types.setter
    def excl_output_by_types(self, val):
        self._excl_output_by_types = val

    @property
    def excl_input_by_scopes(self):
        return self._excl_input_by_scopes

    @excl_input_by_scopes.setter
    def excl_input_by_scopes(self, val):
        self._excl_input_by_scopes = val

    @property
    def excl_input_by_types(self):
        return self._excl_input_by_types

    @excl_input_by_types.setter
    def excl_input_by_types(self, val):
        self._excl_input_by_types = val

    @property
    def incl_output_by_scopes(self):
        return self._incl_output_by_scopes

    @incl_output_by_scopes.setter
    def incl_output_by_scopes(self, val):
        self._incl_output_by_scopes = val

    @property
    def incl_output_by_types(self):
        return self._incl_output_by_types

    @incl_output_by_types.setter
    def incl_output_by_types(self, val):
        self._incl_output_by_types = val

    @property
    def incl_input_by_scopes(self):
        return self._incl_input_by_scopes

    @incl_input_by_scopes.setter
    def incl_input_by_scopes(self, val):
        self._incl_input_by_scopes = val

    @property
    def incl_input_by_types(self):
        return self._incl_input_by_types

    @incl_input_by_types.setter
    def incl_input_by_types(self, val):
        self._incl_input_by_types = val

    def run(self, graph=None):
        """Edit the graph by adding swapin and swapout ops.

        Swapin and swapout ops are in the host.

        The graph is modified in-place.

        Return:

          a set of added ops.
        """
        if graph:
            self._graph = graph

        if not self._graph:
            raise ValueError('The dataflow graph is required but has not been'
                             ' provided.')
        self._version = self._graph.version

        self._log_info("Editing model for LMS")
        start_time = time.time()

        all_ops = self._graph.get_operations()
        n_edges = 0
        for op in all_ops:
            if 'lms/swap' in op.name:
                self._log_info('This model has already been updated with LMS '
                               'swap operations. LMS will not re-process it.')
                return
            for op1 in ut.fanouts(op):
                n_edges += 1
        self._log_info(
            "The graph has {} vertices and {} edges.".format(
                len(all_ops), n_edges)
        )

        # build a control output topology
        self._control_outputs = ut.build_control_outputs(self._graph)

        # build a topological sort
        self._topo_sort = topos.TOPOS(self._graph)
        self._topo_sort.build()
        self._log_info("Original categorized topological sort has {} levels".format(
            self._topo_sort.size))

        # serialize the topological sort if enabled
        if self._serialization:
            init_ops = self._force_variable_initialization(
                {op for op in all_ops
                 if op.type in {'Variable', 'VariableV2'}})
            m = 0
            if len(init_ops) > 0:
                # when are all variables initialized?
                self._topo_sort.reset()
                self._topo_sort.build()
                m = max({self._get_level(op) for op in init_ops})
            self._log_info("Serialize the topological sort for levels: " +
                           "{}".format(self._serialization), offset=2)
            self._topo_sort.serialize_for(self._serialization, min=m)
            self._log_info("New categorized topological sort has {} levels".format(
                self._topo_sort.size), offset=2)
            self._rebuild_control_outputs(True)  # force rebuilding

        if self._debug and self._debug_level >= 1:
            for i in range(0, self._topo_sort.size):
                self._log_info("[{}]: {}".format(
                    i, [(op.name, op.type)
                        for op in self._get_ops_by_level(i)]), 1)

        # roughly estimate swapin_threshold in auto mode
        if self._swapout_threshold < 0:
            self._log_info("Use auto mode for setting swapout_threshold")
            self._swapout_threshold = self._topo_sort.size//2

        self._print_configuration()
        self._log_histogram()  # build a histogram of distance

        # swapping tensors
        # inclusive source ops
        self._incl_src_ops = self._filter_scopes_and_types(
            all_ops, self._incl_output_by_scopes, self._incl_output_by_types)
        # inclusive dest ops
        self._incl_dest_ops = self._filter_scopes_and_types(
            all_ops, self._incl_input_by_scopes, self._incl_input_by_types)
        # exclusive source ops
        self._excl_src_ops = self._filter_scopes_and_types(
            all_ops, self._excl_output_by_scopes,
            self._excl_output_by_types | self._unused_types | self._variable_ops)
        # exclusive dest ops
        self._excl_dest_ops = self._filter_scopes_and_types(
            all_ops, self._excl_input_by_scopes,
            self._excl_input_by_types | self._unused_types | self._variable_ops)
        # add swapout/swapin ops
        self._rewrite_for_swapping()
        # make sure we are working with the latest control outputs topology
        self._rebuild_control_outputs()
        # add ctrl. dependencies
        if self._swap_ops:
            if self._sync_mode == 0:  # async mode
                self._add_control_dependencies()
            else:  # sync mode
                self._sync_ops(self._sync_mode)

        # print log information
        n_swapout_ops = len({op[1] for op in self._swap_ops})
        n_swapin_ops = len({op[2] for op in self._swap_ops})
        self._log_info(
            "Added {} operations to the model".format(
                n_swapout_ops + n_swapin_ops) +
            " ({} swap-out operations and {} swap-in operations)".format(
                n_swapout_ops, n_swapin_ops))
        self._log_info("Editing model for LMS, took: {} ms".format(
            (time.time()-start_time)*1000))

    def _force_variable_initialization(self, vars):
        """
        For each variable, it should be assigned a value before
        the other ops consumming the variable.
        But, by analyzing a graph statically, we have no dependency
        bewteen a Assign operation and other read/write operations.
        So we need to create dependency explicitly.
        ```
        VariableV2:
          input: None
          output: reading ops, Assign, writing ops, ...
        ```
        """
        init_ops = set()
        for x in vars:
            outs = ut.fanouts(x)
            assign_ops = {op for op in outs if op.type == "Assign"}

            # initialization op is the first Assign op
            m = min({self._get_level(op) for op in assign_ops})
            first_assign_ops = {
                op for op in assign_ops
                if self._get_level(op) == m
            }
            if len(first_assign_ops) != 1:
                self._log_info("Variable: {}".format(x.name))
                self._log_info("Fanout ops: {}".format(
                    {(op.name, op.type, self._get_level(op))
                     for op in outs}))
                raise ValueError(
                    'Could not find the first assignment for {}'.format(x.name))

            # initialize the variable first
            for y in outs - first_assign_ops:
                self._add_control_inputs(y, first_assign_ops)
            init_ops |= first_assign_ops

        return init_ops

    def _rewrite_for_swapping(self):
        """Add swapin and swapout ops for ops that are reachable from `all_ops`.

        Args:
          all_ops: a list of `tf.Operation`
        """
        cand_ops = set(self._graph.get_operations())

        # filter by source operations
        if self._incl_src_ops:
            # if inclusive mode is enabled,
            # only proceed included ops
            cand_ops &= self._incl_src_ops
        cand_ops -= self._excl_src_ops

        for op in cand_ops:
            self._insert_swap_nodes(op)

    def _sync_ops(self, sync_mode):
        """TODO: write comment
        """
        src, sout, sin, dest = 0, 1, 2, 3
        # sync for swap-out ops
        if sync_mode in {1, 3}:
            souts = {op[sout] for op in self._swap_ops}
            src_sout = {(op[src], op[sout]) for op in self._swap_ops}
            for src, sout in src_sout:
                self._sync_swapout(src, sout, souts)

        # sync for swap-in ops
        if sync_mode in {2, 3}:
            sins = {op[sin] for op in self._swap_ops}
            sin_dest = {(op[sin], op[dest]) for op in self._swap_ops}
            for sin, dest in sin_dest:
                self._sync_swapin(sin, dest, sins)

    def _sync_swapout(self, src, sout, souts):
        """TODO: write comment
        Need to update control outputs topology before calling
        this method
        """
        self._log_info("Do synchronization for the swap-out " +
                       "{}.".format(sout.name), 1)

        fs = ut.fanouts(src) | self._get_control_outputs(src)
        fs -= souts  # self-loop and cycles among swap-outs

        # avoid duplication
        for op in fs:
            if sout in op.control_inputs:
                fs.remove(op)
            if sout in ut.fanins(op):
                fs.remove(op)

        for op in fs:
            self._add_control_inputs(op, sout)

    def _sync_swapin(self, sin, dest, sins):
        """TODO: write comment
        """
        self._log_info("Do synchronization for the swap-in " +
                       "{}.".format(sin.name), 1)

        fs = ut.fanins(dest) | set(dest.control_inputs)
        fs -= sins  # self-loop and cycles among swap-ins
        fs -= (set(sin.control_inputs) | ut.fanins(sin))  # avoid duplication
        fs -= (ut.fanouts(sin) | self._get_control_outputs(sin))  # cycles
        self._add_control_inputs(sin, fs)

    def _groupby(self, ops, limit=5):
        """Group `ops` into groups so that topological distance between
        two consecutive ops in a group is within `limit`.

        Args:
          ops: a set of `tf.Operation`.
          limit: a threshold

        Return:
          A list of sets of `tf.Operation`.
        """
        ops_levels = [(op, self._get_level(op)) for op in ops]
        x = sorted([i[1] for i in ops_levels])
        xs = [(i, i) for i in x]

        ys = [xs[0]]
        for i in range(1, len(xs)):
            last = ys[-1]
            curr = xs[i]
            if (curr[0] - last[1] <= limit):
                last = (last[0], curr[1])
                ys[-1] = last
            else:
                ys.append(curr)

        zs = []
        for y in ys:
            gs = set()
            gs = {op[0]
                  for op in ops_levels
                  if (op[1] >= y[0] and op[1] <= y[1])}
            zs.append(gs)
        return zs

    def _insert_swap_nodes(self, src_op):
        """Insert swapin and swapout ops for the given operation into the graph.

        This method does an in-place modification to the graph.

        Args:
          src_op: a `tf.Operation`
        """
        self._log_info("Operation: {}".format(src_op), 2)
        if self._debug and self._debug_level >= 2:
            for ts in src_op.outputs:
                self._log_info("Output tensor: {}".format(ts), 2, 1)
                for op in ts.consumers():
                    self._log_info("Consuming op: {}".format(op.name), 2, 2)

        # obtain candidates
        ts_dests = {}
        src_op_level = self._get_level(src_op)
        for ts in src_op.outputs:
            # filter by tensor shape
            # do not swap 1-dimension or unknown shape tensors.
            ndims = ts.shape.ndims
            if ndims is None or ndims <= 1:
                continue

            # filter by topological distance
            # candidates are ops whose distance to `src_op` is
            # greater than threshold
            cands = {
                op
                for op in ts.consumers()
                if self._get_level(op) - src_op_level > self._swapout_threshold
            }

            # filter by dest operations
            if self._incl_dest_ops:
                cands &= self._incl_dest_ops
            cands -= self._excl_dest_ops

            if cands:
                ts_dests[ts] = cands
            else:
                continue

        if ts_dests:
            self._log_info("Operation: {}".format(src_op.name), 1)
        else:
            return

        # rewrite the graph
        for ts in ts_dests:
            # group near candidates by topological distance
            dests_grp = self._groupby(ts_dests[ts], self._swapin_groupby)

            # insert swapout and swap-in ops
            sout, sin_dest = self._insert_swap_nodes_for_ts(
                src_op, ts, dests_grp)

            # keep newly added ops
            for sin, dest in sin_dest:
                self._swap_ops.append((src_op, sout, sin, dest))

    def _insert_swap_nodes_for_ts(self, src_op, ts, targets):
        """Insert swapin and swapout ops for the given tensor into the graph.

        This method does an in-place modification to the graph.

        Args:
          src_op: a `tf.Operation`.
          ts: a `tf.Tensor`, an output of `src_op`.
          targets: a list of sets of consuming ops of `src_op`.

        Return:
          A tuple of a swap-out op and a set of pairs of a consuming op and
          a swap-in op.
        """
        # create a swap_out node
        swapout_op = self._add_swapout(src_op, ts)
        self._add_control_inputs(swapout_op, src_op, offset=4)

        # create swap_in nodes
        sin_dest = set()
        for dest_ops in targets:
            # swap_in op
            swapin_op = self._add_swapin(swapout_op, dest_ops, ts)
            # for control dependency
            ops_levels = [(op, self._get_level(op)) for op in dest_ops]
            x = sorted([i[1] for i in ops_levels])[0]  # the earliest op
            dest_op = [op[0] for op in ops_levels if op[1] == x][0]
            sin_dest.add((swapin_op, dest_op))

        return (swapout_op, sin_dest)

    def _add_swapout(self, src_op, ts0):
        """Add a swapout operation to the graph to swap out the output tensor `ts0`
        of the operation `src_op`.

        This method does an in-place modification to the graph.

        Example: the graph before and after this method invoked.
        ```
        Before
          (src_op) -> |ts0| -> (dest_op)

        After:
          (src_op) -> |ts0| -> (swapout_op)
          |ts0| -> (dest_op)
        ```

        Args:
          src_op: a `tf.Operation` that produces the tensor `ts0`.
          ts0: a output `tf.Tensor` of `src_op` being swapped out.

        Return:
          A `tf.Operation` newly added to the graph.
        """
        with tf.device(self._cpu_device):
            swap_out = tf.identity(
                ts0,
                name="lms/swapout_{}".format(
                    ts0.name.replace("/", "_").replace(":", "_")))

        # Connect: src-node -> swap-out
        ut.reroute_input(ts0, swap_out.op.inputs[0], swap_out.op)
        self._log_info("Swap-out: Tensor {} (shape: {})".format(
            ts0.name, ts0.shape), 1, 2)
        self._log_info("Swap-out operation: {}".format(
                swap_out.op.name), 1, 4)
        self._log_info("Connect: {} => {}".format(
            src_op.name, swap_out.op.name), 1, 4)

        return swap_out.op

    def _add_swapin(self, swapout_op, dest_ops, ts0):
        """Add a swapin operation to the graph. The swapin ops reads
        the output tensor of `swapout_op` and passes it to `dest_ops`,
        replacing the input tensors `ts0` of `dest_ops`.

        This method does an in-place modification to the graph.

        Example: the graph before and after this method invoked.
        ```
        Before
          |ts0| -> (swapout_op)
          |ts0| -> (dest_op)

        After:
          |ts0| -> (swapout_op) -> (swapin_op) -> (dest_op)
        ```

        Args:
          swapout_op: a `tf.Operation` that swapped out the tensor `ts0`.
          dest_ops: a set of `tf.Operation` that will consume the output
                    tensor of `swapout_op`.
          ts0: a `tf.Tensor` being the original input tensor of `dest_op`.

        Return:
          A `tf.Operation` newly added to the graph.
        """
        self._log_info("Swap-in: Tensor {} (shape: {})".format(
            ts0.name, ts0.shape), 1, 2)

        with tf.device(self._cpu_device):
            swap_in = tf.identity(
                ts0,
                name="lms/swapin_{}".format(
                    ts0.name.replace("/", "_").replace(":", "_")))

        # Connect: swap_out -> swap_in
        ut.reroute_input(swapout_op.outputs[0],
                         swap_in.op.inputs[0], swap_in.op)
        self._log_info("Swap-in operation: {}".format(
            swap_in.op.name), 1, 4)
        self._log_info("Connect: {} => {}".format(
            swapout_op.name, swap_in.op.name), 1, 4)

        # Connect: swap_in -> dest_ops
        for dest_op in dest_ops:
            ut.reroute_input(swap_in.op.outputs[0], ts0, dest_op)
            self._log_info("Connect: {} => {}".format(
                swap_in.op.name, dest_op.name), 1, 4)
        return swap_in.op

    def _add_control_dependencies(self):
        """Add control dependency operations for all consuming ops.
        """
        if (self._swapin_ahead < 0):
            self._sequential_strategy()
        else:
            # Use the user-defined ahead
            for src, _, sin, dest in self._swap_ops:
                self._add_control_dependency(
                    src, dest, sin, self._swapin_ahead)

    def _sequential_strategy(self):
        """This strategy is to make sure swapins are done in
        a sequential way with respect to the topological levels of
        consuming ops.
        """
        src, sin, dest = 0, 2, 3  # indices
        # sort by destination op's level
        x = sorted(self._swap_ops,
                   key=lambda ops: self._get_level(ops[dest]))

        # a fixed setting for the first swapins.
        x0_dest_level = self._get_level(x[0][dest])
        ahead = 3
        k = 0
        for i in range(0, len(x)):
            if self._get_level(x[i][dest]) == x0_dest_level:
                self._add_control_dependency(
                    x[i][src], x[i][dest], x[i][sin], ahead)
                k = i
            else:
                break

        lb = x0_dest_level
        last_level = lb
        for i in range(k+1, len(x)):
            curr_level = self._get_level(x[i][dest])
            if curr_level != last_level:
                lb = last_level
                last_level = curr_level
            ahead = curr_level - lb
            self._add_control_dependency(
                x[i][src], x[i][dest], x[i][sin], ahead)

    def _add_control_dependency(self, src_op, dest_op, swapin_op, ahead):
        """Find and add a control dependency to the graph.

        This method does an in-place modification to the graph.

        Args:
          src_op: a `tf.Operation`.
          dest_op: a `tf.Operation`.
          swapin_op: a `tf.Operation`.
        """
        self._log_info("Find a control dependency for swapping in" +
                       " {} (level {})".format(dest_op.name,
                                               self._get_level(dest_op)), 1)

        re = self._search_by_level(src_op, dest_op, ahead)

        ctrld_op = re[0]
        if ctrld_op:
            self._add_control_inputs(swapin_op, ctrld_op)
        else:
            self._log_info(
                "No control dependency op found for the swap-in " +
                "{}.".format(swapin_op.name), 1)
            # do synchronization
            swapins = {op[2] for op in self._swap_ops}
            self._sync_swapin(swapin_op, dest_op, swapins)

    def _search_by_level(self, src_op, dest_op, distance):
        """Find a control dependency operation using topological sort.

        Args:
          src_op: a `tf.Operation` that has a tensor swapped out.
          dest_op: a `tf.Operation` that consumes a tensor swapped in.
          distance: an `integer`. The distance in the topological sort
            between `dest_op` and a candidate for control dependency ops
            must be greater than `distance`.

        Return:
          A tuple of (`tf.Operation`, an `integer`). The first item is
          the control dependency operation that triggers swapping in the input
          tensor of `dest_op`. The second item is the level of the control
          dependency operation in the topological sort.
        """
        result_ops = set()

        range_ub = self._get_level(dest_op) - distance
        range_lb = self._get_level(src_op) + 1

        ctrld_level = -1
        for i in reversed(range(range_lb, range_ub)):
            candidates = self._get_ops_by_level(i)
            # on the longest path from `src_op` to `dest_op`
            candidates = {
                op
                for op in candidates
                if self._is_reachable(op, dest_op, False)
            }
            # not in a condition scope
            candidates = {
                op
                for op in candidates
                if "/cond/" not in op.name
            }
            if candidates:
                result_ops |= candidates
                ctrld_level = i
                break

        if result_ops:
            ctrld_op = next(iter(result_ops))
            return (ctrld_op, ctrld_level)
        else:
            return (None, -1)

    def _is_on_longest_path(self, src_op, dest_op, op):
        """Check if `op` is on the longest path from `src_op` to `dest_op`.
        """
        if not self._is_reachable(src_op, op, True):
            return False
        if not self._is_reachable(op, dest_op, True):
            return False
        return True

    def _filter_scopes_and_types(self, within_ops, scopes, types):
        """Filter out ops that are not in `scopes` and not of `types`.

        Args:
          within_ops: an object convertible to a list of `tf.Operation`.
          scopes: a list of scope path.
          types: a list of tf.DataType.
        Return:
          A set of `tf.Operation`.
        """
        ops = set()
        for scope in scopes:
            ops |= {op for op in within_ops
                    if op.name.startswith(scope)}
        ops |= {op
                for op in within_ops
                if op.type in types}
        return ops

    def _is_reachable(self, src_op, dest_op, via_longest=True):
        """Check whether there exists a path from src_op to dest_op.
        The path's length must be equal to the distance from
        `src_op` to `dest_ops`. In other words, we search for the
        longest path from `src_op` to `dest_op`.

        Args:
          src_op: a starting operation.
          dest_op: a destination operation.

        Return:
          True/False.
        """
        src_level = self._get_level(src_op)
        dest_level = self._get_level(dest_op)

        fanouts = ut.fanouts(src_op)
        for l in range(src_level+1, dest_level):
            latest_ops = self._get_ops_by_level(l)
            latest_ops &= fanouts

            if not via_longest:
                if dest_op in latest_ops:
                    return True
            else:
                fanouts = set()

            for op in latest_ops:
                fanouts |= ut.fanouts(op)

        if dest_op in fanouts:
            return True
        else:
            return False

    def _get_level(self, op):
        """Return the topological level of an operation.

        Args:
          op: a `tf.Operation`.

        Return:
          an integer.
        """
        ret = self._topo_sort.get_level(op)
        if ret is None:
            return -1
        else:
            return ret

    def _get_ops_by_level(self, level):
        """Return a set of ops with the given level.

        Args:
          level: an integer.

        Return:
          a set of `tf.Operation`
        """
        return self._topo_sort.get_ops(level).copy()

    def _log_info(self, message, level=0, offset=0):
        """Log debug information.

        Args:
          message: a formatted string.
          level: an `integer`.
        """
        if level == 0 or (self._debug and self._debug_level >= level):
            # Use tf.logging.info instead of print, since print
            # is not thread safe, which can break tests.
            tf.logging.info("[LMS][{}] ".format(level) +
                            ' '*offset +
                            "{}".format(message))

    def _print_configuration(self):
        """Print configuration information about LMS.
        """
        self._log_info("swapout_threshold: {}".format(self._swapout_threshold))
        self._log_info("swapin_groupby: {}".format(self._swapin_groupby))
        if self._sync_mode == 1:
            self._log_info(
                "sync_mode was turned on for swap-out ops")
            self._log_info("swapin_ahead: {}".format(
                "auto mode" if self._swapin_ahead < 0 else self._swapin_ahead))
        elif self._sync_mode == 2:
            self._log_info(
                "sync_mode was turned on for swap-in ops. " +
                "swapin_ahead will be ignored")
        elif self._sync_mode >= 3:
            self._log_info(
                "sync_mode was turned on for both swap-out and swap-in ops. " +
                "swapin_ahead will be ignored")
        elif self._sync_mode == 0:
            self._log_info("swapin_ahead: {}".format(
                "auto mode" if self._swapin_ahead < 0 else self._swapin_ahead))
        else:
            pass

    def _get_control_outputs(self, op):
        """Return a set of control outputs of an operation.
        """
        if op in self._control_outputs:
            return self._control_outputs[op]
        else:
            return set()

    def _rebuild_control_outputs(self, force=False):
        """Rebuild the control_outputs dictionary if there are ops added
        to the graph.
        """
        if force or (self._version != self._graph.version):
            self._control_outputs = ut.build_control_outputs(self._graph)

    def _update_control_outputs(self, ops, couts):
        """Update control_output sets for operations in `ops`.
        """
        for op in ops:
            if op in self._control_outputs:
                self._control_outputs[op] |= couts
            else:
                self._control_outputs[op] = couts

    def _add_control_inputs(self, op, cops, offset=0):
        """Add control dependencies from `cops` to `op`.

        Args:
          op: a tf.Operation to which the control inputs are added.
          cops: an object convertible to a list of `tf.Operation`.
        Raises:
          TypeError: if op is not a tf.Operation
          ValueError: if any cop in cops is already a control input of op.
        """

        ut.add_control_inputs(op, cops)
        flag = True
        try:
            _ = iter(cops)
        except Exception:  # pylint: disable=broad-except
            flag = False

        if not flag:
            self._update_control_outputs({cops}, {op})
            self._log_info(
                "Control dependency: {} (level {}) => {} (level {})".format(
                    cops.name, self._get_level(cops),
                    op.name, self._get_level(op)),
                1, offset)
        else:
            self._update_control_outputs(cops, {op})
            for cop in cops:
                self._log_info(
                    "Control dependency: " +
                    "{} (level {}) => {} (level {})".format(
                        cop.name, self._get_level(cop),
                        op.name, self._get_level(op)),
                    1, offset)

    def _log_histogram(self):
        """Log a histogram of distances for edges emanated from `all_ops`.

        Args:
          all_ops: a set of `tf.Operation` to traverse

        Return:
          A dictionary of distance and frequency.
        """
        hist = {}
        all_ops = self._graph.get_operations()
        import tempfile
        _, f_name = tempfile.mkstemp()
        f = open(f_name, "w")
        f.write("#distance\tfrequency\n")
        for op1 in all_ops:
            for op2 in ut.fanouts(op1):
                dist = self._get_level(op2) - self._get_level(op1)
                if dist in hist:
                    hist[dist] += 1
                else:
                    hist[dist] = 1

        for v in sorted(hist):
            f.write("{}\t{}\n".format(v, hist[v]))
        f.close()
        self._log_info(
            "A histogram of distances was written to {}".format(f_name))
        return hist

    # Implementation of begin from tf.train.SessionRunHook
    def begin(self):
        """Implementation of the begin method which is inherited from
        tf.train.SessionRunHook.
        """
        self.run(tf.get_default_graph())

    # from tf.keras.callbacks.Callback
    def set_model(self, model):
        """Implementation of the set_model method which is inherited from
        tf.keras.callbacks.Callback.
        """
        self.run(tf.get_default_graph())
