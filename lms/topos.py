# (C) Copyright IBM Corp. 2018. All Rights Reserved.
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

"""TOPOS
"""
from six.moves import queue as Queue
import toposort

from tensorflow.contrib.graph_editor import util


class TOPOS(object):
    """TOPOS class builds a topological order from the computational graph.
    """
    def __init__(self, seed_ops):
        """Create a TOPOS object.

        Args:
          seed_ops: a list of `tf.Operation`.
        """
        self._seed_ops = seed_ops

        self._topo_sort = {}
        self._orders = {}

        # input data
        self._excl_types = {'Const', 'Placeholder', 'PlaceholderWithDefault'}
        self._excl_types |= {'VariableV2'}  # learnable parameters
        # variable ops
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/kernels/variable_ops.cc
        self._excl_types |= {'Read', 'Assign', 'VarHandleOp',
                             'VarIsInitializedOp', 'VariableShape',
                             'ReadVariableOp', 'AssignVariableOp',
                             'AssignAddVariableOp', 'AssignSubVariableOp',
                             'ResourceGather', 'ResourceScatterAdd',
                             'ResourceScatterSub', 'ResourceScatterMul',
                             'ResourceScatterDiv', 'ResourceScatterMin',
                             'ResourceScatterMax', 'ResourceScatterUpdate',
                             'ResourceScatterNdUpdate', 'ResourceScatterNdAdd'}
        self._excl_types |= {'Fill', 'Range', 'RandomUniform'}
        self._excl_types |= {'Identity'}  # read data
    
    def build(self):
        """Build a topological order
        """
        topo_sort = list(toposort.toposort(self._build_dependency_dict()))

        k = 0
        for i in range(0, len(topo_sort)):
            # remove ops related to variable initialization
            xs = {op for op in topo_sort[i]
                  if op.type not in self._excl_types}
            if xs:
                self._topo_sort[k] = xs
                k += 1 
            else:
                continue

        # build a dict of (op, order)
        self._build_order_dict()

    def _build_dependency_dict(self):
        """Build a dictionary of dependencies among nodes.
        """
        open_set = Queue.Queue()
        closed_set = set()

        dep_dict = {}
        for op in self._seed_ops:
            open_set.put(op)

        while not open_set.empty():
            src_op = open_set.get()

            # do action for src_op
            dep_ops = set(src_op.control_inputs)
            for t in src_op.inputs:
                dep_ops |= set(util.get_generating_ops(t))
            dep_dict[src_op] = dep_ops

            next_ops = set()
            for t in src_op.outputs:
                next_ops |= set(util.get_consuming_ops(t))
            for op in next_ops:
                if op in closed_set:
                    continue
                if op not in open_set.queue:
                    open_set.put(op)

            closed_set.add(src_op)

        return dep_dict

    def _build_order_dict(self):
        """Build a dictionary to quickly find an order of an ops.
        """
        for order, dep_ops in self._topo_sort.items():
            for op in dep_ops:
                self._orders[op] = order

    def get_order(self, op):
        """Return the order of an operation.

        Args:
          op: a `tf.Operation`.

        Return:
          An integer.
        """
        if op in self._orders:
            return self._orders[op]
        else:
            return None

    def get_ops(self, order):
        """Return a set of ops with the same order.

        Args:
          order: an integer.

        Return:
          A set of `tf.Operation`
        """
        if order in self._topo_sort:
            return self._topo_sort[order]
        else:
            return None

    @property
    def size(self):
        """The number of orders in the topological order.
        """
        return len(self._topo_sort)
