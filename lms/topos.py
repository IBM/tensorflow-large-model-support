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

from lms import util as ut


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

    def build(self):
        """Build a topological order
        """
        topo_sort = list(toposort.toposort(self._build_dependency_dict()))

        for i in range(0, len(topo_sort)):
            self._topo_sort[i] = topo_sort[i]

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
            dep_ops |= ut.fanins(src_op)
            dep_dict[src_op] = dep_ops

            next_ops = set()
            next_ops |= ut.fanouts(src_op)
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

    def reset(self):
        """Reset the topological sort
        """
        self._topo_sort = {}
        self._orders = {}

    def serialize_for(self, levels, min=1):
        """Serialize ops for multiple levels in the topological sort
        
        Args:
          levels: a list of strings of Python slicings
        """

        # build a list of indices
        xs = set()
        ys = [i for i in range(0, self.size)]
        for s in levels:
            sl = slice(*map(lambda x: int(x.strip())
                            if x.strip()
                            else None, s.split(':')))
            xs |= set(ys[sl])
        indices = sorted(list(xs))
        indices = [i for i in indices if i > min]

        prev_ops = set()
        prev_level = -1
        for i in indices:
            if (not prev_ops) and (prev_level + 1 == i):
                prev_ops = set()
            prev_ops = self._serialize_at(i, prev_ops)
            prev_level = i

        # rebuild the topo sort
        self.reset()
        self.build()

    def _serialize_at(self, order, prev_ops, rebuild=False):
        """Serialize ops at the same level in the topological sort
        """
        xs = self.get_ops(order)
        if xs is None:
            return set()
                
        # do not serialize levels including ops in the "/cond/" scope.
        cond_ops = {op for op in xs if "/cond/" in op.name}
        if len(cond_ops) > 0:
            return set()
        else:
            pass

        head_ops = {next(iter(xs))}
        tail_ops = xs - head_ops

        k_ops = head_ops
        if prev_ops:
            for op in k_ops:
                ut.add_control_inputs(
                    op,
                    prev_ops - set(op.control_inputs))

        for op in tail_ops:
            ut.add_control_inputs(
                op,
                k_ops - set(op.control_inputs))
            k_ops = {op}

        # rebuild the topo sort
        if rebuild:
            self.reset()
            self.build()
        else:
            pass

        return k_ops

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
