import queue as Queue
from toposort import toposort as tps

from tensorflow.contrib.graph_editor import util


class TOPOS(object):
    def __init__(self, seed_ops, graph, incl_ops):
        self.graph = graph
        self.seed_ops = seed_ops

        self.incl_ops = incl_ops
        self.topo_sort = {}

    def build_dependency_dict(self):
        open_set = Queue.Queue()
        closed_set = set()

        dep_dict = {}
        for op in self.seed_ops:
            open_set.put(op)

        # traversal in the fw phase
        while not open_set.empty():
            src_op = open_set.get()

            # do action for src_op
            dep_ops = set(src_op.control_inputs)
            for t in src_op.inputs:
                dep_ops |= set(util.get_generating_ops(t))
            dep_ops &= self.incl_ops
            dep_dict[src_op] = dep_ops

            next_ops = set()
            for t in src_op.outputs:
                next_ops |= set(util.get_consuming_ops(t))
            next_ops &= self.incl_ops
            for op in next_ops:
                if op in closed_set:
                    continue
                if op not in open_set.queue:
                    open_set.put(op)

            closed_set.add(src_op)

        return dep_dict

    def build(self):
        topo_sort = list(tps(self.build_dependency_dict()))
        for i in range(0, len(topo_sort)):
            self.topo_sort[i] = topo_sort[i]

    def get_order(self, op):
        result = -1
        for order, dep_ops in self.topo_sort.items():
            if op in dep_ops:
                result = order
                break
        return result

    def get_ops(self, order):
        return self.topo_sort[order]

    @property
    def size(self):
        return len(self.topo_sort)
