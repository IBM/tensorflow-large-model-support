import queue as Queue
from toposort import toposort as tps

import tensorflow.contrib.graph_editor as ge
from tensorflow.contrib.graph_editor import util

class TOPOS(object):
    def __init__(self, seed_ops, graph, grad_ops):
        self.graph = graph
        self.seed_ops = seed_ops
        self.grad_ops = grad_ops

        self.topo_sort = {}
        self.bw_starting_order_ = -1

    def build(self):
        topo_sort = list(tps(self.build_dependency_dict()))
        for i in range(0, len(topo_sort)):
            self.topo_sort[i] = topo_sort[i]
    
        # if a bw op has the same order with a fw op,
        # then remove the bw op
        self.clean_bw_ops()

        for i in range(0, len(self.topo_sort)):
            ops = self.topo_sort[i]
            if (ops & self.grad_ops):
                self.bw_starting_order_ = i
                break

        # if there are non-bw ops in the bw phase,
        # then remove them, and do reordering
        self.clean_update_ops()
        self.reindex()

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

    def clean_bw_ops(self):
        '''There are some bw ops that 
             - have no incoming bw ops except its fw op, or
             - have no outgoing ops.
        Execution order of these ops may depend on Tensorflow runtime.
        '''

        for i in range(0, len(self.topo_sort)):
            dep_ops = self.topo_sort[i]
            fw_dep_ops = dep_ops - self.grad_ops
            bw_dep_ops = dep_ops & self.grad_ops
            if fw_dep_ops:
                self.topo_sort[i] = fw_dep_ops
            else:
                self.topo_sort[i] = dep_ops

    def clean_update_ops(self):
        '''Remove ops that are in the update phase
        '''
        for i in range(0, len(self.topo_sort)):
            ops = self.topo_sort[i]
            # remove ops that are not bw or fw op
            # e.g ops in the update phase
            ops = {op for op in ops
                   if (set(ge.get_forward_walk_ops(op))
                       & self.grad_ops)}
            self.topo_sort[i] = ops

    def reindex(self):
        ''' Remove orders with empty set and reindex
        '''
        topo_sort = {}
        index = 0
        for i in range(0, len(self.topo_sort)):
            ops = self.topo_sort[i]
            if ops:
                topo_sort[index] = ops
                index += 1
        self.topo_sort = topo_sort

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

    @property
    def bw_starting_order(self):
        return self.bw_starting_order_
