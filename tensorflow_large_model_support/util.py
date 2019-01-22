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
"""Utility functions for editing graph
"""
import time
from functools import wraps


def fanins(op):
    """Return all incomming operations.

    Args:
      op: a `tf.Operation`.

    Return:
      A set of `tf.Operation`.
    """
    return {t.op for t in op.inputs}


def fanouts(op):
    """Return all outgoing operations.

    Args:
      op: a `tf.Operation`.

    Return:
      A set of `tf.Operation`.
    """
    ops = set()
    for t in op.outputs:
        for op in t.consumers():
            ops |= {op}
    return ops


def add_control_inputs(op, cops):
    """Add control dependencies from `cops` to `op`.

    Args:
      op: a tf.Operation to which the control inputs are added.
      cops: an object convertible to a list of `tf.Operation`.
    Raises:
      TypeError: if op is not a tf.Operation
      ValueError: if any cop in cops is already a control input of op.
    """
    flag = True
    try:
        _ = iter(cops)
    except Exception:  # pylint: disable=broad-except
        flag = False
    if not flag:
        op._add_control_input(cops)
    else:
        for cop in cops:
            if cop in op.control_inputs:
                raise ValueError("{}".format(cop.name) + " is already " +
                                 "a control_input of {}".format(op.name))
        op._add_control_inputs(cops)  # pylint: disable=protected-access


def is_cpu_op(op):
    return "CPU" in op.device.upper()


def build_control_outputs(graph):
    """Build a dictionary of (op, control_outputs).
    """
    ops = graph.get_operations()
    control_outputs = {}
    for op in ops:
        for cin in op.control_inputs:
            if cin in control_outputs:
                control_outputs[cin] |= {op}
            else:
                control_outputs[cin] = {op}
    return control_outputs


def reroute_input(ts0, ts1, op1):
    """Replace the input `ts1` of operation `op1` by tensor `ts0`.
    """
    for i, t in enumerate(op1.inputs):
        if t is ts1:
            op1._update_input(i, ts0)  # pylint: disable=protected-access


def get_tensor_size(ts, bs=None):
    """Return the size of tensor in bytes.
    The first unknown dimension of a tensor will be filled by `bs`.
    The other unknown dimenstions of a tensor will be filled by a default
    value.
    """
    d, s = 1, 1  # `d` default value for i-th unknown dimension (i != 0)
    ndims = ts.shape.ndims
    if ndims is None or ndims <= 1:
        return d
    for i in range(0, ndims):
        v = ts.shape[i].value
        if v is None:
            if i == 0:
                v = bs if bs is not None else d
            else:
                v = d
        s *= v
    return s*(ts.dtype.size)


# decorators
def measure_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        rt = f(*args, **kwargs)
        args[0]._log_info(
            "took: {} ms".format((time.time()-start_time)*1000), 0)
        return rt
    return wrapper
