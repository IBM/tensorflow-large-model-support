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

"""Utility functions for editing graph
"""
import time
from functools import wraps

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.core.framework import attr_value_pb2


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


def is_gpu_op(op, gpu_device):
    if gpu_device is None:
        if is_cpu_op(op):
            return False
        else:
            return True
    else:
        return op.device.upper() == gpu_device.upper()


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
    if ndims is None:
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
            "{}() took: {} ms".format(
                f.__name__,
                (time.time()-start_time)*1000), 0)
        return rt
    return wrapper


def get_var_or_handle(gvars):
    """If a variable is a resource variable then return its handle.
    Otherwise, return itself.
    """
    rvars = set()
    for v in gvars:
        if hasattr(v, "handle") and hasattr(v.handle, "op") and isinstance(
                v.handle.op, tf.Operation):
            rvars.add(v.handle.op)  # ResourceVariable
        elif v._variable is not None:
            rvars.add(v._variable.op)  #RefVariable
        else:
            rvars.add(v)  # Variable, VariableV2
    return rvars


def get_op_size(op, batch_size):
    op_size = 0
    for ts in op.inputs:
        op_size += get_tensor_size(ts, batch_size)
    for ts in op.outputs:
        op_size += get_tensor_size(ts, batch_size)
    return op_size


def protect_op_from_optimizers(op):
    op._set_attr('_grappler_do_not_remove',
                 attr_value_pb2.AttrValue(b=True))
