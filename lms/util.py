"""Utility functions for editing graph
"""


def fanins(op):
    """Return all incomming operations.

    Args:
      op: a `tf.Operation`.

    Return:
      A list of `tf.Operation`.
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
