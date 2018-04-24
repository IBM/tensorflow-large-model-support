from tensorflow.python.framework import ops
from tensorflow.python.training import session_run_hook
from lms.lms import LMS


class LMSHook(session_run_hook.SessionRunHook):
    ''' This hook is to modify the input graph for Large Model Support
    by adding swap operations
    '''
    def __init__(self, optimizer_scopes=set(),
                 starting_scope=None,
                 excl_scopes=set(),
                 incl_scopes=set(),
                 excl_types=set(),
                 incl_types=set(),
                 lb=1, ub=10000,
                 n_tensors=-1,
                 fuse_swapins=False,
                 ctrld_strategy="chain_rule",
                 swap_branches=False,
                 branch_threshold=0,
                 debug=False,
                 debug_level=1,
                 cpu_device="/cpu:0"):
        self.lms_obj = LMS(optimizer_scopes=optimizer_scopes,
                           starting_scope=starting_scope,
                           excl_scopes=excl_scopes,
                           incl_scopes=incl_scopes,
                           excl_types=excl_types,
                           incl_types=incl_types,
                           lb=lb, ub=ub,
                           n_tensors=n_tensors,
                           fuse_swapins=fuse_swapins,
                           ctrld_strategy=ctrld_strategy,
                           swap_branches=swap_branches,
                           branch_threshold=branch_threshold,
                           debug=debug,
                           debug_level=debug_level,
                           cpu_device=cpu_device)

    def begin(self):
        self.lms_obj.run(ops.get_default_graph())
