import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def is_available():
    r"""Returns whether PyTorch is built with DNNL support."""
    return torch._C.has_dnnl

def set_flags(_enabled):
    orig_flags = (torch._C._get_dnnl_enabled(),)
    torch._C._set_dnnl_enabled(_enabled)
    return orig_flags

@contextmanager
def flags(enabled=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])

class DnnlModule(PropModule):
    def __init__(self, m, name):
        super(DnnlModule, self).__init__(m, name)

    enabled = ContextProp(torch._C._get_dnnl_enabled, torch._C._set_dnnl_enabled)

# Cool stuff from torch/backends/cudnn/__init__.py and
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = DnnlModule(sys.modules[__name__], __name__)