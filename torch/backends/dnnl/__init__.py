import torch


def is_available():
    r"""Returns whether PyTorch is built with DNNL support."""
    return torch._C.has_dnnl
