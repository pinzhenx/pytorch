from __future__ import absolute_import, division, print_function, unicode_literals

import torch


class DnnlLinear(torch.jit.ScriptModule):
    def __init__(self, dense_module):
        super(DnnlLinear, self).__init__()
        self.register_buffer('weight', dense_module.weight.to_dnnl())
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_dnnl())
        else:
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_dnnl())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense())

    @torch.jit.script_method
    def __setstate__(self, state):
        # type: (Tuple[Tensor, Tensor]) -> None
        self.weight = state[0].to_dnnl()
        self.bias = state[1].to_dnnl()

    @torch.jit.script_method
    def forward(self, x):
        x_dnnl = x if x.is_dnnl else x.to_dnnl()
        y_dnnl = torch._C._nn.dnnl_linear(x_dnnl, self.weight, self.bias)
        y = y_dnnl if x.is_dnnl else y_dnnl.to_dense()
        return y


class DnnlConv2d(torch.jit.ScriptModule):
    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module):
        super(DnnlConv2d, self).__init__()

        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups

        self.register_buffer('weight', dense_module.weight.to_dnnl())
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_dnnl())
        else:
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_dnnl())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense())

    @torch.jit.script_method
    def __setstate__(self, state):
        # type: (Tuple[Tensor, Tensor]) -> None
        self.weight = torch._C._nn.dnnl_reorder_conv2d_weight(
            state[0].to_dnnl(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_dnnl()

    @torch.jit.script_method
    def forward(self, x):
        return torch.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups)


class DnnlBatchNorm2d(torch.jit.ScriptModule):
    __constants__ = ['exponential_average_factor', 'eps']

    def __init__(self, dense_module):
        super(DnnlBatchNorm2d, self).__init__()

        assert(not dense_module.training)
        assert(dense_module.track_running_stats)
        assert(dense_module.affine)

        if dense_module.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = dense_module.momentum
        self.eps = dense_module.eps

        self.register_buffer('weight', dense_module.weight.to_dnnl())
        self.register_buffer('bias', dense_module.bias.to_dnnl())
        self.register_buffer('running_mean', dense_module.running_mean.to_dnnl())
        self.register_buffer('running_var', dense_module.running_var.to_dnnl())

    @torch.jit.script_method
    def __getstate__(self):
        weight = self.weight.to_dense()
        bias = self.bias.to_dense()
        running_mean = self.running_mean.to_dense()
        running_var = self.running_var.to_dense()
        return (weight, bias, running_mean, running_var)

    @torch.jit.script_method
    def __setstate__(self, state):
        # type: (Tuple[Tensor, Tensor, Tensor, Tensor]) -> None
        self.weight = state[0].to_dnnl()
        self.bias = state[1].to_dnnl()
        self.running_mean = state[2].to_dnnl()
        self.running_var = state[3].to_dnnl()

    @torch.jit.script_method
    def forward(self, x):
        return torch.batch_norm(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            False,  # training
            self.exponential_average_factor,
            self.eps,
            False,  # cuda_enabled
        )


def to_dnnl(module):
    def m_fn(m):
        if isinstance(m, torch.nn.Linear):
            return DnnlLinear(m)
        elif isinstance(m, torch.nn.Conv2d):
            return DnnlConv2d(m)
        elif isinstance(m, torch.nn.BatchNorm2d):
            return DnnlBatchNorm2d(m)
        else:
            return m

    def m_fn_rec(m):
        new_m = m_fn(m)
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m))
        return new_m

    return m_fn_rec(module)