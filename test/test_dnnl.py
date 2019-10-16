from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import sys
import unittest

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import torch.jit
import torch.backends.dnnl
from torch.utils import dnnl as dnnl_utils
from common_utils import TestCase, run_tests, TemporaryFileName

from torch.autograd.gradcheck import gradgradcheck, gradcheck


ITERATIONS = 1

# Comment the line below to find out the CI machines having DNNL build disabled
@unittest.skipIf(not torch._C.has_dnnl, "DNNL build is disabled")
class TestDnnl(TestCase):
    def test_conversion(self):
        for _ in range(ITERATIONS):
            for cpu_tensor in [torch.randn((1, 2, 3, 4),
                                        dtype=torch.float, device=torch.device('cpu')),
                            torch.randn((1, 2, 3, 4, 5),
                                        dtype=torch.float, device=torch.device('cpu'))[:, :, :, :, 1]]:
                cpu_tensor.requires_grad_()
                dnnl_tensor = cpu_tensor.to_dnnl()
                cpu_tensor_1 = dnnl_tensor.to_dense()
                self.assertEqual(cpu_tensor, cpu_tensor_1)
                self.assertEqual(dnnl_tensor.dtype, torch.float)
                self.assertEqual(dnnl_tensor.device, torch.device('cpu'))
                self.assertEqual(dnnl_tensor.size(), torch.Size([1, 2, 3, 4]))
                self.assertEqual(dnnl_tensor.numel(), cpu_tensor.numel())
                self.assertEqual(dnnl_tensor.element_size(), cpu_tensor.element_size())
                self.assertRaisesRegex(RuntimeError,
                                    "Cannot access data pointer of Tensor that doesn't have storage",
                                    lambda: dnnl_tensor.data_ptr() != 0)

    # def test_unsupported(self):
    #     # unsupported types and unsupported types with gpu
    #     for dtype in [torch.double, torch.half, torch.uint8, torch.int8,
    #                   torch.short, torch.int, torch.long]:
    #         with self.assertRaises(RuntimeError) as context:
    #             torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cpu')).to_dnnl()
    #         if torch.cuda.is_available():
    #             with self.assertRaises(RuntimeError) as context:
    #                 torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cuda')).to_dnnl()
    #     # supported type with gpu
    #     if torch.cuda.is_available():
    #         with self.assertRaises(RuntimeError) as context:
    #             torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cuda')).to_dnnl()
    #     # some factory functions
    #     for creator in [torch.ones, torch.randn, torch.rand]:
    #         with self.assertRaises(RuntimeError) as context:
    #             creator(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'), layout=torch._dnnl)

    # def test_autograd_to_dnnl(self):
    #     # DNNL only supports float32
    #     root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

    #     def func(root):
    #         return root.to_dnnl().to_dense()

    #     # because DNNL only supports float32, we need to lessen the precision.
    #     # these numbers are just empirical results that seem to work.
    #     self.assertWarnsRegex(lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2),
    #                           'double precision floating point')
    #     self.assertWarnsRegex(lambda: gradgradcheck(func, [root], atol=4e-2, rtol=1e-2),
    #                           'double precision floating point')

    # def test_autograd_from_dnnl(self):
    #     # DNNL only supports float32
    #     root = torch.randn(4, 5, dtype=torch.float32).to_dnnl().requires_grad_()

    #     def func(root):
    #         return root.to_dense()

    #     # because DNNL only supports float32, we need to lessen the precision.
    #     # these numbers are just empirical results that seem to work.
    #     self.assertWarnsRegex(lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2),
    #                           'double precision floating point')

    # def test_detach(self):
    #     root = torch.randn(4, 5, dtype=torch.float32).to_dnnl().requires_grad_()

    #     detach = root.detach()
    #     self.assertEqual((4, 5), detach.size())
    #     self.assertFalse(detach.requires_grad)
    #     self.assertTrue(root.requires_grad)

    #     detach_ = root.detach_()
    #     self.assertEqual((4, 5), detach_.size())
    #     self.assertFalse(detach_.requires_grad)
    #     self.assertFalse(root.requires_grad)

    def test_repr(self):
        self.assertTrue("layout=torch._dnnl" in str(torch.randn((1, 2, 3, 4),
                                                                  dtype=torch.float, device=torch.device('cpu')).to_dnnl()))

    def test_conv2d(self):
        for _ in range(ITERATIONS):
            for groups in [1, 4]:
                N = torch.randint(3, 10, (1,)).item()
                C = torch.randint(1, 32, (1,)).item() * groups
                M = torch.randint(1, 32, (1,)).item() * groups
                x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100
                for bias in [True, False]:
                    conv2d = torch.nn.Conv2d(in_channels=C,
                                            out_channels=M,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=bias,
                                            groups=groups).float()
                    # print('X: ', x)
                    # print('W: ', conv2d.weight)
                    dnnl_conv2d = dnnl_utils.to_dnnl(copy.deepcopy(conv2d))
                    with torch.backends.dnnl.flags(enabled=False):
                        y = conv2d(x)
                    yy = dnnl_conv2d(x.to_dnnl()).to_dense()
                    # print(y)
                    # print(yy)
                    self.assertEqual(
                        y,
                        yy, prec=1e-3)

                # self._test_serialization(dnnl_conv2d, (x.to_dnnl(),))
                # self._test_tracing(dnnl_conv2d, (x.to_dnnl(),))

    def test_relu(self):
        for _ in range(ITERATIONS):
            w = torch.randint(5, 64, (1,)).item()
            h = torch.randint(5, 64, (1,)).item()
            x = torch.randn((h, w), dtype=torch.float32) * 10
            
            y = torch.relu(x)
            yy = torch.relu(x.to_dnnl()).to_dense()
            # print(y)
            # print(yy)
            self.assertEqual(y, yy)

    def test_relu_(self):
        for _ in range(ITERATIONS):
            w = torch.randint(5, 64, (1,)).item()
            h = torch.randint(5, 64, (1,)).item()
            x = torch.randn((h, w), dtype=torch.float32) * 10
            x1 = torch.randn((h, w), dtype=torch.float32) * 10
            x2 = x1.clone().to_dnnl()
            self.assertEqual(torch.relu_(x1), torch.relu_(x2).to_dense())

    def test_max_pool2d(self):
        for _ in range(ITERATIONS):
            N = torch.randint(3, 32, (1,)).item()
            C = torch.randint(3, 32, (1,)).item()
            for stride in [1, 2, 3]:
                for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
                    x = torch.randn(N, C, H, W, dtype=torch.float32) * 10

                    for ceil_mode in [False, True]:
                        max_pool2d = torch.nn.MaxPool2d(
                            kernel_size=3 if not ceil_mode else 7,
                            stride=stride,
                            padding=1,
                            ceil_mode=ceil_mode)
                        # print('########### thnn ############')
                        y = max_pool2d(x)
                        # print('########### dnnl ############')
                        yy = max_pool2d(x.to_dnnl()).to_dense()
                        self.assertEqual(
                            y,
                            yy)

    def test_avg_pool2d(self):
        for _ in range(ITERATIONS):
            N = torch.randint(3, 32, (1,)).item()
            C = torch.randint(3, 32, (1,)).item()
            x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10

            for count_include_pad in [True, False]:
                avg_pool2d = torch.nn.AvgPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=count_include_pad)
                # print('########### thnn ############')
                y = avg_pool2d(x)
                # print('########### dnnl ############')
                yy = avg_pool2d(x.to_dnnl()).to_dense()
                self.assertEqual(
                    y,
                    yy)

    def test_adaptive_avg_pool2d(self):
        for _ in range(ITERATIONS):
            N = torch.randint(3, 32, (1,)).item()
            C = torch.randint(3, 32, (1,)).item()
            x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

            adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

            self.assertEqual(
                adaptive_avg_pool2d(x),
                adaptive_avg_pool2d(x.to_dnnl()).to_dense())

    def test_batch_norm2d(self):
        for _ in range(ITERATIONS):
            N = torch.randint(3, 100, (1,)).item()
            C = torch.randint(3, 100, (1,)).item()
            x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10

            # TODO: support training
            for train in [False]:
                bn = torch.nn.BatchNorm2d(C).float().train(train)
                dnnl_bn = dnnl_utils.to_dnnl(copy.deepcopy(bn))
                y = bn(x)
                yy = dnnl_bn(x.to_dnnl()).to_dense()

                # print(x)
                # print(bn.weight)
                # print(bn.bias)
                # print(bn.running_var)
                # print(bn.running_mean)
                # print(y)
                # print(yy)
                self.assertEqual(
                    y,
                    yy)

                # self._test_serialization(dnnl_bn, (x.to_dnnl(),))
                # self._test_tracing(dnnl_bn, (x.to_dnnl(),))

    def test_add(self):
        for _ in range(ITERATIONS):
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(3, 100, (1,)).item()
            alpha = torch.randn(1, dtype=torch.float32).item()

            x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
            y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
            mx = x.to_dnnl()
            my = y.to_dnnl()

            # add
            # print('#########')
            self.assertEqual(
                x + y,
                (mx + my).to_dense(), 1e-4)
            # print('#########')
            self.assertEqual(
                torch.add(x, y, alpha=alpha),
                torch.add(mx, my, alpha=alpha).to_dense(), 1e-4)

            # add_
            # print('#########')
            x += y
            mx += my
            self.assertEqual(x, mx.to_dense(), 1e-4)

            # add_out
            # print('#########')
            out = x.clone()
            dnnl_out = out.to_dnnl()
            torch.add(x, y, alpha=alpha, out=out)
            torch.add(mx, my, alpha=alpha, out=dnnl_out)
            self.assertEqual(out, dnnl_out.to_dense(), 1e-4)

    def test_mul(self):
        for _ in range(ITERATIONS):
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(3, 100, (1,)).item()
            value = torch.randn(1, dtype=torch.float32).item()

            x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
            y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
            mx = x.to_dnnl()
            my = y.to_dnnl()

            # mul
            self.assertEqual(
                x * y,
                (mx * my).to_dense())

            self.assertEqual(
                x * value,
                (mx * value).to_dense())

            self.assertEqual(
                torch.mul(x, y),
                torch.mul(mx, my).to_dense())

            self.assertEqual(
                torch.mul(x, value),
                torch.mul(mx, value).to_dense())

            # mul_
            x *= y
            mx *= my
            self.assertEqual(x, mx.to_dense())

            x *= value
            mx *= value
            self.assertEqual(x, mx.to_dense())

            # mul_out
            out = x.clone()
            dnnl_out = out.to_dnnl()
            torch.mul(x, y, out=out)
            torch.mul(mx, my, out=dnnl_out)
            self.assertEqual(out, dnnl_out.to_dense())

            out = x.clone()
            dnnl_out = out.to_dnnl()
            torch.mul(x, value, out=out)
            torch.mul(mx, value, out=dnnl_out)
            self.assertEqual(out, dnnl_out.to_dense())

    # def test_view(self):
    #     x = torch.randn(3, 4, 5, dtype=torch.float32).to_dnnl()
    #     self.assertRaisesRegex(RuntimeError,
    #                            "Change to use reshape",
    #                            lambda: x.view(x.size(0), -1))
    #     print(x)
    #     print(x.view(x.size(0), -1))

    def test_reshape(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        size = (x.size(0), -1)
        
        res = x.reshape(size)
        mres = x.to_dnnl().reshape(size).to_dense()
        print(res)
        print(mres)
        self.assertEqual(
            res,
            mres,
        )
        # # test whether share same memory for plain format tensor
        # y = x.to_dnnl()
        # z = y.reshape(size).add_(y.reshape(size))
        # self.assertEqual(
        #     y.reshape(size).to_dense(),
        #     z.to_dense(),
        # )

    def test_clone(self):
        for _ in range(ITERATIONS):
            x = torch.randn(16, 32, 48, 64, dtype=torch.float32) * 10
            self.assertEqual(
                x.clone(),
                x.to_dnnl().clone().to_dense(),
            )
            # test whether share same memory
            y = x.to_dnnl()
            z = y.clone().add_(y)
            self.assertNotEqual(
                y.to_dense(),
                z.to_dense(),
            )

    # def test_transpose(self):
    #     x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
    #     for dim1 in range(x.ndim):
    #         for dim2 in range(x.ndim):
    #             self.assertEqual(
    #                 x.transpose(dim1, dim2),
    #                 x.to_dnnl().transpose(dim1, dim2).to_dense(),
    #             )

    def test_linear(self):
        for _ in range(ITERATIONS):
            in_features = torch.randint(3, 10, (1,)).item()
            out_features = torch.randint(3, 100, (1,)).item()
            x = torch.randn(3, in_features, dtype=torch.float32) * 10

            for bias in [True, False]:
                linear = torch.nn.Linear(in_features, out_features, bias=bias).float()
                dnnl_linear = dnnl_utils.to_dnnl(copy.deepcopy(linear))
                self.assertEqual(
                    linear(x),
                    dnnl_linear(x.to_dnnl()).to_dense())

                # self._test_serialization(dnnl_linear, (x.to_dnnl(),))
                # self._test_tracing(dnnl_linear, (x.to_dnnl(),))

    def test_softmax(self):
        for _ in range(ITERATIONS):
            x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
            for dim in range(x.ndim):
                softmax = torch.nn.Softmax(dim=dim)
                self.assertEqual(
                    softmax(x),
                    softmax(x.to_dnnl()).to_dense())

    def test_sigmoid(self):
        for _ in range(ITERATIONS):
            x = torch.randn(4, 5, dtype=torch.float32) * 10
            dnnl_x = x.to_dnnl()
            self.assertEqual(
                torch.sigmoid(x),
                torch.sigmoid(dnnl_x).to_dense(),
            )
            # inplace
            torch.sigmoid_(x)
            torch.sigmoid_(dnnl_x)
            self.assertEqual(x, dnnl_x.to_dense())

    def _test_serialization(self, module, inputs):
        with TemporaryFileName() as fname:
            torch.jit.save(module, fname)
            loaded = torch.jit.load(fname)
            self.assertEqual(
                module(*inputs).to_dense(),
                loaded(*inputs).to_dense())

    def _test_tracing(self, module, inputs):
        traced = torch.jit.trace(module, inputs, check_trace=False)
        self.assertEqual(
            module(*inputs).to_dense(),
            traced(*inputs).to_dense())

    def test_set_data_tensorimpl_type(self):
        # Dense tensor has impl of type `TensorImpl`, while DNNL tensor has impl
        # of type `OpaqueTensorImpl<IDeepTensorWrapperPtr>`.
        x = torch.randn((1, 2), dtype=torch.float, device=torch.device('cpu'))
        x_dnnl = x.to_dnnl()
        with self.assertRaisesRegex(RuntimeError, 'incompatible tensor type'):
            x.data = x_dnnl

    def test_empty(self):
        x1 = torch.empty(4, 5, 2, 3, dtype=torch.float32)
        x2 = torch.empty(4, 5, 2, 3, dtype=torch.float32, layout=torch._dnnl)
        self.assertEqual(x1.size(), x2.to_dense().size())
        self.assertEqual(x1.dtype, x2.to_dense().dtype)

    def test_zero_(self):
        x1 = torch.randn(4, 5, dtype=torch.float32) * 10
        x2 = x1.clone().to_dnnl()
        # print(x1)
        # print(x2)
        self.assertEqual(
            x1.zero_(),
            x2.zero_().to_dense(),
        )

    def test_is_dnnl(self):
        x = torch.randn(1, dtype=torch.float32)
        self.assertFalse(x.is_dnnl)
        self.assertTrue(x.to_dnnl().is_dnnl)

    # def test_is_dnnl_jit(self):
    #     class EnsureDnnl(torch.jit.ScriptModule):
    #         @torch.jit.script_method
    #         def forward(self, x):
    #             if not x.is_dnnl:
    #                 x = x.to_dnnl()
    #             return x

    #     m = EnsureDnnl()
    #     x = torch.randn(1, dtype=torch.float32)
    #     self.assertTrue(m(x).is_dnnl)
    #     self.assertTrue(m(x.to_dnnl()).is_dnnl)

    def _test_imagenet_model(self, model): 
        print('################ Imperative ####################')
        model = model.train(False).float()
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        with torch.backends.dnnl.flags(enabled=False):
            y = model(x)
        print('################ DNNL ####################')
        dnnl_model = dnnl_utils.to_dnnl(copy.deepcopy(model))
        # print('### ITERATION BEGIN ###')
        # sys.stdout.flush()
        # yy = dnnl_model(x.to_dnnl()).to_dense()
        # print('### ITERATION BEGIN ###')
        # sys.stdout.flush()
        # for _ in range(200):
        yy = dnnl_model(x.to_dnnl()).to_dense()

        # # print(y)
        # # print(yy)

        with torch.no_grad():
            self.assertEqual(
                y,
                yy,
            )

    @skipIfNoTorchVision
    def test_resnet18(self):
        model = torchvision.models.resnet.resnet18(pretrained=False)
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    def test_resnext50_32x4d(self):
        model = torchvision.models.resnet.resnext50_32x4d(pretrained=False)
        self._test_imagenet_model(model)


if __name__ == '__main__':
    run_tests()
