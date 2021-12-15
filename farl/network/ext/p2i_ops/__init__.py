# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Tuple, List, Union

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(os.path.abspath(__file__))
if 'CUDA_HOME' in os.environ:
    os.environ['CUDA_HOME'] = '/usr/local/cuda'

ext = load(
    'ext',
    sources=[
        os.path.join(module_path, 'ext.cpp'),
        os.path.join(module_path, 'p2i_sum.cu'),
        os.path.join(module_path, 'p2i_max.cu'),
    ],
    extra_cuda_cflags=['--expt-extended-lambda',
                       '-O3', '-use_fast_math']
)
assert ext is not None


# p2i


class P2ISumFunction(Function):
    @staticmethod
    def forward(ctx, points, point_features, batch_inds, background,
                kernel_kind, kernel_radius):
        ctx.save_for_backward(points, point_features, batch_inds)
        ctx.kernel_kind = kernel_kind
        ctx.kernel_radius = kernel_radius

        out = ext.p2i_sum_forward_gpu(
            points.contiguous(),
            point_features.contiguous(),
            batch_inds.contiguous(),
            background.contiguous(), kernel_kind, kernel_radius)

        return (out,)

    @staticmethod
    def backward(ctx, out_grad):
        points, point_features, batch_inds = ctx.saved_tensors
        kernel_kind = ctx.kernel_kind
        kernel_radius = ctx.kernel_radius

        points_grad, point_features_grad = ext.p2i_sum_backward_gpu(
            out_grad.contiguous(), points.contiguous(),
            point_features.contiguous(), batch_inds.contiguous(),
            kernel_kind, kernel_radius)

        background_grad = out_grad
        return (points_grad, point_features_grad, None,
                background_grad, None, None)


class P2IMaxFunction(Function):
    @staticmethod
    def forward(ctx, points, point_features, batch_inds, background,
                kernel_kind, kernel_radius):

        out, out_point_ids = ext.p2i_max_forward_gpu(
            points.contiguous(),
            point_features.contiguous(),
            batch_inds.contiguous(),
            background.contiguous(), kernel_kind, kernel_radius)

        ctx.save_for_backward(points, point_features, out_point_ids)
        ctx.kernel_kind = kernel_kind
        ctx.kernel_radius = kernel_radius

        ctx.mark_non_differentiable(out_point_ids)

        return (out, out_point_ids)

    @staticmethod
    def backward(ctx, out_grad, _):
        points, point_features, out_point_ids = ctx.saved_tensors
        kernel_kind = ctx.kernel_kind
        kernel_radius = ctx.kernel_radius

        points_grad, point_features_grad, background_grad = ext.p2i_max_backward_gpu(
            out_grad.contiguous(), out_point_ids, points.contiguous(),
            point_features.contiguous(),
            kernel_kind, kernel_radius)

        return (points_grad, point_features_grad, None,
                background_grad, None, None)


_p2i_kernel_kind_dict = {'cos': 0, 'gaussian_awing': 1}


def p2i(points: torch.Tensor, point_features: torch.Tensor,
        batch_inds: torch.Tensor, background: torch.Tensor,
        kernel_radius: float, kernel_kind_str: str = 'cos', reduce: str = 'sum',
        with_auxilary_output: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Paint point cloud features on to 2D feature maps.

    Args:
        points (torch.Tensor): float, [npoints x (X,Y)]. X, Y are all absolute coordinates.
        point_features (torch.Tensor): float, [npoints x channels]
        batch_inds (torch.Tensor): int32, [npoints]
        background (torch.Tensor): float, [batch x channels x out_h x out_w]
        kernel_radius (float):
        kernel_kind_str (str): {'cos'}
        reduce (str): {'sum', 'max'}

    Returns:
      - torch.Tensor: float, [batch x channels x out_h x out_w]
    """
    kernel_kind = _p2i_kernel_kind_dict[kernel_kind_str]

    assert points.size(0) == point_features.size(0)
    assert batch_inds.size(0) == points.size(0)
    assert background.size(1) == point_features.size(1)

    points = points[:, [1, 0]]

    if reduce == 'sum':
        assert kernel_kind == 0  # other kinds not implemented yet for p2i_sum
        result = P2ISumFunction.apply(points, point_features, batch_inds, background,
                                      kernel_kind, kernel_radius)
    elif reduce == 'max':
        result = P2IMaxFunction.apply(points, point_features, batch_inds, background,
                                      kernel_kind, kernel_radius)
    else:
        raise RuntimeError(f'Invalid reduce value: {reduce}')
    if with_auxilary_output:
        return result
    return result[0]
