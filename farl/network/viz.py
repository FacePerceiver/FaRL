# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

from .geometry import denormalize_points
from .ext.p2i_ops import p2i


def _expand_as_rgbs(x):
    _, c, _, _ = x.shape
    if c == 3:
        return [x]

    if c % 3 > 0:
        x = torch.cat([
            x, x[:, [-1], :, :].expand(
                -1, 3 - c % 3, -1, -1)], dim=1)
    c = x.size(1)
    assert c % 3 == 0
    return list(x.split([3] * (c // 3), dim=1))


def _visualize_flags(flags, size, num_flags):
    batch_size = flags.size(0)
    flags = flags.to(dtype=torch.uint8)
    has_what = [flags & torch.full_like(flags, 1 << i)
                for i in range(num_flags)]
    # batch x 1 x 1 x 4
    vis_im = torch.stack(has_what, dim=1).float().view(
        batch_size, 1, 1, num_flags)
    vis_im = F.interpolate(vis_im.expand(-1, 3, -1, -1),
                           size=size, mode='nearest')
    return vis_im


def visualize_in_row(*data) -> torch.Tensor:
    """Visualize data in one row.

    Args:
        *data (list): A list of (value, modal, [v_min, v_max]) tuples.

        Each tuple defines the following inputs:

            value (torch.Tensor): The data value to visualize.
            modal (str): The modal type string of the data.
                Supported data modal types are:

                * "BHW", "BNHW", "BHWN" for tensors;
                * "flags_{K}" for binary flags, with K being the number of bits;
                * "points" for points, where `value` is a tensor with shape [B, N, 2].

            v_min (float): Optional, to normalize value.
            v_max (float): Optional, to normalize value.

    Returns:
        torch.Tensor: A tensor with shape b x 3 x h x w.
    """
    batch = None
    size = None
    device = None

    row = []
    for v in data:
        assert isinstance(v, (tuple, list))
        if len(v) == 2:
            value, modal = v
            v_min, v_max = 0.0, 1.0
        elif len(v) == 4:
            value, modal, v_min, v_max = v
        else:
            raise RuntimeError(
                'Input either (value, modal) or (value, modal, v_min, v_max)')

        if value is None:
            assert batch is not None
            assert size is not None
            assert device is not None
            value = torch.rand(batch, 1, size[0], size[1], device=device)
            modal = 'BNHW'
            v_min, v_max = 0.0, 1.0

        if modal == 'BHW':
            assert isinstance(value, torch.Tensor)
            value = value.detach().float()

            batch = value.size(0)
            size = value.shape[1:]
            device = value.device

            value = (value - v_min) / (v_max - v_min)
            row.append(value.unsqueeze(
                1).expand(-1, 3, -1, -1))

        elif modal == 'BNHW':
            assert isinstance(value, torch.Tensor)
            value = value.detach().float()

            batch = value.size(0)
            size = value.shape[2:]
            device = value.device

            value = (value - v_min) / (v_max - v_min)
            row += _expand_as_rgbs(value)

        elif modal == 'BHWN':
            assert isinstance(value, torch.Tensor)
            value = value.detach().float().permute(0, 3, 1, 2)

            batch = value.size(0)
            size = value.shape[2:]
            device = value.device

            value = (value - v_min) / (v_max - v_min)
            row += _expand_as_rgbs(value)

        elif modal.startswith('flags_'):
            assert isinstance(value, torch.Tensor)
            value = value.detach().float()

            batch = value.size(0)
            device = value.device

            num_flags = int(modal.split('_')[1])
            assert size is not None
            row.append(_visualize_flags(value, size, num_flags))

        elif modal == 'points':
            points, background = value

            if background is None:
                background = torch.rand(
                    batch, 1, size[0], size[1], device=device)
            else:
                assert isinstance(background, torch.Tensor)
                background = background.detach().float()
                background = (background - v_min) / (v_max - v_min)

            if points is None:
                canvas = background
            else:
                assert isinstance(points, torch.Tensor)
                points = points.detach().float()
                points = denormalize_points(
                    points, background.size(2), background.size(3))

                npoints = points.size(1)
                batch = background.size(0)
                assert points.size(0) == batch
                channels = background.size(1)

                points = points.reshape(npoints * batch, 2)

                point_colors = torch.ones(
                    npoints * batch, channels, dtype=background.dtype, device=background.device)
                batch_inds = torch.arange(batch).unsqueeze(1).expand(-1, npoints).reshape(
                    npoints * batch).to(dtype=torch.int32, device=background.device)
                canvas = p2i(points, point_colors, batch_inds, background, 5)

            row.append(canvas)

    return torch.cat(row, dim=-1)
