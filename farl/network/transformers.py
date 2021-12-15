# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


from blueprint.ml.util import deal_with_remote_file
from blueprint import Context

from . import farl


def _make_fpns(vision_patch_size: int, output_channels: int):
    if vision_patch_size in {16, 14}:
        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels,
                               kernel_size=2, stride=2),
            nn.SyncBatchNorm(output_channels),
            nn.GELU(),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2))

        fpn2 = nn.ConvTranspose2d(
            output_channels, output_channels, kernel_size=2, stride=2)
        fpn3 = nn.Identity()
        fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.ModuleList([fpn1, fpn2, fpn3, fpn4])
    elif vision_patch_size == 8:
        fpn1 = nn.Sequential(nn.ConvTranspose2d(
            output_channels, output_channels, kernel_size=2, stride=2))
        fpn2 = nn.Identity()
        fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)
        fpn4 = nn.MaxPool2d(kernel_size=4, stride=4)
        return nn.ModuleList([fpn1, fpn2, fpn3, fpn4])
    else:
        raise NotImplementedError()


def _resize_pe(pe: torch.Tensor, new_size: int, mode: str = 'bicubic', num_tokens: int = 1) -> torch.Tensor:
    """Resize positional embeddings.

    Args: 
        pe (torch.Tensor): A tensor with shape (num_tokens + old_size ** 2, width). pe[0, :] is the CLS token.

    Returns:
        torch.Tensor: A tensor with shape (num_tokens + new_size **2, width).
    """
    l, w = pe.shape
    old_size = int(math.sqrt(l-num_tokens))
    assert old_size ** 2 + num_tokens == l
    return torch.cat([
        pe[:num_tokens, :],
        F.interpolate(pe[num_tokens:, :].reshape(1, old_size, old_size, w).permute(0, 3, 1, 2),
                      (new_size, new_size), mode=mode, align_corners=False).view(w, -1).t()], dim=0)


class FaRLVisualFeatures(nn.Module):
    """Extract features from FaRL visual encoder.

    Args:
        image (torch.Tensor): Float32 tensor with shape [b, 3, h, w], 
            normalized to [0, 1].

    Returns:
        List[torch.Tensor]: A list of features.
    """
    image_mean: torch.Tensor
    image_std: torch.Tensor
    output_channels: int
    num_outputs: int

    def __init__(self, model_type: str,
                 model_path: str, output_indices: Optional[List[int]] = None,
                 forced_input_resolution: Optional[int] = None,
                 apply_fpn: bool = True, _ctx: Optional[Context] = None):
        super().__init__()

        model_path = deal_with_remote_file(
            model_path, _ctx.copy2local, _ctx.blob_root)
        self.visual = farl.load_farl(model_type, model_path)

        vision_patch_size = self.visual.conv1.weight.shape[-1]

        self.input_resolution = self.visual.input_resolution
        if forced_input_resolution is not None and \
                self.input_resolution != forced_input_resolution:
            # resizing the positonal embeddings
            self.visual.positional_embedding = nn.Parameter(
                _resize_pe(self.visual.positional_embedding,
                           forced_input_resolution//vision_patch_size))
            self.input_resolution = forced_input_resolution

        self.output_channels = self.visual.transformer.width

        if output_indices is None:
            output_indices = self.__class__.get_default_output_indices(
                model_type)
        self.output_indices = output_indices
        self.num_outputs = len(output_indices)

        self.register_buffer('image_mean', torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        if apply_fpn:
            self.fpns = _make_fpns(vision_patch_size, self.output_channels)
        else:
            self.fpns = None

    @staticmethod
    def get_output_channel(model_type):
        if model_type == 'base':
            return 768
        if model_type == 'large':
            return 1024
        if model_type == 'huge':
            return 1280

    @staticmethod
    def get_default_output_indices(model_type):
        if model_type == 'base':
            return [3, 5, 7, 11]
        if model_type == 'large':
            return [7, 11, 15, 23]
        if model_type == 'huge':
            return [8, 14, 20, 31]

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        # b x 3 x res x res
        _, _, input_h, input_w = image.shape
        if input_h != self.input_resolution or input_w != self.input_resolution:
            image = F.interpolate(image, self.input_resolution,
                                  mode='bilinear', align_corners=False)
        image = (image - self.image_mean) / self.image_std

        x = image.to(self.visual.conv1.weight.data)

        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        N, _, S, S = x.shape

        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1],
                                   dtype=x.dtype, device=x.device),
                       x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2).contiguous()  # NLD -> LND

        features = []
        cls_tokens = []
        for blk in self.visual.transformer.resblocks:
            x = blk(x)  # [S ** 2 + 1, N, D]
            # if idx in self.output_indices:
            feature = x[1:, :, :].permute(
                1, 2, 0).view(N, -1, S, S).contiguous().float()
            features.append(feature)
            cls_tokens.append(x[0, :, :])

        features = [features[ind] for ind in self.output_indices]
        cls_tokens = [cls_tokens[ind] for ind in self.output_indices]

        if self.fpns is not None:
            for i, fpn in enumerate(self.fpns):
                features[i] = fpn(features[i])

        return features, cls_tokens
