# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F


class FaceParsingTransformer(nn.Module):
    """Face parsing transformer.

    Args:
        image (torch.Tensor): Float32 tensor with shape [b, 3, h, w], normalized to [0, 1].

    Returns:
        logits (torch.Tensor): Float32 tensor with shape [b, nclasses, out_size[0], out_size[1]]
        aux_outputs (dict): Empty.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, out_size: Tuple[int, int]):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.out_size = out_size
        self.cuda().float()

    def forward(self, image):
        features, _ = self.backbone(image)
        logits = self.head(features)
        return F.interpolate(logits, size=self.out_size, mode='bilinear', align_corners=False), dict()
