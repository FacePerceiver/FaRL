# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn


class MMSEG_UPerHead(nn.Module):
    """Wraps the UPerHead from mmseg for segmentation.
    """

    def __init__(self, num_classes: int,
                 in_channels: list = [384, 384, 384, 384], channels: int = 512):
        super().__init__()

        from mmseg.models.decode_heads import UPerHead
        self.head = UPerHead(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=channels,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

    def forward(self, inputs):
        return self.head(inputs)
