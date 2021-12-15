from typing import List, Optional

import torch.nn as nn

from ...network import Activation, heatmap2points


class FaceAlignmentTransformer(nn.Module):
    """Face alignment transformer.

    Args:
        image (torch.Tensor): Float32 tensor with shape [b, 3, h, w], normalized to [0, 1].

    Returns:
        landmark (torch.Tensor): Float32 tensor with shape [b, npoints, 2], coordinates normalized to [0, 1].
        aux_outputs:
            heatmap (torch.Tensor): Float32 tensor with shape [b, npoints, S, S]
    """

    def __init__(self, backbone: nn.Module, heatmap_head: nn.Module,
                 heatmap_act: Optional[str] = 'relu'):
        super().__init__()
        self.backbone = backbone
        self.heatmap_head = heatmap_head
        self.heatmap_act = Activation(heatmap_act)
        self.cuda().float()

    def forward(self, image):
        features, _ = self.backbone(image)
        heatmap = self.heatmap_head(features)  # b x npoints x s x s
        heatmap_acted = self.heatmap_act(heatmap)
        landmark = heatmap2points(heatmap_acted)  # b x npoints x 2
        return landmark, {'heatmap': heatmap, 'heatmap_acted': heatmap_acted}
