# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Mapping, Any, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from blueprint.ml import Task, ForwardFlags

from ...network.viz import visualize_in_row


class FaceParsing(Task):
    """Standard segmentation with crossentropy loss supervision.

    Requires the corresponding network inputs an image and outputs the segmentation logits.

    Returns:
        pred_logit: [b, h, w, num_classes]
    """

    def __init__(self, network_name: str = 'main',
                 network_name_eval: Optional[str] = None,
                 image_tag: str = 'image', label_tag: str = 'label',
                 pred_logit_tag: str = 'pred_logit') -> None:
        super().__init__()
        self.network_name = network_name
        self.network_name_eval = network_name_eval
        if self.network_name_eval is None:
            self.network_name_eval = self.network_name
        self.image_tag = image_tag
        self.label_tag = label_tag
        self.pred_logit_tag = pred_logit_tag

    def setup_networks(self, networks: Mapping[str, nn.Module]):
        self.segmentation_net = networks[self.network_name]
        self.segmentation_net_eval = networks[self.network_name_eval]

    def forward(self, data: Mapping[str, torch.Tensor], flags: ForwardFlags
                ) -> Tuple[
                    Optional[torch.Tensor],
                    Mapping[str, torch.Tensor],
                    Mapping[str, torch.Tensor],
                    Mapping[str, torch.Tensor]]:

        # b x c x h x w
        if self.training:
            pred_logit, aux_outputs = self.segmentation_net(
                data[self.image_tag].cuda().permute(0, 3, 1, 2).contiguous())
        else:
            pred_logit, aux_outputs = self.segmentation_net_eval(
                data[self.image_tag].cuda().permute(0, 3, 1, 2).contiguous())

        if flags.with_losses:
            gt_label = data[self.label_tag].to(
                device=pred_logit.device, dtype=torch.int64)  # batch, h, w

            batch, channels, h, w = pred_logit.shape

            assert gt_label.shape == (batch, h, w)

            pred_logit_vec = pred_logit.permute(
                0, 2, 3, 1).reshape(-1, channels)  # (batchxhxw), channels

            gt_label_vec = gt_label.view(-1)  # (batchxhxw)

            ce_loss_vec = F.cross_entropy(
                pred_logit_vec, target=gt_label_vec, reduction='none')  # (batchxhxw)

            ce_loss = ce_loss_vec.view(batch, h, w).mean([1, 2])

            loss, losses = ce_loss.mean(), {'ce_loss': ce_loss}
        else:
            loss, losses = None, dict()

        if flags.with_outputs:
            outputs = {**aux_outputs,
                       self.pred_logit_tag: pred_logit.permute(0, 2, 3, 1)}
        else:
            outputs = dict()

        if flags.with_images:
            images = {self.pred_logit_tag: visualize_in_row(
                (pred_logit.softmax(dim=1), 'BNHW', 0.0, 1.0))}
        else:
            images = dict()

        return loss, losses, outputs, images
