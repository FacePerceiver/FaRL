# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Mapping, Any, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from blueprint.ml import Task, ForwardFlags

from ...network import normalize_points, denormalize_points, points2heatmap
from ...network.viz import visualize_in_row


class FaceAlignment(Task):
    """ Face alignment tasks.
    """

    def __init__(self,
                 network_name: str = 'main',
                 network_name_eval: Optional[str] = None,
                 image_tag: str = 'image', landmark_tag: str = 'landmark',
                 pred_landmark_tag: str = 'pred_landmark',
                 loss_weights: Mapping[str, float] = {'coord_l1_loss': 1.0},
                 heatmap_size: int = 128,
                 heatmap_radius: float = 16.0,
                 heatmap_interpolate_mode: str = 'bilinear') -> None:

        super().__init__()
        self.network_name = network_name
        self.network_name_eval = network_name_eval
        if self.network_name_eval is None:
            self.network_name_eval = self.network_name
        self.image_tag = image_tag
        self.landmark_tag = landmark_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.loss_weights = loss_weights
        self.heatmap_size = heatmap_size
        self.heatmap_radius = heatmap_radius
        self.heatmap_interpolate_mode = heatmap_interpolate_mode

    def setup_networks(self, networks: Mapping[str, nn.Module]):
        self.regression_net = networks[self.network_name]
        self.regression_net_eval = networks[self.network_name_eval]

    def forward(self, data: Mapping[str, torch.Tensor], flags: ForwardFlags
                ) -> Tuple[
                    Optional[torch.Tensor],
                    Mapping[str, torch.Tensor],
                    Mapping[str, torch.Tensor],
                    Mapping[str, torch.Tensor]]:
        # b x c x h x w
        image = data[self.image_tag].cuda().permute(0, 3, 1, 2).contiguous()
        _, _, h, w = image.shape

        # b x n x 2
        if self.training:
            net = self.regression_net
        else:
            net = self.regression_net_eval
        pred_landmark, aux_outputs = net(image)

        cache = dict()
        if flags.with_losses:
            landmark = normalize_points(
                data[self.landmark_tag].to(image), h, w)

            # compute all losses
            def _compute_named_loss(name: str) -> torch.Tensor:
                if name == 'coord_l1_loss':
                    return (landmark - pred_landmark).norm(dim=-1).mean([1])

                if name.startswith('heatmap'):
                    if 'pred_heatmap' not in cache:
                        cache['pred_heatmap'] = F.interpolate(
                            aux_outputs['heatmap'], (self.heatmap_size,
                                                     self.heatmap_size),
                            mode=self.heatmap_interpolate_mode, align_corners=False)
                    if 'pred_heatmap_acted' not in cache:
                        cache['pred_heatmap_acted'] = F.interpolate(
                            aux_outputs['heatmap_acted'], (self.heatmap_size,
                                                           self.heatmap_size),
                            mode=self.heatmap_interpolate_mode, align_corners=False)
                    if 'heatmap' not in cache:
                        # render gt heatmap
                        with torch.no_grad():
                            cache['heatmap'] = points2heatmap(
                                landmark, (self.heatmap_size, self.heatmap_size), self.heatmap_radius)

                if name == 'heatmap_l1_loss':
                    return (cache['pred_heatmap_acted'] - cache['heatmap']).abs().mean([1, 2, 3])
                if name == 'heatmap_l2_loss':
                    return (cache['pred_heatmap'] - cache['heatmap']).pow(2).mean([1, 2, 3])
                if name == 'heatmap_ce_loss':
                    bce_loss = F.binary_cross_entropy_with_logits(
                        cache['pred_heatmap'], cache['heatmap'], reduction='none')
                    return bce_loss.mean([1, 2, 3])

                raise RuntimeError(f'Unknown loss name: {name}.')

            losses = {name: _compute_named_loss(
                name) for name, weight in self.loss_weights.items() if weight != 0.0}
            loss = sum([l * self.loss_weights[name]
                        for name, l in losses.items()]).mean()
        else:
            loss, losses = None, dict()

        if flags.with_outputs:
            outputs = {self.pred_landmark_tag: denormalize_points(
                pred_landmark, h, w)}
            if 'heatmap' in cache:
                outputs['heatmap'] = cache['heatmap']
            if 'pred_heatmap' in cache:
                outputs['pred_heatmap'] = cache['pred_heatmap']
            if 'pred_heatmap_acted' in cache:
                outputs['pred_heatmap_acted'] = cache['pred_heatmap_acted']
        else:
            outputs = dict()

        if flags.with_images:
            images = {
                self.pred_landmark_tag: visualize_in_row(((pred_landmark, image), 'points'))}
            if 'heatmap' in cache:
                images['heatmap'] = visualize_in_row(
                    (cache['heatmap'], 'BNHW'))
                images['heatmap_sum'] = visualize_in_row(
                    (cache['heatmap'].sum(1), 'BHW'))

            if 'pred_heatmap_acted' in cache:
                images['pred_heatmap_acted'] = visualize_in_row(
                    (cache['pred_heatmap_acted'], 'BNHW'))
                images['pred_heatmap_acted_sum'] = visualize_in_row(
                    (cache['pred_heatmap_acted'].sum(1), 'BHW'))
        else:
            images = dict()

        return loss, losses, outputs, images
