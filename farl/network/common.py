from typing import List, Optional, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from blueprint.ml import Logger


def load_checkpoint(net: nn.Module, checkpoint_path: str, network_name: str):
    states = torch.load(open(checkpoint_path, 'rb'), map_location={
                        'cuda:0': f'cuda:{torch.cuda.current_device()}'})
    network_states = states['networks']
    net.load_state_dict(network_states[network_name])
    return net


class Activation(nn.Module):
    def __init__(self, name: Optional[str], **kwargs):
        super().__init__()
        if name == 'relu':
            self.fn = F.relu
        elif name == 'softplus':
            self.fn = F.softplus
        elif name == 'gelu':
            self.fn = F.gelu
        elif name == 'sigmoid':
            self.fn = torch.sigmoid
        elif name == 'sigmoid_x':
            self.epsilon = kwargs.get('epsilon', 1e-3)
            self.fn = lambda x: torch.clamp(
                x.sigmoid() * (1.0 + self.epsilon*2.0) - self.epsilon,
                min=0.0, max=1.0)
        elif name == None:
            self.fn = lambda x: x
        else:
            raise RuntimeError(f'Unknown activation name: {name}')

    def forward(self, x):
        return self.fn(x)


class MLP(nn.Module):
    def __init__(self, channels: List[int], act: Optional[str]):
        super().__init__()
        assert len(channels) > 1
        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.Linear(channels[i], channels[i+1]))
            if i+1 < len(channels):
                layers.append(Activation(act))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Residual(nn.Module):
    def __init__(self, net: nn.Module, res_weight_init: Optional[float] = 0.0):
        super().__init__()
        self.net = net
        if res_weight_init is not None:
            self.res_weight = nn.Parameter(torch.tensor(res_weight_init))
        else:
            self.res_weight = None

    def forward(self, x):
        if self.res_weight is not None:
            return self.res_weight * self.net(x) + x
        else:
            return self.net(x) + x


class SE(nn.Module):
    def __init__(self, channel: int, r: int = 1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channel, channel//r, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(channel//r, channel, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: b x channel x h x w
        v = x.mean([2, 3], keepdim=True)  # b x channel x 1 x 1
        v = self.branch(v)  # b x channel x 1 x 1
        return x * v


def verbose_execution(model: nn.Module, logger: Optional[Logger]):
    _print = print if logger is None else logger.log_info
    for name, layer in model.named_children():
        layer._layer_name_ = name
        layer.register_forward_hook(
            lambda layer, _, output: _print(
                f"{layer._layer_name_}: shape={output.shape}, mean={output.mean().item()},"
                f" max={output.max().values.item()}, min={output.min().values.item()}")
        )
    return model
