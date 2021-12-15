from collections import OrderedDict
import logging
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as checkpoint
from .utils.comm import comm
import numpy as np
from timm.models.layers import trunc_normal_, DropPath


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes *
                 self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]
                      * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, LayerNorm)):
            if comm.is_main_process():
                logging.debug('=> init {} gamma to 1'.format(m))
                logging.debug('=> init {} beta to 0'.format(m))
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            if comm.is_main_process():
                logging.debug(
                    '=> init weight of Linear/Conv2d from tranc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.debug('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3)
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def add_drop_path(self, drop_path):
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 use_checkpoint=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.use_checkpoint = use_checkpoint
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, drop_path=dpr[i])
            for i in range(layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if comm.is_main_process():
                logging.debug(
                    '=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.debug('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_all=False):
        all_x = []
        for i, blk in enumerate(self.resblocks):
            if self.training and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if return_all:
                all_x.append(x)
        if return_all:
            return all_x
        else:
            return x


class VisualTransformer(nn.Module):
    positional_embedding: nn.Parameter

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 pool_type: str = 'default',
                 skip_cls: bool = False,
                 drop_path_rate=0.,
                 **kwargs):
        super().__init__()
        self.pool_type = pool_type
        self.skip_cls = skip_cls
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.config = kwargs.get("config", None)
        self.sequence_length = (input_resolution // patch_size) ** 2 + 1
        self.conv_pool = None
        if (self.pool_type == 'linear'):
            if (not self.skip_cls):
                self.conv_pool = nn.Conv1d(
                    width, width, self.sequence_length, stride=self.sequence_length, groups=width)
            else:
                self.conv_pool = nn.Conv1d(
                    width, width, self.sequence_length-1, stride=self.sequence_length, groups=width)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                self.sequence_length, width
            )
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(
            width, layers, heads, drop_path_rate=drop_path_rate)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        if self.config is not None and self.config.MIM.ENABLE:
            logging.info("MIM ENABLED")
            self.mim = True
            self.lm_transformer = Transformer(
                width, self.config.MIM.LAYERS, heads)
            self.ln_lm = LayerNorm(width)
            self.lm_head = nn.Linear(width, self.config.MIM.VOCAB_SIZE)
            self.mask_token = nn.Parameter(scale * torch.randn(width))
        else:
            self.mim = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if comm.is_main_process():
                logging.debug(
                    '=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.debug('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, **kwargs):
        if "bool_masked_pos" in kwargs:
            return self.forward_mim(x, **kwargs)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if (self.pool_type == 'average'):
            if self.skip_cls:
                x = x[:, 1:, :]
            x = torch.mean(x, dim=1)
        elif (self.pool_type == 'linear'):
            if self.skip_cls:
                x = x[:, 1:, :]
            x = x.permute(0, 2, 1)
            x = self.conv_pool(x)
            x = x.permute(0, 2, 1).squeeze()
        else:
            x = x[:, 0, :]
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x

    def forward_mim(self, x: torch.Tensor, bool_masked_pos, return_all_tokens=False, disable_vlc=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        batch_size, seq_len, _ = x.size()
        mask_token = self.mask_token.unsqueeze(
            0).unsqueeze(0).expand(batch_size, seq_len, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        masked_x = x * (1 - w) + mask_token * w
        if disable_vlc:
            x = masked_x
            masked_start = 0
        else:
            x = torch.cat([x, masked_x], 0)
            masked_start = batch_size
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        masked_x = x[:, masked_start:]
        masked_x = self.lm_transformer(masked_x)
        masked_x = masked_x.permute(1, 0, 2)
        masked_x = masked_x[:, 1:]
        masked_x = self.ln_lm(masked_x)
        if not return_all_tokens:
            masked_x = masked_x[bool_masked_pos]
        logits = self.lm_head(masked_x)
        assert self.pool_type == "default"
        result = {"logits": logits}
        if not disable_vlc:
            x = x[0, :batch_size]
            x = self.ln_post(x)
            if self.proj is not None:
                x = x @ self.proj
            result["feature"] = x
        return result

    def extract(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, return_all=True)
        return x

    def encode_image_cls(self, image):
        y1 = self.extract(image)
        y1 = y1[-1][0]
        if hasattr(self, "head"):
            y1 = self.head(y1)
        return y1

    def encode_image_mean(self, image):
        y = self.extract(image)
        y1 = y[-1][1:].mean(0)
        if hasattr(self, "head"):
            y1 = self.head(y1)
        return y1

    def encode_image_both(self, image):
        y = self.extract(image)
        y1 = y[-1][1:].mean(0)
        y2 = y[-1][0]
        y = torch.cat([y1, y2], -1)
        if hasattr(self, "head"):
            y = self.head(y)
        return y


def load_farl(model_type, model_file) -> VisualTransformer:
    checkpoint = torch.load(model_file, map_location='cpu')
    if model_type == "base":
        model = VisualTransformer(
            input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512)
    elif model_type == "large":
        model = VisualTransformer(
            input_resolution=224, patch_size=16, width=1024, layers=24, heads=16, output_dim=512)
    elif model_type == "huge":
        model = VisualTransformer(
            input_resolution=224, patch_size=14, width=1280, layers=32, heads=16, output_dim=512)
    else:
        raise
    model.transformer.use_checkpoint = True
    state_dict = {}
    for name, weight in checkpoint["state_dict"].items():
        if name.startswith("visual"):
            state_dict[name[7:]] = weight
    inco = model.load_state_dict(state_dict, strict=False)
    # print(inco.missing_keys)
    assert len(inco.missing_keys) == 0
    return model
