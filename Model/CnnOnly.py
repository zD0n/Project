# Pure CNN derived from Model/VitCnnGlobal.py
#
# VitCnnGlobal = CNNLocalFeature stem + global-attention Transformer.
# This model keeps the stem and the patch projection verbatim and replaces
# the Transformer with residual conv blocks, so it is a pure CNN with no
# attention, no CLS token and no position embedding.
#
# The token grid is the same one VitCnnGlobal projects (stride-1 throughout,
# patch_size grouping), so POOL=cls/POOL=attn in Run9.py pool over the exact
# tokens the cnn_vit arm does and the two arms stay comparable.
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class CNNLocalFeature(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) + x

class PureCNN(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, channels,
                 cnn_channels, dropout=0., emb_dropout=0., **kwargs):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width

        self.cnn = CNNLocalFeature(in_channels=channels, out_channels=cnn_channels)
        self.blocks = nn.Sequential(
            *[ConvBlock(cnn_channels, dropout=dropout) for _ in range(depth)]
        )
        self.to_token = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(cnn_channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
        )

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches

        self.dim = dim
        self.tokens = num_patches
        self.dropout = nn.Dropout(emb_dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward_tokens(self, img):
        x = self.cnn(img)
        x = self.blocks(x)
        return self.dropout(self.to_token(x))

    def forward_pooled(self, img):
        return self.forward_tokens(img).mean(dim=1)

    def forward(self, img):
        return self.mlp_head(self.forward_pooled(img))


"""
------------------------------------ Usage ------------------------------------

model = PureCNN(
    image_size=(128, 128),
    patch_size=(8, 8),
    num_classes=6,
    dim=256,
    depth=6,
    channels=1,
    cnn_channels=64,
    dropout=0.3,
    emb_dropout=0.1,
).to(device)

Run9.py: MODEL=cnn_only. PureCNN is its own backbone -- it exposes .dim,
.tokens, forward_tokens() and forward_pooled() -- so Run9 wraps it directly
instead of going through ViTBackbone. heads / mlp_dim / dim_head / pool are
accepted and ignored (see **kwargs), so a ViT config dict still constructs it.

---------------------------------------------------------------------------------
"""
