# Vision Transformer (2D window local attention + global attention in one
# trunk) helped by a CNN that injects local features into the patch
# embeddings. One transformer, one CLS token -- testing uses only the ViT
# path with its CLS token.
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class CNNLocalFeature(nn.Module):

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y) + y
        return self.conv3(y)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class DropPath(nn.Module):

    def __init__(self, p = 0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class Attention(nn.Module):
    # global attention: every token attends to every token
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LocalAttention(nn.Module):

    def __init__(self, dim, grid, heads = 8, dim_head = 64, window_size = 4, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.grid_h, self.grid_w = grid

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        xn = self.norm(x)
        b, n, d = xn.shape
        h = self.heads
        win = self.window_size

        # pad the grid up to a multiple of the window on both axes
        ph = (win - self.grid_h % win) % win
        pw = (win - self.grid_w % win) % win
        gh, gw = self.grid_h + ph, self.grid_w + pw

        grid = rearrange(xn[:, 1:], 'b (gh gw) d -> b gh gw d', gh = self.grid_h, gw = self.grid_w)
        grid = F.pad(grid, (0, 0, 0, pw, 0, ph))
        windows = rearrange(grid, 'b (hw w1) (vw w2) d -> b (hw vw) (w1 w2) d', w1 = win, w2 = win)

        qkv = self.to_qkv(windows).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (w1 w2) (h d) -> b h n (w1 w2) d',
                                          h = h, w1 = win, w2 = win), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n (w1 w2) d -> b n (w1 w2) (h d)',
                        h = h, w1 = win, w2 = win)
        out = self.to_out(out)

        out = rearrange(out, 'b (hw vw) (w1 w2) d -> b (hw w1) (vw w2) d',
                        w1 = win, w2 = win, hw = gh // win, vw = gw // win)
        out = out[:, :self.grid_h, :self.grid_w, :]
        out = rearrange(out, 'b gh gw d -> b (gh gw) d')

        full = x.new_zeros(b, n, d)
        full[:, 1:] = out
        return full

class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, grid,
                 window_size = 4, drop_path = 0.1, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalAttention(dim, grid, heads = heads, dim_head = dim_head,
                               window_size = window_size, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
                DropPath(drop_path),
                DropPath(drop_path),
                DropPath(drop_path),
            ]))

    def forward(self, x):
        for local_attn, global_attn, ff, dp_local, dp_global, dp_ff in self.layers:
            x = dp_local(local_attn(x)) + x
            x = dp_global(global_attn(x)) + x
            x = dp_ff(ff(x)) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 cnn_channels = 64, window_size = 4, drop_path = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        grid = (image_height // patch_height, image_width // patch_width)

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches = num_patches

        # patch embedding of the stem -- this is the token source the ViT itself uses
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))


        self.cnn = CNNLocalFeature(in_channels=channels, out_channels=cnn_channels)
        self.local_proj = nn.Linear(cnn_channels * patch_height * patch_width, dim)

        # single trunk with 2D local + global attention at every depth
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, grid,
                                       window_size = window_size,
                                       drop_path = drop_path, dropout = dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # help the ViT find local features: CNN features enter the same tokens
        vit_tokens = self.to_patch_embedding(img)
        cnn_feat = self.cnn(img)
        cnn_tokens = rearrange(cnn_feat, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                               p1 = self.patch_height, p2 = self.patch_width)
        x = vit_tokens + self.local_proj(cnn_tokens)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

"""
------------------------------------ Usage ------------------------------------

vit_model = ViT(
    image_size=(64, 128),
    patch_size=(8, 8),
    num_classes=len(emotion_map),
    dim=256,
    depth=6,
    heads=8,
    dim_head=64,
    mlp_dim=1024,
    channels=1,
    dropout=0.1,
    emb_dropout=0.1,
    cnn_channels=64,
    window_size=4,
    drop_path=0.1
).to(device)


---------------------------------------------------------------------------------
"""
