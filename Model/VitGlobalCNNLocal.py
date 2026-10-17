# Vision Transformer (global) + CNN (local) fused by a Transformer
import torch
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
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

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

class Attention(nn.Module):
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

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,
                 cnn_channels = 64, global_depth = None, fusion_depth = None,
                 fusion_heads = None, fusion_mlp_dim = None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        global_depth = depth if global_depth is None else global_depth
        fusion_depth = depth if fusion_depth is None else fusion_depth
        fusion_heads = heads if fusion_heads is None else fusion_heads
        fusion_mlp_dim = mlp_dim if fusion_mlp_dim is None else fusion_mlp_dim

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches = num_patches

        # global branch -- ViT with global attention
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.global_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.global_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.global_transformer = Transformer(dim, global_depth, heads, dim_head, mlp_dim, dropout)

        # local branch -- CNN feature map patchified into tokens
        self.cnn = CNNLocalFeature(in_channels=channels, out_channels=cnn_channels)
        self.local_proj = nn.Linear(cnn_channels * patch_height * patch_width, dim)
        self.local_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        # fusion -- CLS + global tokens + local tokens, tagged by branch
        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.type_embedding = nn.Parameter(torch.randn(2, dim))
        self.transformer = Transformer(dim, fusion_depth, fusion_heads, dim_head, fusion_mlp_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        b = img.shape[0]
        ph, pw = self.patch_height, self.patch_width

        # global features
        global_tokens = self.to_patch_embedding(img)
        global_cls = repeat(self.global_cls_token, '1 1 d -> b 1 d', b = b)
        global_tokens = torch.cat((global_cls, global_tokens), dim=1)
        global_tokens = global_tokens + self.global_pos_embedding
        global_tokens = self.dropout(global_tokens)
        global_tokens = self.global_transformer(global_tokens)

        # local features
        cnn_feat = self.cnn(img)
        local_tokens = rearrange(cnn_feat, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = ph, p2 = pw)
        local_tokens = self.local_proj(local_tokens) + self.local_pos_embedding
        local_tokens = self.dropout(local_tokens)

        # fusion
        fusion_cls = repeat(self.fusion_cls_token, '1 1 d -> b 1 d', b = b)
        global_tokens = global_tokens + self.type_embedding[0]
        local_tokens = local_tokens + self.type_embedding[1]
        x = torch.cat((fusion_cls, global_tokens, local_tokens), dim=1)

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
    cnn_channels=64
).to(device)

Optional: global_depth / fusion_depth / fusion_heads / fusion_mlp_dim override the
shared depth / heads / mlp_dim for each stage separately. The fusion Transformer
is exposed as .transformer and the classifier as .mlp_head, as in Model/VitGlobal.py.

---------------------------------------------------------------------------------
"""
