# Vision Transformer with CNN Global
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ConvNeXtBlock(nn.Module):
    """Channel-preserving block: dim -> dim. Needs a stem beforehand to set dim."""
    def __init__(self, dim, expansion_ratio=4, dropout=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, H, W)
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)      # -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)      # -> (B, C, H, W)
        return residual + x


class CNNLocalFeature(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_blocks=2, expansion_ratio=4, dropout=0.):
        super().__init__()
        # stem: project to out_channels once, since depthwise blocks can't change channel count
        self.stem = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(out_channels, expansion_ratio=expansion_ratio, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.stem(x)
        return self.blocks(x)

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
    
class LocalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, window_size=3, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x):
        x = self.norm(x)
        b, n, d = x.shape
        h = self.heads
        dim_head = d // h
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        pad = self.window_size // 2
        k = torch.cat([torch.zeros(b, h, pad, dim_head, device=x.device), k,
                       torch.zeros(b, h, pad, dim_head, device=x.device)], dim=2)
        v = torch.cat([torch.zeros(b, h, pad, dim_head, device=x.device), v,
                       torch.zeros(b, h, pad, dim_head, device=x.device)], dim=2)

        out = []
        for i in range(n):
            k_local = k[:, :, i:i+self.window_size, :]
            v_local = v[:, :, i:i+self.window_size, :]
            attn = torch.matmul(q[:, :, i:i+1, :], k_local.transpose(-1, -2)) * self.scale
            attn = self.attend(attn)
            attn = self.dropout(attn)
            out.append(torch.matmul(attn, v_local))
        out = torch.cat(out, dim=2)
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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels, dim_head, dropout, emb_dropout, cnn_channels):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.cnn = CNNLocalFeature(in_channels=channels, out_channels=cnn_channels)
        self.linear_proj = nn.Linear(cnn_channels * patch_height * patch_width, dim)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_height * patch_width),
            nn.Linear(channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        cnn_feat = self.cnn(img)
        B, C, H, W = cnn_feat.shape
        ph, pw = self.patch_height, self.patch_width
        cnn_patches = rearrange(cnn_feat, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw)
        cnn_tokens = self.linear_proj(cnn_patches)
        vit_tokens = self.to_patch_embedding(img)
        x = vit_tokens + cnn_tokens  
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


"""
------------------------------------ Usage ------------------------------------

vit_model = ViTWithCNN(
    image_size=(64, 128),
    patch_size=(8, 8),
    num_classes=len(emotion_map),
    dim=256,
    depth=6,
    dim_head=64,
    heads=8,
    mlp_dim=1024,
    channels=1,
    dropout=0.1,
    emb_dropout=0.1,
    cnn_channels=64
).to(device)

---------------------------------------------------------------------------------
"""