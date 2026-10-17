import torch
from torch import nn
from einops import rearrange, repeat

class ConvNeXtTokenBlock(nn.Module):

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = rearrange(x, 'b c n -> b n c')
        x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = rearrange(x, 'b n c -> b c n')
        return residual + self.drop_path(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the token axis of an (b, dim, n) tensor.

    Mirrors the Attention used in the ViT variants: pre-norm, no bias on
    qkv, residual inside the block so the outer loop stays
    ``x = block(x)``.
    """

    def __init__(self, dim, heads=8, dim_head=64, drop_path=0.0, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = rearrange(x, 'b c n -> b n c')
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b n c -> b c n')
        return residual + self.drop_path(out)


class ConvNeXtClsModel(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, channels, drop_path,
                 heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        self.dim = dim
        target_h, target_w = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        self.tokens = (target_h // 8) * (target_w // 8)

        self.stem = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, dim, kernel_size=1)
        )
        # CLS
        self.cls_token = nn.Parameter(torch.randn(1, dim, 1))

        # Interleave conv token mixing with global self-attention: each stage
        # is ConvNeXt block (local, cheap) followed by attention (global).
        self.transformer = nn.ModuleList([
            block
            for _ in range(depth)
            for block in (
                ConvNeXtTokenBlock(dim=dim, drop_path=drop_path),
                SelfAttention(dim, heads=heads, dim_head=dim_head,
                              drop_path=drop_path, dropout=dropout),
            )
        ])
        
        self.final_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward_tokens(self, img):
        """Used natively when POOL='attn' -> extracts sequence patches minus CLS"""
        x = self.stem(img)
        b, c, h, w = x.shape
        
        # Flatten spatial layout
        spatial_tokens = rearrange(x, 'b c h w -> b c (h w)')
        
        # Prepend [CLS] token vector
        cls_tokens = repeat(self.cls_token, '1 c 1 -> b c 1', b=b)
        x = torch.cat((cls_tokens, spatial_tokens), dim=2)
        
        # Run mixing blocks
        for block in self.transformer:
            x = block(x)
            
        x = rearrange(x, 'b c n -> b n c')
        x = self.final_norm(x)
        # Drop index 0 ([CLS]) and return only the feature patch list
        return x[:, 1:]

    def forward(self, img):
        """Used natively when POOL='cls' -> extracts the CLS prediction vector"""
        x = self.stem(img)
        b, c, h, w = x.shape
        
        spatial_tokens = rearrange(x, 'b c h w -> b c (h w)')
        cls_tokens = repeat(self.cls_token, '1 c 1 -> b c 1', b=b)
        x = torch.cat((cls_tokens, spatial_tokens), dim=2)
        
        for block in self.transformer:
            x = block(x)
            
        x = rearrange(x, 'b c n -> b n c')
        x = self.final_norm(x)

        cls_out = x[:, 0]
        return self.mlp_head(cls_out)
