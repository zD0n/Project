# Fixed & Vectorized Vision Transformer with CNN Local Attention
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class CNNLocalFeature(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LocalAttention(nn.Module):
    def __init__(self, dim, heads=8, window_size=3, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.window_size = window_size

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.scale = (dim // heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.norm(x)
        b, n, d = x.shape
        h = self.heads
        dim_head = d // h

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        # Vectorized Sliding Window Attention (No Python Loop)
        pad = self.window_size // 2
        k_padded = F.pad(k, (0, 0, pad, pad))  # Pad along sequence dimension
        v_padded = F.pad(v, (0, 0, pad, pad))

        # Unfold sequence into local sliding windows: (b, h, n, window_size, dim_head)
        k_windows = k_padded.unfold(dimension=2, size=self.window_size, step=1).transpose(-1, -2)
        v_windows = v_padded.unfold(dimension=2, size=self.window_size, step=1).transpose(-1, -2)

        # q shape expanded: (b, h, n, 1, dim_head)
        q_exp = q.unsqueeze(3)

        # Compute dot product attention across local window
        dots = torch.matmul(q_exp, k_windows.transpose(-1, -2)).squeeze(3) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Pool local values according to attention weights
        out = torch.matmul(attn.unsqueeze(3), v_windows).squeeze(3)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LocalAttention(
                            dim=dim, heads=heads, window_size=3, dropout=dropout
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels,
        dropout,
        emb_dropout,
        cnn_channels,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.cnn = CNNLocalFeature(in_channels=channels, out_channels=cnn_channels)
        self.linear_proj = nn.Linear(
            cnn_channels * patch_height * patch_width, dim
        )

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(channels * patch_height * patch_width),
            nn.Linear(channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
        )

        self.dropout = nn.Dropout(emb_dropout)

        # Added CLS Token to support Run9.py's ViTBackbone hook
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        cnn_feat = self.cnn(img)

        ph, pw = self.patch_height, self.patch_width
        cnn_patches = rearrange(
            cnn_feat,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=ph,
            p2=pw,
        )
        cnn_tokens = self.linear_proj(cnn_patches)
        vit_tokens = self.to_patch_embedding(img)

        # Combine CNN local features and linear patch embeddings
        x = vit_tokens + cnn_tokens
        b, n, _ = x.shape

        # Prepend CLS token to match ViT standard format
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # Output pooling (matches Run9.py backbone integration)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)