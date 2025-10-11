#Scattering Transform
import os
import torch
from kymatio import Scattering1D
import librosa
import numpy as np
import torch.nn.functional as F

# ===========================
# Parameters
# ===========================
T = 16000          # 1 second of audio at 16kHz
J = 6              # scattering scale
Q = 8              # wavelets per octave
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize scattering
scattering = Scattering1D(J=J, shape=T, Q=Q, frontend='torch')
scattering = scattering.to(device)

# ===========================
# Paths
# ===========================
audio_dir = r"C:\Coding\emodb\wav"  # folder containing your audio files

# ===========================
# Collect all scattering features
# ===========================
vit_features = []
labels = []

emo_map = {
    "W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
    "F": "happiness", "T": "sadness", "N": "neutral"
}
idx2label = {i: emo for i, emo in enumerate(sorted(emo_map.values()))}
label2idx = {v: k for k, v in idx2label.items()}

for filename in os.listdir(audio_dir):
    if not filename.lower().endswith(".wav"):
        continue

    audio_path = os.path.join(audio_dir, filename)
    print(f"Processing {filename} ...")

    # ---------------------------
    # Load audio
    # ---------------------------
    y, sr = librosa.load(audio_path, sr=16000)  # force 16kHz
    if len(y) < T:
        y = np.pad(y, (0, T - len(y)))
    else:
        y = y[:T]

    # ---------------------------
    # Scattering transform
    # ---------------------------
    x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T]
    Sx = scattering(x)  # [1, channels, time]

    # ---------------------------
    # Prepare for ViT: resize to 256x256 and repeat channels
    # ---------------------------
    Sx = Sx.unsqueeze(1)  # [1, 1, channels, time]
    Sx_resized = F.interpolate(Sx, size=(256, 256), mode="bilinear")
    Sx_resized = Sx_resized.repeat(1, 3, 1, 1)  # [1, 3, 256, 256]
    Sx_resized = Sx_resized.squeeze(0)          # [3, 256, 256]

    vit_features.append(Sx_resized.cpu())

    # ---------------------------
    # Label from filename
    # EMODB normal format: 03a01Fa.wav → 'F'
    # ---------------------------
    emo_code = filename[5]
    labels.append(label2idx[emo_map[emo_code]])

# ===========================
# Final dataset tensors
# ===========================
X = torch.stack(vit_features)  # [num_samples, 3, 256, 256]
y = torch.tensor(labels)       # [num_samples]

print("Final dataset shapes:")
print("  X:", X.shape)
print("  y:", y.shape)


# VIT WITH CNN BUT BIGGER
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ---------------------------
# Helper
# ---------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ---------------------------
# CNN for local features (flexible input channels)
# ---------------------------
class CNNLocalFeature(nn.Module):
    """
    CNN with numeric channel growth: in_channels -> 32 -> 64 -> 128
    Can accept Mel (1 channel) or LEAF (n_filters channels)
    """
    def __init__(self, in_channels=3, dropout=0.05):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)  # [B, 32, H, W]
        x = self.conv2(x)  # [B, 64, H, W]
        x = self.conv3(x)  # [B, 128, H, W]
        x = self.conv4(x)  # [B, 128, H, W]
        return x

# ---------------------------
# Transformer blocks
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LocalAttention(dim, heads=heads, dim_head=dim_head, window_size=3, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# ---------------------------
# ViT with CNN fusion (supports Mel or LEAF)
# ---------------------------
class ViTWithCNN(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, in_channels=1, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "image dimensions must be divisible by the patch size."

        self.patch_height = patch_height
        self.patch_width = patch_width

        # CNN: in_channels -> 32 -> 64 -> 128
        self.cnn = CNNLocalFeature(in_channels=in_channels, dropout=0.05)
        self.linear_proj = nn.Linear(128 * patch_height * patch_width, dim)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches

        # ViT patch embedding from raw input
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(in_channels * patch_height * patch_width),
            nn.Linear(in_channels * patch_height * patch_width, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape  # x: [B, in_channels, H, W]

        # CNN features
        cnn_feat = self.cnn(x)  # [B, 128, H, W]
        ph, pw = self.patch_height, self.patch_width
        cnn_patches = rearrange(cnn_feat, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=ph, p2=pw)
        cnn_tokens = self.linear_proj(cnn_patches)

        # ViT patch embedding
        vit_tokens = self.to_patch_embedding(x)

        # merge CNN + ViT tokens
        x = vit_tokens + cnn_tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(x.size(1))]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()

        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True


#Set up model
from torch.utils.data import TensorDataset, DataLoader, Subset

dataset = TensorDataset(X, y)
labels_np = y.cpu().numpy()

num_classes = len(torch.unique(y))
num_train_per_class = 30
num_test_per_class = 10

train_indices, test_indices = [], []

for cls in range(num_classes):
    cls_idx = np.where(labels_np == cls)[0]
    np.random.shuffle(cls_idx)
    train_indices.extend(cls_idx[:num_train_per_class].tolist())
    test_indices.extend(cls_idx[num_train_per_class:num_train_per_class+num_test_per_class].tolist())

train_dataset = Subset(dataset, train_indices)
test_dataset  = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# ===========================
# Model init
# ===========================
vit_model = ViTWithCNN(
    image_size=256,
    patch_size=32,
    num_classes=num_classes,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    in_channels=3,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

# ===========================
# Training setup
# ===========================
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)
epochs = 30
early_stopping = EarlyStopping(patience=5, verbose=True)

# ===========================
# Training loop
# ===========================
for epoch in range(epochs):
    vit_model.train()
    total_loss, correct_train, total_train = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vit_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    train_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    # Validation
    vit_model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vit_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
    val_loss /= len(test_loader)
    val_acc = correct_val / total_val

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    early_stopping(val_loss, vit_model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

vit_model.load_state_dict(early_stopping.best_state)
print("Best model loaded.")

# ===========================
# Evaluation
# ===========================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

vit_model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vit_model(inputs)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(num_classes)]))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[idx2label[i] for i in range(num_classes)])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
