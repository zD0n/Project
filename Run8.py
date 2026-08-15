import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn, optim

from leaf_pytorch.frontend import Leaf
from Model import VitCnnGlobal, VitCnnLocal, VitGlobal, VitLocal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CREMAD_MAP = {"anger": 0, "disgust": 1, "fear": 2, "neutral": 3, "sad": 4, "happy": 5}
IEMOCAP_MAP = {"neutral": 0, "sad": 1, "anger": 2, "happy": 3, "excited": 4}
LABEL_ALIASES = {
    "sadness": "sad",
    "happiness": "happy",
    "excitement": "excited",
    "angry": "anger",
    "neutral state": "neutral",
}

DATASET_DIR = os.environ.get("DATASET_DIR", "Dataset2")
RESULT_DIR = "./results/LeafTorch_ViT"
SAMPLE_RATE = 16000


def _env(name, default, cast=float):
    return cast(os.environ.get(name, str(default)))


LEAF_N_FILTERS = _env("LEAF_N_FILTERS", 64, int)
TARGET_SIZE = _env("TARGET_SIZE", 64, int)
WINDOW_LEN = _env("WINDOW_LEN", 25, float)
LEAF_LR = _env("LEAF_LR", 1e-5)
LEARN_POOLING = _env("LEARN_POOLING", 0, int)   # 0 freezes the Gaussian lowpass
PCEN = _env("PCEN", 0, int)                     # 0 = log compression, as Run5/Run7

NUM_EPOCHS = _env("NUM_EPOCHS", 30, int)
BATCH_SIZE = _env("BATCH_SIZE", 32, int)
LR = _env("LR", 3e-4)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 0.05)
LABEL_SMOOTHING = _env("LABEL_SMOOTHING", 0.1)
WARMUP_EPOCHS = _env("WARMUP_EPOCHS", 5, int)

DIM = _env("DIM", 256, int)
DEPTH = _env("DEPTH", 6, int)
HEADS = _env("HEADS", 8, int)
MLP_DIM = _env("MLP_DIM", 1024, int)
PATCH = _env("PATCH", 8, int)
DIM_HEAD = _env("DIM_HEAD", 64, int)
CNN_CHANNELS = _env("CNN_CHANNELS", 64, int)
DROPOUT = _env("DROPOUT", 0.2)
EMB_DROPOUT = _env("EMB_DROPOUT", 0.1)

COORD_CHANNELS = _env("COORD_CHANNELS", 1, int)
SPEC_AUGMENT = _env("SPEC_AUGMENT", 1, int)
FREQ_MASK = _env("FREQ_MASK", 8, int)
TIME_MASK = _env("TIME_MASK", 16, int)
NORMALIZE = _env("NORMALIZE", 1, int)

# vit | vit_local | cnn_vit | cnn_vit_local | convnext  (see README)
MODEL = os.environ.get("MODEL", "cnn_vit").lower()
RESUME = _env("RESUME", 0, int)

# ConvNeXt only (MODEL=convnext); ignored by the ViT variants.
CONVNEXT_SIZE = os.environ.get("CONVNEXT_SIZE", "tiny").lower()
DROP_PATH = _env("DROP_PATH", 0.1)          # stochastic depth, ConvNeXt-T default
HEAD_INIT_SCALE = _env("HEAD_INIT_SCALE", 1.0)
CONVNEXT_PRETRAINED = _env("CONVNEXT_PRETRAINED", 0, int)  # downloads ImageNet weights
CONVNEXT_22K = _env("CONVNEXT_22K", 0, int)                # 22k instead of 1k weights

SPLIT_MODE = os.environ.get("SPLIT_MODE", "speaker")
TEST_FRAC = _env("TEST_FRAC", 0.2)
VAL_FRAC = _env("VAL_FRAC", 0.1)
SEED = _env("SEED", 99, int)
DATASET = os.environ.get("DATASET", "auto").lower()
VIT_CHANNELS = 1 + (2 if COORD_CHANNELS else 0)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no CUDA device visible.")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
labels_df = pd.read_csv(os.path.join(DATASET_DIR, "labels.csv"))
name_col = "Filename" if "Filename" in labels_df.columns else "filename"
labels_dict = dict(zip(labels_df[name_col], labels_df["Label"]))


def canon(label):
    s = str(label).strip().lower()
    return LABEL_ALIASES.get(s, s)


vocab = {canon(v) for v in labels_dict.values()}
if DATASET == "auto":
    DATASET = "iemocap" if vocab & {"excited", "frustration", "other"} else "cremad"

EMOTION_MAP = IEMOCAP_MAP if DATASET == "iemocap" else CREMAD_MAP
NUM_CLASSES = len(EMOTION_MAP)
RESULT_DIR = os.path.join(RESULT_DIR, DATASET)


def speaker_of(fname):
    """CREMA-D 1001_... -> '1001'; IEMOCAP Ses01F_impro01_F000 -> 'Ses01F'."""
    stem = os.path.splitext(fname)[0]
    if DATASET == "iemocap":
        parts = stem.split("_")
        return stem[:5] + (parts[-1][0] if parts[-1] else "")
    return stem.split("_")[0]


audio_dir = next(
    (os.path.join(DATASET_DIR, d) for d in ("audios", "AudioWAV")
     if os.path.isdir(os.path.join(DATASET_DIR, d))),
    DATASET_DIR,
)
all_wavs = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))

audio_files, skipped = [], {}
for f in all_wavs:
    lab = labels_dict.get(os.path.splitext(f)[0])
    if lab is None:
        skipped["<no label row>"] = skipped.get("<no label row>", 0) + 1
        continue
    c = canon(lab)
    if c in EMOTION_MAP:
        audio_files.append(f)
    else:
        skipped[c] = skipped.get(c, 0) + 1

print(f"Dataset: {DATASET} | {NUM_CLASSES} classes: {', '.join(EMOTION_MAP)}")
print(f"Usable clips: {len(audio_files)} of {len(all_wavs)}")
if skipped:
    print("  skipped labels: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
if not audio_files:
    raise SystemExit("No clips matched the class set -- check DATASET / labels.csv")

print(f"Loading {len(audio_files)} audio files...")
all_audio, all_labels = [], []
for fname in audio_files:
    base = os.path.splitext(fname)[0]
    all_labels.append(EMOTION_MAP[canon(labels_dict[base])])
    audio = sf.read(os.path.join(audio_dir, fname), dtype="float32")
    if isinstance(audio, tuple):
        audio = audio[0]
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    all_audio.append(audio)

all_labels = np.array(all_labels, dtype=np.int64)
lengths = np.array([a.shape[0] for a in all_audio])
print(f"Loaded {len(all_audio)} files | max length: {lengths.max()} samples | "
      f"total {lengths.sum() / SAMPLE_RATE:.1f}s")

# One fixed input length for every batch: variable shapes make the CUDA
# allocator grow without bound and epochs get slower and slower.
FIXED_SECONDS = _env("FIXED_SECONDS", 0.0)
if FIXED_SECONDS <= 0:
    FIXED_SECONDS = min(6.0, max(2.0, float(np.percentile(lengths, 75)) / SAMPLE_RATE))
FIXED_SAMPLES = int(round(FIXED_SECONDS * SAMPLE_RATE))
_trunc = int((lengths > FIXED_SAMPLES).sum())
print(f"Fixed input window: {FIXED_SECONDS:.2f}s ({FIXED_SAMPLES} samples) | "
      f"{_trunc} of {len(lengths)} clips cropped ({_trunc / len(lengths) * 100:.1f}%)")

# LEAF convolves at full waveform resolution before pooling to frames.
_peak_gb = BATCH_SIZE * 2 * LEAF_N_FILTERS * FIXED_SAMPLES * 4 / 1e9
print(f"LEAF peak intermediate: ~{_peak_gb:.2f} GB/copy "
      f"({BATCH_SIZE} x {2 * LEAF_N_FILTERS} x {FIXED_SAMPLES})")

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
actors = np.array([speaker_of(f) for f in audio_files])
rng = np.random.RandomState(SEED)

if SPLIT_MODE == "speaker":
    uniq = np.unique(actors)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * TEST_FRAC)))
    n_val = max(1, int(round(len(uniq) * VAL_FRAC)))
    test_a, val_a = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    test_idx = np.array([i for i, a in enumerate(actors) if a in test_a])
    val_idx = np.array([i for i, a in enumerate(actors) if a in val_a])
    train_idx = np.array([i for i, a in enumerate(actors)
                          if a not in test_a and a not in val_a])
    held_out = "{} test / {} val / {} train speakers".format(
        len(test_a), len(val_a), len(uniq) - len(test_a) - len(val_a))
else:
    p = rng.permutation(len(audio_files))
    n_test, n_val = int(round(len(p) * TEST_FRAC)), int(round(len(p) * VAL_FRAC))
    test_idx, val_idx, train_idx = p[:n_test], p[n_test:n_test + n_val], p[n_test + n_val:]
    held_out = "random split -- speakers appear in BOTH train and test"

print("\nSplit ({}): train {} | val {} | test {}".format(
    SPLIT_MODE, len(train_idx), len(val_idx), len(test_idx)))
print("  " + held_out)
if SPLIT_MODE == "speaker":
    ov = (set(actors[train_idx]) & set(actors[test_idx])) | \
         (set(actors[train_idx]) & set(actors[val_idx]))
    print("  speaker overlap train/eval: {} (must be 0)".format(len(ov)))


def make_batch(indices, train=False):
    """(B, FIXED_SAMPLES) -- one constant shape all run.

    Clips longer than the window are cropped at a random offset while training
    (free augmentation) and from the start at eval, so results are
    deterministic. Shorter clips are zero-padded.
    """
    batch = np.zeros((len(indices), FIXED_SAMPLES), dtype=np.float32)
    for j, i in enumerate(indices):
        x = all_audio[i]
        n = x.shape[0]
        if n > FIXED_SAMPLES:
            off = np.random.randint(0, n - FIXED_SAMPLES + 1) if train else 0
            batch[j] = x[off:off + FIXED_SAMPLES]
        else:
            batch[j, :n] = x
    return batch


# ---------------------------------------------------------------------------
# Frontend + feature pipeline
# ---------------------------------------------------------------------------
leaf = Leaf(
    n_filters=LEAF_N_FILTERS,
    sample_rate=SAMPLE_RATE,
    window_len=WINDOW_LEN,
    preemp=False,          # leaf_pytorch has no pre-emphasis layer
    init_min_freq=60.0,
    init_max_freq=7800.0,
    pcen_compression=bool(PCEN),
).to(device)

if not LEARN_POOLING:
    for p in leaf._pooling.parameters():
        p.requires_grad_(False)

leaf_weights_path = os.path.join(RESULT_DIR, "leaf_weights.pth")
os.makedirs(RESULT_DIR, exist_ok=True)
if RESUME and os.path.exists(leaf_weights_path):
    state = torch.load(leaf_weights_path, map_location=device)
    if all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        leaf.load_state_dict(state)
        print(f"Loaded LEAF weights from {leaf_weights_path}")
    else:
        print("REFUSED to resume: saved LEAF weights are non-finite.")
elif os.path.exists(leaf_weights_path):
    print("Ignoring existing leaf_weights.pth (set RESUME=1 to continue from it).")

_coord_cache = {}


def coord_planes(b, h, w):
    """CoordViT: normalized (time, freq) position as two extra input planes."""
    key = (h, w)
    if key not in _coord_cache:
        ys = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(1, 1, h, w)
        _coord_cache[key] = torch.cat([ys, xs], dim=1)
    return _coord_cache[key].expand(b, 2, h, w)


def spec_augment(x):
    """SCQT-MaxViT style masking. Multiplicative, so autograd stays intact."""
    b, _, h, w = x.shape
    mask = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
    for i in range(b):
        t = int(np.random.randint(0, TIME_MASK + 1))
        if t > 0 and h > t:
            t0 = int(np.random.randint(0, h - t))
            mask[i, :, t0:t0 + t, :] = 0.0
        f = int(np.random.randint(0, FREQ_MASK + 1))
        if f > 0 and w > f:
            f0 = int(np.random.randint(0, w - f))
            mask[i, :, :, f0:f0 + f] = 0.0
    return x * mask


def leaf_features(waveforms, train):
    """(B, T) numpy -> (B, VIT_CHANNELS, TARGET, TARGET) on `device`.

    leaf_pytorch emits (B, n_filters, frames) i.e. freq-major; the transpose
    keeps time on rows, matching Run5/Run6/Run7.
    """
    x = torch.from_numpy(waveforms).to(device, non_blocking=True).unsqueeze(1)
    feats = leaf(x)                                # (B, n_filters, frames)
    if not PCEN:
        feats = torch.log(feats + 1e-5)
    feats = feats.transpose(1, 2).unsqueeze(1)     # (B, 1, frames, n_filters)
    feats = F.interpolate(feats, size=(TARGET_SIZE, TARGET_SIZE),
                          mode="bilinear", align_corners=False)
    if NORMALIZE:
        mean = feats.mean(dim=(-2, -1), keepdim=True)
        std = feats.std(dim=(-2, -1), keepdim=True)
        feats = (feats - mean) / (std + 1e-5)
    if train and SPEC_AUGMENT:
        feats = spec_augment(feats)
    if COORD_CHANNELS:
        b, _, h, w = feats.shape
        feats = torch.cat([feats, coord_planes(b, h, w)], dim=1)
    return feats


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_MODELS = {"vit": VitGlobal, "vit_local": VitLocal,
           "cnn_vit": VitCnnGlobal, "cnn_vit_local": VitCnnLocal}

_CONVNEXT_SIZES = {
    "tiny":   dict(depths=[3, 3, 9, 3],  dims=[96, 192, 384, 768]),
    "small":  dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    "base":   dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large":  dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
    "xlarge": dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]),
}


def build_convnext():
    """ConvNeXt from the vendored ./ConvNeXt checkout, on LEAF features.

    The stem is stride-4 and each of the three later stages halves again, so
    the map is TARGET_SIZE/32 before global pooling -- 2x2 at the default
    TARGET_SIZE=64. Below 32 it collapses to nothing, hence the guard.

    in_chans follows VIT_CHANNELS, so COORD_CHANNELS works here too: the
    coordinate planes just become extra input channels on the stem.
    """
    from ConvNeXt.models.convnext import ConvNeXt, model_urls

    if CONVNEXT_SIZE not in _CONVNEXT_SIZES:
        raise SystemExit(f"CONVNEXT_SIZE={CONVNEXT_SIZE!r} unknown; "
                         f"choose one of {sorted(_CONVNEXT_SIZES)}")
    if TARGET_SIZE < 32:
        raise SystemExit(f"TARGET_SIZE={TARGET_SIZE} is too small for ConvNeXt "
                         "(4 stages downsample by 32x; use >= 32)")

    model = ConvNeXt(in_chans=VIT_CHANNELS, num_classes=NUM_CLASSES,
                     drop_path_rate=DROP_PATH, head_init_scale=HEAD_INIT_SCALE,
                     **_CONVNEXT_SIZES[CONVNEXT_SIZE])

    if CONVNEXT_PRETRAINED:
        key = "convnext_{}_{}".format(CONVNEXT_SIZE, "22k" if CONVNEXT_22K else "1k")
        if key not in model_urls:
            raise SystemExit(f"No published weights for {key} "
                             "(xlarge is 22k-only; set CONVNEXT_22K=1)")
        state = torch.hub.load_state_dict_from_url(model_urls[key], map_location="cpu")["model"]
        # Two mismatches against an ImageNet checkpoint: a 1000/21841-way head,
        # and an RGB stem when VIT_CHANNELS is 1 or 3+coords.
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        stem = "downsample_layers.0.0.weight"
        if state[stem].shape[1] != VIT_CHANNELS:
            # Collapse RGB to one filter, then spread it over the real channel
            # count so the summed stem response keeps its original scale.
            state[stem] = (state[stem].mean(dim=1, keepdim=True)
                           .repeat(1, VIT_CHANNELS, 1, 1) * (3.0 / VIT_CHANNELS))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded {key}: {len(missing)} missing, {len(unexpected)} unexpected "
              "(the head is expected to be missing)")
    return model


if MODEL == "convnext":
    vit_model = build_convnext().to(device)
    print(f"Model: convnext_{CONVNEXT_SIZE} (drop_path={DROP_PATH}, "
          f"pretrained={CONVNEXT_PRETRAINED})")
elif MODEL in _MODELS:
    _kw = dict(image_size=(TARGET_SIZE, TARGET_SIZE), patch_size=(PATCH, PATCH),
               num_classes=NUM_CLASSES, dim=DIM, depth=DEPTH, heads=HEADS,
               mlp_dim=MLP_DIM, channels=VIT_CHANNELS, pool="cls",
               dropout=DROPOUT, emb_dropout=EMB_DROPOUT)
    if MODEL.startswith("cnn_"):
        _kw["cnn_channels"] = CNN_CHANNELS
    if MODEL != "cnn_vit_local":
        _kw["dim_head"] = DIM_HEAD

    vit_model = _MODELS[MODEL].ViT(**_kw).to(device)
    print(f"Model: {MODEL} ({_MODELS[MODEL].__name__})")
else:
    raise SystemExit(f"MODEL={MODEL!r} unknown; "
                     f"choose one of {sorted(list(_MODELS) + ['convnext'])}")


leaf_trainable = [p for p in leaf.parameters() if p.requires_grad]
print(f"Trainable params -- leaf: {sum(p.numel() for p in leaf_trainable)}, "
      f"clf: {sum(p.numel() for p in vit_model.parameters())}")
print(f"Methods: coord={COORD_CHANNELS} specaug={SPEC_AUGMENT} norm={NORMALIZE} "
      f"pcen={PCEN}")

# One optimizer, one autograd graph -- the frontend trains with the classifier.
optimizer = optim.AdamW([
    {"params": list(vit_model.parameters()), "lr": LR, "weight_decay": WEIGHT_DECAY},
    {"params": leaf_trainable, "lr": LEAF_LR, "weight_decay": 0.0},
])
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def lr_scale(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / max(1, WARMUP_EPOCHS)
    p = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * p))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


@torch.no_grad()
def evaluate(indices):
    """WA (accuracy) and UAR (mean per-class recall)."""
    leaf.eval()
    vit_model.eval()
    preds = []
    for i in range(0, len(indices), BATCH_SIZE):
        idx = indices[i:i + BATCH_SIZE]
        out = vit_model(leaf_features(make_batch(idx), train=False))
        preds.append(out.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    tgts = all_labels[indices]
    wa = float((preds == tgts).mean())
    recalls = [float((preds[tgts == c] == c).mean())
               for c in range(NUM_CLASSES) if (tgts == c).any()]
    return wa, float(np.mean(recalls)), preds, tgts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
n_train = len(train_idx)
num_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\nTraining {NUM_EPOCHS} epoch(s) | {n_train} train samples | "
      f"batch_size={BATCH_SIZE} | {num_batches} batches/epoch\n")

best_path = os.path.join(RESULT_DIR, "best.pth")
train_log, best_uar = [], -1.0

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    leaf.train()
    vit_model.train()
    perm = train_idx[np.random.permutation(n_train)]
    epoch_loss, epoch_correct = 0.0, 0

    for b in range(num_batches):
        idx = perm[b * BATCH_SIZE:min((b + 1) * BATCH_SIZE, n_train)]
        bs = len(idx)

        x_in = leaf_features(make_batch(idx, train=True), train=True)
        target_t = torch.tensor(all_labels[idx], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        output = vit_model(x_in)
        loss = criterion(output, target_t)

        loss.backward()      # reaches the Gabor filters directly -- one graph
        optimizer.step()

        epoch_loss += loss.item() * bs
        epoch_correct += (output.argmax(1) == target_t).sum().item()

    scheduler.step()
    if device == "cuda":
        torch.cuda.empty_cache()
    avg_loss = epoch_loss / n_train
    acc = epoch_correct / n_train * 100
    val_wa, val_uar, _, _ = evaluate(val_idx)
    elapsed = time.time() - t0

    train_log.append({"epoch": epoch + 1, "loss": avg_loss, "train_acc": acc,
                      "val_wa": val_wa * 100, "val_uar": val_uar * 100,
                      "lr": optimizer.param_groups[0]["lr"],
                      "time_s": round(elapsed, 1)})

    star = ""
    if not np.isfinite(avg_loss):
        star = "  !! non-finite loss -- not checkpointing"
    elif val_uar > best_uar:
        best_uar = val_uar
        torch.save({"vit": vit_model.state_dict()}, best_path)
        torch.save(leaf.state_dict(), leaf_weights_path)
        star = "  <- best"
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]  loss={avg_loss:.4f}  train={acc:.1f}%  "
          f"val_WA={val_wa * 100:.1f}%  val_UAR={val_uar * 100:.1f}%  "
          f"time={elapsed:.1f}s{star}")

# ---------------------------------------------------------------------------
# Test on the best-val checkpoint
# ---------------------------------------------------------------------------
vit_model.load_state_dict(torch.load(best_path, map_location=device)["vit"])
leaf.load_state_dict(torch.load(leaf_weights_path, map_location=device))

test_wa, test_uar, test_preds, test_targets = evaluate(test_idx)
train_wa, _, _, _ = evaluate(train_idx)

print("\n" + "=" * 66)
print(f"TEST -- leaf_pytorch + {MODEL} | {DATASET} | {SPLIT_MODE} split | "
      f"{len(test_idx)} clips")
print(f"  WA  {test_wa * 100:.2f}%")
print(f"  UAR {test_uar * 100:.2f}%   (chance = {100 / NUM_CLASSES:.1f}%)")
print(f"  train WA {train_wa * 100:.2f}%  ->  generalization gap "
      f"{(train_wa - test_wa) * 100:.1f} points")
print("=" * 66)

rev_map = {v: k for k, v in EMOTION_MAP.items()}
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for t, p in zip(test_targets, test_preds):
    cm[t][p] += 1
cm_df = pd.DataFrame(cm,
                     index=[rev_map[i] for i in range(NUM_CLASSES)],
                     columns=[rev_map[i] for i in range(NUM_CLASSES)])
cm_df.index.name = "True \\ Pred"
print("\nConfusion Matrix (test):")
print(cm_df.to_string())

cm_df.to_csv(os.path.join(RESULT_DIR, "confusion_matrix.csv"))
pd.DataFrame(train_log).to_csv(os.path.join(RESULT_DIR, "training_log.csv"), index=False)

run = {"frontend": "leaf_pytorch", "model": MODEL, "dataset": DATASET,
       "classes": NUM_CLASSES, "split_mode": SPLIT_MODE, "epochs": NUM_EPOCHS,
       "batch": BATCH_SIZE, "lr": LR, "leaf_lr": LEAF_LR,
       "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
       "dropout": DROPOUT, "emb_dropout": EMB_DROPOUT, "dim": DIM, "depth": DEPTH,
       "heads": HEADS, "mlp_dim": MLP_DIM, "patch": PATCH, "dim_head": DIM_HEAD,
       "cnn_channels": CNN_CHANNELS, "target_size": TARGET_SIZE,
       "leaf_filters": LEAF_N_FILTERS, "learn_pooling": LEARN_POOLING, "pcen": PCEN,
       "coord_channels": COORD_CHANNELS, "spec_augment": SPEC_AUGMENT,
       "freq_mask": FREQ_MASK, "time_mask": TIME_MASK, "normalize": NORMALIZE,
       "convnext_size": CONVNEXT_SIZE if MODEL == "convnext" else "",
       "drop_path": DROP_PATH if MODEL == "convnext" else "",
       "convnext_pretrained": CONVNEXT_PRETRAINED if MODEL == "convnext" else "",
       "fixed_seconds": FIXED_SECONDS, "seed": SEED,
       "best_val_uar": best_uar * 100, "test_wa": test_wa * 100,
       "test_uar": test_uar * 100, "train_wa": train_wa * 100}
sweep = os.path.join(RESULT_DIR, "sweep.csv")
pd.DataFrame([run]).to_csv(sweep, mode="a", header=not os.path.exists(sweep), index=False)

print(f"\nSaved to {RESULT_DIR}  (results appended to sweep.csv)")
