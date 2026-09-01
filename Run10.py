# =============================================================================
# Run10 -- the LEAF freeze/unfreeze ablation.
#
# Run8 with one thing added: every parameter group inside the LEAF frontend can
# be frozen on its own. Everything else -- data, splits, classifiers, schedule,
# sweep.csv contract -- is Run8 unchanged, so a Run10 row and a Run8 row of the
# same configuration are comparable.
#
# THE QUESTION. LEAF's claim is that a frontend learned end to end beats a fixed
# one. "Learned" is not one thing though: LEAF has four separate parameter
# groups, and the claim is only interesting if it says which of them carry it.
# Freezing a group leaves the op in the forward pass at its initialization, so
# each cell isolates the value of *learning* that stage rather than the value of
# the stage existing at all.
#
# The four axes are LEARN_FILTERS, LEARN_POOLING, LEARN_COMPRESSION and
# LEARN_SMOOTHING -- see the block where they are read for what each one owns.
# 2^4 = 16 cells; H100_Set_up/run_leaf_ablation.sh runs them.
#
# 1111 is full LEAF. 0000 is LEAF frozen at initialization -- a fixed Gabor
# filterbank, which is the honest baseline for the claim and a different thing
# from FRONTEND=mel.
# =============================================================================
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
SAMPLE_RATE = 16000

# leaf = learnable LEAF frontend (trains with the classifier).
# mel  = fixed log-mel filterbank, the classical baseline LEAF is measured
#        against. Same output contract, same downstream pipeline, so the two
#        differ only in the frontend.
FRONTEND = os.environ.get("FRONTEND", "leaf").lower()
if FRONTEND not in ("leaf", "mel"):
    raise SystemExit(f"FRONTEND={FRONTEND!r} unknown; choose 'leaf' or 'mel'")
# Separate result dirs, or a mel run overwrites a leaf run's checkpoints and
# appends to its sweep.csv.
#
# RESULTS_ROOT moves the whole tree without disturbing that separation, so a
# grid run under one banner keeps its rows out of the main results/ -- e.g.
# RESULTS_ROOT="results/Ablation Study" writes
# "results/Ablation Study/{Leaf,Mel}Torch_ViT/<dataset>/sweep.csv". Only the
# root moves: frontend and dataset still get their own folder underneath.
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "./results")
RESULT_DIR = os.path.join(
    RESULTS_ROOT, "{}Torch_ViT".format("Leaf" if FRONTEND == "leaf" else "Mel"))


def _env(name, default, cast=float):
    return cast(os.environ.get(name, str(default)))


LEAF_N_FILTERS = _env("LEAF_N_FILTERS", 64, int)
TARGET_SIZE = _env("TARGET_SIZE", 64, int)
WINDOW_LEN = _env("WINDOW_LEN", 25, float)
LEAF_LR = _env("LEAF_LR", 1e-5)
# ---------------------------------------------------------------------------
# The LEAF ablation axes
# ---------------------------------------------------------------------------
# Leaf.forward (leaf_pytorch/frontend.py) is four ops, three of which hold
# parameters -- and the third holds two groups that do unrelated jobs:
#
#   filterbank    _complex_conv       Gabor bandpass. The parameters are each
#                                     filter's centre frequency and bandwidth,
#                                     i.e. where the frontend listens.
#   squared mod   _activation         fixed, no parameters. Nothing to ablate.
#   pooling       _pooling            Gaussian lowpass. Per-channel width and
#                                     bias, i.e. the time-frequency tradeoff.
#   compression   _compression        PCEN alpha/delta/root -- the shape of the
#                                     static compression curve.
#   smoothing     _compression.ema    PCEN's EMA coefficient, the "s" in sPCEN.
#                                     The time constant of the gain control,
#                                     which is a different decision from the
#                                     curve shape above, so it gets its own
#                                     switch rather than riding along with it.
#
# 1 = trains with the classifier at LEAF_LR. 0 = stays at its initialization.
LEARN_FILTERS = _env("LEARN_FILTERS", 1, int)
LEARN_POOLING = _env("LEARN_POOLING", 1, int)
LEARN_COMPRESSION = _env("LEARN_COMPRESSION", 1, int)
LEARN_SMOOTHING = _env("LEARN_SMOOTHING", 1, int)

# PCEN defaults to 1 here, unlike Run8. With log compression Leaf builds no
# _compression module at all, so the last two axes would have nothing to freeze
# and the 16-cell factorial would quietly become 4 configurations run 4x each --
# wasted GPU hours and an ablation table with duplicate rows.
PCEN = _env("PCEN", 1, int)
if not PCEN and not (LEARN_COMPRESSION and LEARN_SMOOTHING):
    raise SystemExit(
        "PCEN=0 removes the compression stage, so LEARN_COMPRESSION=0 or "
        "LEARN_SMOOTHING=0 has nothing to freeze and this cell would duplicate "
        "another. Use PCEN=1 to sweep the compression axes.")

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

# vit | vit_local | cnn_vit | cnn_vit_local | convnext | vit_b16  (see README)
MODEL = os.environ.get("MODEL", "cnn_vit").lower()
RESUME = _env("RESUME", 0, int)

# Split one NUM_EPOCHS schedule across several jobs. NUM_EPOCHS stays the TOTAL
# (e.g. 400); EPOCHS_PER_RUN caps how many this process does before saving and
# exiting (e.g. 100, so four jobs finish the schedule).
#
# This is not the same as running 100 epochs four times: the cosine LR schedule
# and the optimizer moments are defined over the full NUM_EPOCHS and restored
# between jobs, so the result matches one uninterrupted 400-epoch run. Four
# independent 100-epoch runs would replay warmup and the whole cosine decay
# each time, which is a different experiment.
EPOCHS_PER_RUN = _env("EPOCHS_PER_RUN", 0, int)   # 0 = run the whole schedule

# ViT-B/16 only (MODEL=vit_b16); ignored elsewhere. This is the arm that can
# start from ImageNet weights -- the hand-rolled ViTs in Model/ have no
# published checkpoint at any shape, so scratch is their only option.
VIT_PRETRAINED = _env("VIT_PRETRAINED", 0, int)
VIT_TIMM_NAME = os.environ.get("VIT_TIMM_NAME", "vit_base_patch16_224.augreg_in21k")

# ConvNeXt only (MODEL=convnext); ignored by the ViT variants.
CONVNEXT_SIZE = os.environ.get("CONVNEXT_SIZE", "tiny").lower()
DROP_PATH = _env("DROP_PATH", 0.1)          # stochastic depth, ConvNeXt-T default
HEAD_INIT_SCALE = _env("HEAD_INIT_SCALE", 1.0)
CONVNEXT_PRETRAINED = _env("CONVNEXT_PRETRAINED", 0, int)  # downloads ImageNet weights
CONVNEXT_22K = _env("CONVNEXT_22K", 0, int)                # 22k instead of 1k weights

SPLIT_MODE = os.environ.get("SPLIT_MODE", "speaker")
TEST_FRAC = _env("TEST_FRAC", 0.2)
VAL_FRAC = _env("VAL_FRAC", 0.1)
# <1 keeps only that fraction of the training clips. val/test are untouched, so
# a small-data run is still scored against the same held-out speakers.
TRAIN_FRAC = _env("TRAIN_FRAC", 1.0)
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

if TRAIN_FRAC < 1.0:
    _full = len(train_idx)
    keep = max(NUM_CLASSES, int(round(_full * TRAIN_FRAC)))
    train_idx = train_idx[rng.permutation(_full)[:keep]]
    print("\nTRAIN_FRAC={}: training on {} of {} clips".format(
        TRAIN_FRAC, len(train_idx), _full))

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
class MelFrontend(nn.Module):
    """Fixed log-mel filterbank with the same contract as Leaf:

        (B, 1, T) -> (B, n_filters, frames), strictly positive.

    Deliberately not learnable -- that is the point of the comparison. It is
    still a torch module on `device`, so the batch never leaves the GPU and the
    rest of the pipeline is byte-for-byte the same code as the LEAF path.
    """

    def __init__(self, n_filters, sample_rate, window_len, window_stride=10.0):
        super().__init__()
        import torchaudio
        # LEAF convolves over a 25 ms window and pools to a 10 ms hop; match
        # both so the two frontends emit the same number of frames.
        win = int(round(sample_rate * window_len / 1000.0))
        n_fft = 1 << (win - 1).bit_length()          # next power of two >= win
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win,
            hop_length=int(round(sample_rate * window_stride / 1000.0)),
            n_mels=n_filters,
            f_min=60.0,          # same band as the LEAF Gabor init
            f_max=7800.0,
            power=2.0,
        )

    def forward(self, x):
        # 1e-6 keeps the log downstream finite on digital silence.
        return self.mel(x.squeeze(1)) + 1e-6


if FRONTEND == "mel":
    leaf = MelFrontend(
        n_filters=LEAF_N_FILTERS,
        sample_rate=SAMPLE_RATE,
        window_len=WINDOW_LEN,
    ).to(device)
else:
    leaf = Leaf(
        n_filters=LEAF_N_FILTERS,
        sample_rate=SAMPLE_RATE,
        window_len=WINDOW_LEN,
        preemp=False,          # leaf_pytorch has no pre-emphasis layer
        init_min_freq=60.0,
        init_max_freq=7800.0,
        pcen_compression=bool(PCEN),
    ).to(device)

    # requires_grad_(False), not deleting the module: the stage still runs, it
    # just stays at its initialization. Removing the op instead would confound
    # "is learning this stage worth it" with "is this stage worth it" -- two
    # different questions, and only the first is this ablation's.
    _frozen = []
    if not LEARN_FILTERS:
        _frozen.append(("filterbank", list(leaf._complex_conv.parameters())))
    if not LEARN_POOLING:
        _frozen.append(("pooling", list(leaf._pooling.parameters())))
    if leaf._compression is not None:
        if not LEARN_COMPRESSION:
            # alpha/delta/root only. The EMA coefficient lives under .ema and
            # is the smoothing axis, so it is deliberately named out here --
            # _compression.parameters() would have swept it up too.
            _frozen.append(("compression", [leaf._compression.alpha,
                                            leaf._compression.delta,
                                            leaf._compression.root]))
        if not LEARN_SMOOTHING:
            _frozen.append(("smoothing", list(leaf._compression.ema.parameters())))
    for _stage, _ps in _frozen:
        for p in _ps:
            p.requires_grad_(False)
    print("LEAF frozen: " + (", ".join(s for s, _ in _frozen) or "nothing"))

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
    # PCEN is a LEAF option; mel always log-compresses, otherwise raw power
    # spectra would reach the classifier.
    if FRONTEND == "mel" or not PCEN:
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


def build_vit_b16():
    """timm ViT-B/16 -- the arXiv 2502.12379 replication arm.

    timm does the two pieces of surgery an ImageNet checkpoint needs here:
    `in_chans` folds the RGB patch-embedding stem down to VIT_CHANNELS, and
    `num_classes` drops the 21k head for a fresh NUM_CLASSES one. The same call
    builds the scratch arm, so the two differ only in VIT_PRETRAINED and
    nothing else can drift between them.

    `img_size` is passed through rather than pinned to 224: timm interpolates
    the position embeddings, so TARGET_SIZE=64 also works (16 tokens).
    """
    import timm

    if TARGET_SIZE % 16:
        raise SystemExit(f"TARGET_SIZE={TARGET_SIZE} is not a multiple of the "
                         "16-pixel patch; use 64, 128, 224, ...")

    model = timm.create_model(VIT_TIMM_NAME, pretrained=bool(VIT_PRETRAINED),
                              num_classes=NUM_CLASSES, in_chans=VIT_CHANNELS,
                              img_size=TARGET_SIZE, drop_rate=DROPOUT)
    print("Model: {} ({}) | {} tokens".format(
        VIT_TIMM_NAME,
        "ImageNet pretrained" if VIT_PRETRAINED else "scratch",
        (TARGET_SIZE // 16) ** 2))
    return model


if MODEL == "convnext":
    vit_model = build_convnext().to(device)
    print(f"Model: convnext_{CONVNEXT_SIZE} (drop_path={DROP_PATH}, "
          f"pretrained={CONVNEXT_PRETRAINED})")
elif MODEL == "vit_b16":
    vit_model = build_vit_b16().to(device)
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
                     f"choose one of {sorted(list(_MODELS) + ['convnext', 'vit_b16'])}")


# Every arm of a grid writes into the same RESULT_DIR (one folder per frontend
# and corpus), so training_log.csv holds several runs' epochs end to end. This
# stamp goes on every row it writes, which is what makes them separable
# afterwards -- without it the file is one undifferentiated block of epochs.
# The four bits are part of the identity: 16 cells share one RESULT_DIR, and
# without them every cell's epochs would land in training_log.csv under the
# same name. Read the suffix as leaf<filters><pooling><compression><smoothing>.
RUN_TAG = "{}-{}-ts{}-p{}-e{}-s{}-leaf{}{}{}{}".format(
    FRONTEND, MODEL, TARGET_SIZE, PATCH, NUM_EPOCHS, SEED,
    LEARN_FILTERS, LEARN_POOLING, LEARN_COMPRESSION, LEARN_SMOOTHING)

leaf_trainable = [p for p in leaf.parameters() if p.requires_grad]
print(f"Frontend: {FRONTEND}")
print(f"Run tag: {RUN_TAG}")
print(f"Trainable params -- frontend: {sum(p.numel() for p in leaf_trainable)}, "
      f"clf: {sum(p.numel() for p in vit_model.parameters())}")
print(f"Methods: coord={COORD_CHANNELS} specaug={SPEC_AUGMENT} norm={NORMALIZE} "
      f"pcen={PCEN}")
print(f"LEAF learnable: filters={LEARN_FILTERS} pooling={LEARN_POOLING} "
      f"compression={LEARN_COMPRESSION} smoothing={LEARN_SMOOTHING}")

# One optimizer, one autograd graph -- a learnable frontend trains with the
# classifier. mel has no parameters at all, so it contributes no group: an
# empty one would leave LEAF_LR in the logs implying something was training.
_groups = [{"params": list(vit_model.parameters()), "lr": LR,
            "weight_decay": WEIGHT_DECAY}]
if leaf_trainable:
    _groups.append({"params": leaf_trainable, "lr": LEAF_LR, "weight_decay": 0.0})
optimizer = optim.AdamW(_groups)
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
last_path = os.path.join(RESULT_DIR, "last.pth")
train_log, best_uar = [], -1.0
start_epoch = 0

# last.pth is the resume point: unlike best.pth it holds the optimizer moments
# and the schedule position, which is what makes a split schedule equivalent to
# an unbroken one. best.pth stays the thing that gets tested.
if RESUME and os.path.exists(last_path):
    _ck = torch.load(last_path, map_location=device)
    if _ck.get("total_epochs") != NUM_EPOCHS:
        raise SystemExit(
            "REFUSED to resume: {} was written for NUM_EPOCHS={}, this run says {}. "
            "The cosine schedule would not line up -- rerun with the original "
            "total, or delete {} to start over.".format(
                last_path, _ck.get("total_epochs"), NUM_EPOCHS, last_path))
    # 16 cells share last.pth. The total_epochs check above cannot tell them
    # apart -- they all run the same schedule -- so without this a resumed cell
    # would continue the previous one's weights and report them as its own.
    if _ck.get("run_tag", RUN_TAG) != RUN_TAG:
        raise SystemExit(
            "REFUSED to resume: {} belongs to run {}, this run is {}. "
            "Delete it, or give this cell its own RESULTS_ROOT.".format(
                last_path, _ck.get("run_tag"), RUN_TAG))
    vit_model.load_state_dict(_ck["vit"])
    leaf.load_state_dict(_ck["leaf"])
    optimizer.load_state_dict(_ck["optimizer"])
    scheduler.load_state_dict(_ck["scheduler"])
    start_epoch = _ck["epoch"]
    best_uar = _ck["best_uar"]
    if _ck.get("numpy_rng") is not None:
        np.random.set_state(_ck["numpy_rng"])
    print("Resumed from {} at epoch {}/{} (best val UAR so far {:.2f}%)".format(
        last_path, start_epoch, NUM_EPOCHS, best_uar * 100))
elif RESUME:
    print("RESUME=1 but no {} yet -- starting from scratch.".format(last_path))

stop_epoch = NUM_EPOCHS
if EPOCHS_PER_RUN > 0:
    stop_epoch = min(start_epoch + EPOCHS_PER_RUN, NUM_EPOCHS)
if start_epoch >= NUM_EPOCHS:
    print("Schedule already complete ({} epochs).".format(NUM_EPOCHS))
elif stop_epoch < NUM_EPOCHS:
    print("This job runs epochs {}..{} of {}.".format(
        start_epoch + 1, stop_epoch, NUM_EPOCHS))

for epoch in range(start_epoch, stop_epoch):
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

    # Written every epoch, not just on improvement: this is the resume point,
    # so it has to reflect where training actually is, not where it was best.
    # A job killed by the scheduler mid-chunk then loses one epoch, not all.
    torch.save({"epoch": epoch + 1,
                "total_epochs": NUM_EPOCHS,
                "run_tag": RUN_TAG,
                "vit": vit_model.state_dict(),
                "leaf": leaf.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_uar": best_uar,
                "numpy_rng": np.random.get_state()}, last_path)

# Keep what earlier jobs recorded: with a split schedule each job holds only
# its own epochs, and a grid puts several models through this same file.
#
# Every row is stamped with the run it belongs to, so `model` and `run` are what
# separate one arm's curve from the next. Epoch numbers cannot do it -- they
# restart at 1 for every run in the file.
_log_path = os.path.join(RESULT_DIR, "training_log.csv")
_ident = {"run": RUN_TAG, "frontend": FRONTEND, "model": MODEL,
          "dataset": DATASET, "target_size": TARGET_SIZE, "patch": PATCH,
          "seed": SEED}
_log_df = pd.DataFrame([{**_ident, **_r} for _r in train_log])
# Concat rather than mode="a": a plain append writes no header, so rows written
# before this stamp existed would silently take the new columns' places.
# Rewriting is cheap -- the file holds one row per epoch.
if os.path.exists(_log_path):
    _log_df = pd.concat([pd.read_csv(_log_path), _log_df], ignore_index=True)
_log_df.to_csv(_log_path, index=False)

# A chunk that has not reached NUM_EPOCHS stops here. No test score and no
# sweep.csv row: a partially trained model is not a result, and writing one
# would put a number in the sweep that no completed run stands behind.
if stop_epoch < NUM_EPOCHS:
    print()
    print("=" * 66)
    print("Stopped at epoch {} of {}. Checkpoint: {}".format(
        stop_epoch, NUM_EPOCHS, last_path))
    print("Continue with the same settings plus RESUME=1.")
    print("No test/sweep.csv yet -- those come when the schedule finishes.")
    print("=" * 66)
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Test on the best-val checkpoint
# ---------------------------------------------------------------------------
vit_model.load_state_dict(torch.load(best_path, map_location=device)["vit"])
leaf.load_state_dict(torch.load(leaf_weights_path, map_location=device))

test_wa, test_uar, test_preds, test_targets = evaluate(test_idx)
train_wa, _, _, _ = evaluate(train_idx)

print("\n" + "=" * 66)
print(f"TEST -- {FRONTEND} + {MODEL} | {DATASET} | {SPLIT_MODE} split | "
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

# Tagged, unlike Run8's: 16 cells share this folder and a plain name would
# leave only the last one on disk.
cm_df.to_csv(os.path.join(RESULT_DIR, "confusion_matrix_{}.csv".format(RUN_TAG)))

run = {"frontend": FRONTEND, "model": MODEL, "dataset": DATASET,
       "classes": NUM_CLASSES, "split_mode": SPLIT_MODE, "epochs": NUM_EPOCHS,
       "batch": BATCH_SIZE, "lr": LR, "leaf_lr": LEAF_LR,
       "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
       "dropout": DROPOUT, "emb_dropout": EMB_DROPOUT, "dim": DIM, "depth": DEPTH,
       "heads": HEADS, "mlp_dim": MLP_DIM, "patch": PATCH, "dim_head": DIM_HEAD,
       "cnn_channels": CNN_CHANNELS, "target_size": TARGET_SIZE,
       "leaf_filters": LEAF_N_FILTERS, "pcen": PCEN,
       # The ablation table itself. learn_pooling keeps Run8's name and
       # meaning, so Run8 rows and Run10 rows line up in the same reader.
       "run_tag": RUN_TAG,
       "learn_filters": LEARN_FILTERS, "learn_pooling": LEARN_POOLING,
       "learn_compression": LEARN_COMPRESSION, "learn_smoothing": LEARN_SMOOTHING,
       "leaf_trainable_params": sum(p.numel() for p in leaf_trainable),
       "coord_channels": COORD_CHANNELS, "spec_augment": SPEC_AUGMENT,
       "freq_mask": FREQ_MASK, "time_mask": TIME_MASK, "normalize": NORMALIZE,
       "convnext_size": CONVNEXT_SIZE if MODEL == "convnext" else "",
       "drop_path": DROP_PATH if MODEL == "convnext" else "",
       "convnext_pretrained": CONVNEXT_PRETRAINED if MODEL == "convnext" else "",
       "vit_pretrained": VIT_PRETRAINED if MODEL == "vit_b16" else "",
       "train_frac": TRAIN_FRAC, "train_clips": len(train_idx),
       "fixed_seconds": FIXED_SECONDS, "seed": SEED,
       "best_val_uar": best_uar * 100, "test_wa": test_wa * 100,
       "test_uar": test_uar * 100, "train_wa": train_wa * 100}
sweep = os.path.join(RESULT_DIR, "sweep.csv")
# Merge on column names rather than appending blind: the column set grows over
# time (vit_pretrained and train_frac are newer than the rows already on disk),
# and a plain append would file the new values under the old header. Rewriting
# the whole file is free -- it holds one row per run.
row = pd.DataFrame([run])
if os.path.exists(sweep):
    row = pd.concat([pd.read_csv(sweep), row], ignore_index=True)
row.to_csv(sweep, index=False)

print(f"\nSaved to {RESULT_DIR}  (results appended to sweep.csv)")
