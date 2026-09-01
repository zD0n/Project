"""Run9 -- single-stream audio, scored by leave-one-session-out 5-fold CV.

Takes from

    Wen Wu, Chao Zhang, Philip C. Woodland,
    "Emotion recognition by fusing time synchronous and time asynchronous
     representations", ICASSP 2021.  arXiv:2010.14102

the two pieces that do not depend on text:

  * MULTI-HEAD SELF-ATTENTIVE POOLING, in place of reading the ViT's CLS
    token. The paper pools its frame sequence with a five-head attentive layer
    -- three heads on a spiky distribution, two on a smooth one -- so each head
    can weight a different part of the utterance. `POOL=attn` does that over
    the ViT's patch tokens; `POOL=cls` is the Run8 behaviour and the control it
    is measured against.

    One difference worth stating: the paper pools a 1-D sequence of time
    frames, while the ViT's tokens are a 2-D grid of (time, frequency) patches
    -- at TARGET_SIZE=64 and PATCH=8 that is 8x8 = 64 tokens, not 8 time steps.
    So the attention weights time and frequency jointly rather than time alone.
    That is a consequence of pooling a ViT instead of a TDNN, not a knob.

  * THE EVALUATION PROTOCOL. The paper reports leave-one-session-out 5-fold CV
    (8 speakers training, 2 testing per fold), not a single split. Run5..Run8
    all report one split, so their numbers carry no error bar; a Run9 row is a
    mean +- std over five folds.

Its large-margin softmax is available too, as `LOSS=amsoftmax`.

The paper's actual contribution -- fusing a time-synchronous audio+text branch
with a time-asynchronous cross-utterance one -- is NOT reproduced here. Both of
its branches are text-driven and this is an audio-only pipeline, so what is
left is one stream:

    waveform -> LEAF/mel -> (+coord planes) -> ViT -> pool -> FC -> softmax

Everything before the pooling is Run8's, unchanged, so `POOL=cls LOSS=ce` is a
Run8 model measured on the new protocol and the pooling comparison is
controlled.

CREMA-D has no sessions -- 91 independent actors -- so its 5 folds are
actor-disjoint groups instead. Different corpus structure, same guarantee: no
speaker appears in both train and test.

Class sets (IEMOCAP_CLASSES):
    4        the paper's 4-way: neutral, sad, anger, happy (excited merged in)
    5        Run8's 5-way: excited kept separate, for comparing against Run8
    5others  the paper's 5-way: 4-way plus an "others" class (frustration,
             surprise, fear, disgust, other)

Absolute numbers will not land on the paper's 77.57/78.41 -- that system had
text and this one does not, and they report 5531 utterances for 4-way against
6896 here, so the agreement filter differs too. Read Run9's arms against each
other on the same folds, which is controlled.

Environment variables: everything Run8 accepts, plus

    POOL             attn  attn | cls   temporal pooling of the ViT tokens
    ATT_HEADS        5     self-attentive heads (paper: 5)
    ATT_SHARP        3     of those, how many start on the spiky temperature
    ATT_HIDDEN       64    bottleneck width inside the attention
    EMBED_DIM        256   width of the pooled embedding before the classifier
    IEMOCAP_CLASSES  4     4 | 5 | 5others
    LOSS             ce    ce | amsoftmax  (the paper's large-margin family)
    AM_MARGIN        0.2
    AM_SCALE         30
    CV_FOLDS         5
    VAL_FRAC         0.125 share of TRAINING speakers held out for val
    FOLD             -1    -1 runs every fold; 0..4 runs one, for job splitting
    RESUME_FOLD      1     0 restarts an interrupted fold instead of resuming

MODEL accepts the four hand-rolled ViTs and `convnext`. For ConvNeXt, POOL=cls
is its own global average pool (Run8's model) and POOL=attn pools the final
(N,C,H,W) stage map instead -- that map IS the token grid, just spelled
differently. Its ConvNeXt knobs are Run8's, unchanged:

    CONVNEXT_SIZE        tiny   tiny small base large xlarge
    DROP_PATH            0.1    stochastic depth
    HEAD_INIT_SCALE      1.0
    CONVNEXT_PRETRAINED  0      1 downloads ImageNet weights
    CONVNEXT_22K         0      with the above, 22k instead of 1k

Mind the grid size with POOL=attn: ConvNeXt downsamples by 32, so TARGET_SIZE=64
leaves a 2x2 map -- four tokens for five heads, too few for the attention to say
anything the average does not. Use TARGET_SIZE=128 (4x4) or 224 (7x7). Run9
prints a warning when the count is low rather than letting it pass silently.

Folds are the unit of restart. Each finished fold appends a row to folds.csv;
rerunning skips folds already recorded, so a job killed after fold 2 resumes at
fold 3. The aggregate sweep.csv row is written only once all folds are in.

Within a fold, an epoch is the unit of restart. Every epoch overwrites
last_<tag>_fold<k>.pth with the model, frontend, optimizer, scheduler and RNG
state, and a fold that is killed part-way picks up from the next epoch on the
following run rather than starting over -- which matters when the job's wall
clock is shorter than a fold. The file is deleted once the fold reaches
folds.csv. RESUME_FOLD=0 disables it.
"""

import hashlib
import json
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
LABEL_ALIASES = {
    "sadness": "sad",
    "happiness": "happy",
    "excitement": "excited",
    "angry": "anger",
    "neutral state": "neutral",
}

DATASET_DIR = os.environ.get("DATASET_DIR", "Dataset2")
SAMPLE_RATE = 16000

FRONTEND = os.environ.get("FRONTEND", "leaf").lower()
if FRONTEND not in ("leaf", "mel"):
    raise SystemExit(f"FRONTEND={FRONTEND!r} unknown; choose 'leaf' or 'mel'")
# Separate from Run8's results/{Leaf,Mel}Torch_ViT: these rows are 5-fold CV
# means, not single-split scores, and mixing the two in one sweep.csv would put
# incomparable numbers in the same column.
# RESULTS_ROOT moves the whole tree without disturbing the frontend and
# dataset folders underneath, so a grid run under one banner keeps its rows
# out of the main results/ -- e.g. RESULTS_ROOT="results/Ablation Study"
# writes "results/Ablation Study/{Leaf,Mel}Torch_CV/<dataset>/". Same
# variable and same meaning as in Run8.
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "./results")
RESULT_DIR = os.path.join(
    RESULTS_ROOT, "{}Torch_CV".format("Leaf" if FRONTEND == "leaf" else "Mel"))


def _env(name, default, cast=float):
    return cast(os.environ.get(name, str(default)))


LEAF_N_FILTERS = _env("LEAF_N_FILTERS", 64, int)
TARGET_SIZE = _env("TARGET_SIZE", 64, int)
WINDOW_LEN = _env("WINDOW_LEN", 25, float)
LEAF_LR = _env("LEAF_LR", 1e-5)
LEARN_POOLING = _env("LEARN_POOLING", 1, int)   # 1 trains the Gaussian lowpass; 0 freezes it
PCEN = _env("PCEN", 0, int)                     # 0 = log compression, as Run8

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

MODEL = os.environ.get("MODEL", "cnn_vit").lower()

# ConvNeXt only (MODEL=convnext); ignored by the ViT variants. Same names and
# defaults as Run8, so a config carries over between the two scripts.
CONVNEXT_SIZE = os.environ.get("CONVNEXT_SIZE", "tiny").lower()
DROP_PATH = _env("DROP_PATH", 0.1)          # stochastic depth, ConvNeXt-T default
HEAD_INIT_SCALE = _env("HEAD_INIT_SCALE", 1.0)
CONVNEXT_PRETRAINED = _env("CONVNEXT_PRETRAINED", 0, int)  # downloads ImageNet weights
CONVNEXT_22K = _env("CONVNEXT_22K", 0, int)                # 22k instead of 1k weights

# --- what Run9 adds --------------------------------------------------------
POOL = os.environ.get("POOL", "attn").lower()
ATT_HEADS = _env("ATT_HEADS", 5, int)
ATT_SHARP = _env("ATT_SHARP", 3, int)
ATT_HIDDEN = _env("ATT_HIDDEN", 64, int)
EMBED_DIM = _env("EMBED_DIM", 256, int)
IEMOCAP_CLASSES = os.environ.get("IEMOCAP_CLASSES", "4").lower()
LOSS = os.environ.get("LOSS", "ce").lower()
AM_MARGIN = _env("AM_MARGIN", 0.2)
AM_SCALE = _env("AM_SCALE", 30.0)
CV_FOLDS = _env("CV_FOLDS", 5, int)
VAL_FRAC = _env("VAL_FRAC", 0.125)   # of the TRAINING speakers, per fold
FOLD = _env("FOLD", -1, int)
# 1 = pick a killed fold back up where it stopped; 0 = always start it over.
RESUME_FOLD = _env("RESUME_FOLD", 1, int)

SEED = _env("SEED", 99, int)
DATASET = os.environ.get("DATASET", "auto").lower()
VIT_CHANNELS = 1 + (2 if COORD_CHANNELS else 0)

if POOL not in ("attn", "cls"):
    raise SystemExit(f"POOL={POOL!r}; choose 'attn' or 'cls'")
if LOSS not in ("ce", "amsoftmax"):
    raise SystemExit(f"LOSS={LOSS!r}; choose 'ce' or 'amsoftmax'")


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
# Labels
# ---------------------------------------------------------------------------
labels_df = pd.read_csv(os.path.join(DATASET_DIR, "labels.csv"))
name_col = "Filename" if "Filename" in labels_df.columns else "filename"
labels_dict = dict(zip(labels_df[name_col], labels_df["Label"]))


def canon(label):
    s = str(label).strip().lower()
    return LABEL_ALIASES.get(s, s)


vocab = {canon(v) for v in labels_dict.values()}
if DATASET == "auto":
    DATASET = "cremad" if "disgust" in vocab and "excited" not in vocab else "iemocap"

# "4" is the paper's; "5" keeps excited apart so a Run9 row lines up with a
# Run8 row; "5others" is the paper's 5-way, where everything outside the four
# is pooled into one class rather than dropped.
IEMOCAP_SETS = {
    "4": ({"neutral": 0, "sad": 1, "anger": 2, "happy": 3, "excited": 3}, None),
    "5": ({"neutral": 0, "sad": 1, "anger": 2, "happy": 3, "excited": 4}, None),
    "5others": ({"neutral": 0, "sad": 1, "anger": 2, "happy": 3, "excited": 3}, 4),
}

if DATASET == "iemocap":
    if IEMOCAP_CLASSES not in IEMOCAP_SETS:
        raise SystemExit("IEMOCAP_CLASSES={!r}; choose 4, 5 or 5others".format(
            IEMOCAP_CLASSES))
    EMOTION_MAP, OTHERS_CLASS = IEMOCAP_SETS[IEMOCAP_CLASSES]
else:
    EMOTION_MAP, OTHERS_CLASS = CREMAD_MAP, None

NUM_CLASSES = max(EMOTION_MAP.values()) + 1
if OTHERS_CLASS is not None:
    NUM_CLASSES = max(NUM_CLASSES, OTHERS_CLASS + 1)
RESULT_DIR = os.path.join(RESULT_DIR, DATASET)
os.makedirs(RESULT_DIR, exist_ok=True)

# Several labels can share an index (excited folds into happy at 4-way), so
# build the display names by joining rather than inverting the map.
CLASS_NAMES = []
for c in range(NUM_CLASSES):
    names = sorted(k for k, v in EMOTION_MAP.items() if v == c)
    CLASS_NAMES.append("+".join(names) if names else "others")


def speaker_of(fname):
    """CREMA-D 1001_... -> '1001'; IEMOCAP Ses01F_impro01_F000 -> 'Ses01F'.

    IEMOCAP's speaker is the session plus the gender of the *utterance*, not
    the gender in the dialogue name: Ses01F_impro01_M000 is the male actor of
    session 1. Ten speakers over five sessions.
    """
    stem = os.path.splitext(fname)[0]
    if DATASET == "iemocap":
        parts = stem.split("_")
        return stem[:5] + (parts[-1][0] if parts[-1] else "")
    return stem.split("_")[0]


def session_of(fname):
    """IEMOCAP session id, '01'..'05' -- the CV fold key in the paper."""
    return os.path.splitext(fname)[0][3:5]


audio_dir = next(
    (os.path.join(DATASET_DIR, d) for d in ("audios", "AudioWAV")
     if os.path.isdir(os.path.join(DATASET_DIR, d))),
    DATASET_DIR,
)
all_wavs = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".wav"))

audio_files, all_labels, skipped = [], [], {}
for f in all_wavs:
    lab = labels_dict.get(os.path.splitext(f)[0])
    if lab is None:
        skipped["<no label row>"] = skipped.get("<no label row>", 0) + 1
        continue
    c = canon(lab)
    if c in EMOTION_MAP:
        audio_files.append(f)
        all_labels.append(EMOTION_MAP[c])
    elif OTHERS_CLASS is not None:
        audio_files.append(f)
        all_labels.append(OTHERS_CLASS)
    else:
        skipped[c] = skipped.get(c, 0) + 1

all_labels = np.array(all_labels, dtype=np.int64)
print(f"Dataset: {DATASET} | {NUM_CLASSES} classes: {', '.join(CLASS_NAMES)}")
print(f"Usable clips: {len(audio_files)} of {len(all_wavs)}")
if skipped:
    print("  skipped labels: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
if not audio_files:
    raise SystemExit("No clips matched the class set -- check DATASET / labels.csv")
print("  per class: " + ", ".join(
    f"{n}={c}" for n, c in zip(CLASS_NAMES,
                               np.bincount(all_labels, minlength=NUM_CLASSES))))

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
print(f"Loading {len(audio_files)} audio files...")
all_audio = []
for fname in audio_files:
    audio = sf.read(os.path.join(audio_dir, fname), dtype="float32")
    if isinstance(audio, tuple):
        audio = audio[0]
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    all_audio.append(audio)

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

# ---------------------------------------------------------------------------
# Batching and features -- identical to Run8, so POOL=cls reproduces its model
# ---------------------------------------------------------------------------
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


def features(indices, train):
    """(B, VIT_CHANNELS, TARGET, TARGET) on `device`.

    leaf_pytorch emits (B, n_filters, frames) i.e. freq-major; the transpose
    keeps time on rows, matching Run5..Run8.
    """
    wav = make_batch(indices, train=train)
    x = torch.from_numpy(wav).to(device, non_blocking=True).unsqueeze(1)
    feats = frontend(x)                                # (B, n_filters, frames)
    if FRONTEND == "mel" or not PCEN:
        feats = torch.log(feats + 1e-5)
    feats = feats.transpose(1, 2).unsqueeze(1)         # (B, 1, frames, filters)
    feats = F.interpolate(feats, size=(TARGET_SIZE, TARGET_SIZE),
                          mode="bilinear", align_corners=False)
    if NORMALIZE:
        mean = feats.mean(dim=(-2, -1), keepdim=True)
        std = feats.std(dim=(-2, -1), keepdim=True)
        feats = (feats - mean) / (std + 1e-5)
    if train and SPEC_AUGMENT:
        feats = spec_augment(feats)
    if COORD_CHANNELS:
        feats = torch.cat(
            [feats, coord_planes(feats.shape[0], TARGET_SIZE, TARGET_SIZE)], dim=1)
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

if MODEL not in _MODELS and MODEL != "convnext":
    raise SystemExit(
        "MODEL={!r} is not available in Run9; choose one of {}.\n"
        "vit_b16 stays in Run8.".format(
            MODEL, sorted(list(_MODELS) + ["convnext"])))


class SelfAttentivePooling(nn.Module):
    """Multi-head self-attentive pooling over a token sequence (paper: 5 heads).

    Each head learns its own weighting of the sequence and the results are
    concatenated, so one head can key on a loud onset while another spreads
    over the whole utterance. The paper sets three heads to a spiky
    distribution and two to a smoother one; a per-head temperature on the
    logits does that, and making it learnable leaves the split as an
    initialisation rather than a constraint.
    """

    def __init__(self, dim, heads, hidden, sharp):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, heads)
        t = torch.full((heads,), 2.0)                # smooth
        t[:max(0, min(sharp, heads))] = 0.5          # spiky heads first
        self.log_temp = nn.Parameter(t.log())
        self.out_dim = dim * heads

    def forward(self, h):                            # h: (B, tokens, dim)
        e = self.w2(torch.tanh(self.w1(h)))          # (B, tokens, heads)
        e = e / self.log_temp.exp().clamp(min=1e-2)
        a = torch.softmax(e, dim=1)
        return torch.einsum("bth,btd->bhd", a, h).flatten(1)


# --- backbones -------------------------------------------------------------
# Both expose the same four things, so Run9Net does not care which it holds:
#   .dim              width of one token / of the pooled vector
#   .tokens           how many tokens forward_tokens returns
#   forward_pooled()  the backbone's OWN pooling -- what POOL=cls uses, and
#                     what keeps POOL=cls identical to the Run8 model
#   forward_tokens()  the pre-pooling sequence, for POOL=attn

class ViTBackbone(nn.Module):
    """One of the four hand-rolled ViTs in Model/, built exactly as Run8 builds
    it -- no edits to those files.

    A forward hook on `.transformer` catches the token sequence on the way
    past; all four variants have that attribute, so this needs no per-variant
    special-casing and stays in step if those files change. The trunk's own
    mlp_head becomes an Identity, since the classifier here is our own.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.net.mlp_head = nn.Identity()
        self.dim = DIM
        self.tokens = (TARGET_SIZE // PATCH) ** 2
        self._tok = None
        net.transformer.register_forward_hook(
            lambda _m, _i, out: setattr(self, "_tok", out))

    def forward_pooled(self, x):
        return self.net(x)                   # CLS token; head is Identity

    def forward_tokens(self, x):
        self.net(x)
        # Drop the CLS token and keep the patch grid -- that token is the very
        # summary this branch exists to replace.
        return self._tok[:, 1:]


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt from the vendored ./ConvNeXt checkout.

    ConvNeXt has no token sequence in the ViT sense, but it has the same thing
    under another name: stage 4 emits an (N, C, H, W) map that forward_features
    immediately global-average-pools away. Flattening that map gives H*W tokens
    of width C -- the spatial grid the attention pools over, exactly as the
    ViT's patches are a spatial grid.

    So POOL=cls is ConvNeXt's own GAP head (Run8's model) and POOL=attn
    replaces that average with the paper's weighted one. The stage loop is
    spelled out rather than hooked so both paths run identical trunk compute.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.dim = _CONVNEXT_SIZES[CONVNEXT_SIZE]["dims"][-1]
        # Stem is stride-4 and each later stage halves again: 32x total.
        self.tokens = max(1, TARGET_SIZE // 32) ** 2

    def _grid(self, x):
        for i in range(4):
            x = self.net.downsample_layers[i](x)
            x = self.net.stages[i](x)
        return x                             # (B, C, H, W)

    def forward_pooled(self, x):
        # Identical to the vendored forward_features: norm(GAP(map)).
        return self.net.norm(self._grid(x).mean([-2, -1]))

    def forward_tokens(self, x):
        t = self._grid(x).flatten(2).transpose(1, 2)   # (B, H*W, C)
        # Norm per token, then pool -- the reverse of forward_pooled, which
        # pools then norms. Deliberate: the attention scores tokens against
        # each other, so they have to be on a common scale first, and it
        # matches the ViT arm, where the transformer's final LayerNorm has
        # already normed every token before the pooling sees them. It does mean
        # a uniform attention here is not numerically identical to GAP.
        return self.net.norm(t)


class Run9Net(nn.Module):
    """Backbone + the chosen pooling + a fresh classifier head."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone

        if POOL == "attn":
            self.pool = SelfAttentivePooling(backbone.dim, ATT_HEADS,
                                             ATT_HIDDEN, ATT_SHARP)
            emb_in = self.pool.out_dim
        else:
            self.pool = None
            emb_in = backbone.dim

        self.embed = nn.Sequential(nn.Linear(emb_in, EMBED_DIM), nn.ReLU(),
                                   nn.Dropout(DROPOUT))
        if LOSS == "amsoftmax":
            # AM-Softmax needs unit-norm weights and inputs; the scale and
            # margin are applied in the loss, not here.
            self.weight = nn.Parameter(torch.randn(num_classes, EMBED_DIM) * 0.01)
        else:
            self.head = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, img):
        if self.pool is not None:
            z = self.pool(self.backbone.forward_tokens(img))
        else:
            z = self.backbone.forward_pooled(img)
        z = self.embed(z)
        if LOSS == "amsoftmax":
            return F.linear(F.normalize(z), F.normalize(self.weight))
        return self.head(z)


def build_convnext():
    """ConvNeXt on LEAF features, built the same way Run8 builds it.

    in_chans follows VIT_CHANNELS, so COORD_CHANNELS works here too: the
    coordinate planes just become extra input channels on the stem.
    """
    from ConvNeXt.models.convnext import ConvNeXt, model_urls

    if CONVNEXT_SIZE not in _CONVNEXT_SIZES:
        raise SystemExit("CONVNEXT_SIZE={!r} unknown; choose one of {}".format(
            CONVNEXT_SIZE, sorted(_CONVNEXT_SIZES)))
    if TARGET_SIZE < 32:
        raise SystemExit("TARGET_SIZE={} is too small for ConvNeXt "
                         "(4 stages downsample by 32x; use >= 32)".format(TARGET_SIZE))

    # num_classes only sizes a head we replace; keep it valid anyway.
    model = ConvNeXt(in_chans=VIT_CHANNELS, num_classes=NUM_CLASSES,
                     drop_path_rate=DROP_PATH, head_init_scale=HEAD_INIT_SCALE,
                     **_CONVNEXT_SIZES[CONVNEXT_SIZE])

    if CONVNEXT_PRETRAINED:
        key = "convnext_{}_{}".format(CONVNEXT_SIZE, "22k" if CONVNEXT_22K else "1k")
        if key not in model_urls:
            raise SystemExit("No published weights for {} "
                             "(xlarge is 22k-only; set CONVNEXT_22K=1)".format(key))
        state = torch.hub.load_state_dict_from_url(
            model_urls[key], map_location="cpu")["model"]
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
        print("Loaded {}: {} missing, {} unexpected "
              "(the head is expected to be missing)".format(
                  key, len(missing), len(unexpected)))
    return model


def build_model():
    if MODEL == "convnext":
        backbone = ConvNeXtBackbone(build_convnext())
    else:
        kw = dict(image_size=(TARGET_SIZE, TARGET_SIZE), patch_size=(PATCH, PATCH),
                  num_classes=DIM,  # replaced by Identity; only the shape matters
                  dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM,
                  channels=VIT_CHANNELS, pool="cls", dropout=DROPOUT,
                  emb_dropout=EMB_DROPOUT)
        if MODEL.startswith("cnn_"):
            kw["cnn_channels"] = CNN_CHANNELS
        if MODEL != "cnn_vit_local":
            kw["dim_head"] = DIM_HEAD
        backbone = ViTBackbone(_MODELS[MODEL].ViT(**kw))
    return Run9Net(backbone, NUM_CLASSES).to(device)


def build_frontend():
    if FRONTEND == "mel":
        import torchaudio
        # LEAF convolves over a 25 ms window and pools to a 10 ms hop; match
        # both so the two frontends emit the same number of frames.
        win = int(round(SAMPLE_RATE * WINDOW_LEN / 1000.0))
        n_fft = 1 << (win - 1).bit_length()          # next power of two >= win

        class MelFrontend(nn.Module):
            """Fixed log-mel filterbank with the same contract as Leaf:
            (B, 1, T) -> (B, n_filters, frames), strictly positive.

            Deliberately not learnable -- that is the point of the comparison.
            """

            def __init__(self):
                super().__init__()
                self.mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=SAMPLE_RATE, n_fft=n_fft, win_length=win,
                    hop_length=int(round(SAMPLE_RATE * 10.0 / 1000.0)),
                    n_mels=LEAF_N_FILTERS, f_min=60.0, f_max=7800.0, power=2.0)

            def forward(self, x):
                # 1e-6 keeps the log downstream finite on digital silence.
                return self.mel(x.squeeze(1)) + 1e-6

        return MelFrontend().to(device)

    fe = Leaf(n_filters=LEAF_N_FILTERS, sample_rate=SAMPLE_RATE,
              window_len=WINDOW_LEN, preemp=False, init_min_freq=60.0,
              init_max_freq=7800.0, pcen_compression=bool(PCEN)).to(device)
    if not LEARN_POOLING:
        for p in fe._pooling.parameters():
            p.requires_grad_(False)
    return fe


def am_softmax_loss(cos, target):
    """Additive-margin softmax -- the large-margin family the paper uses."""
    m = torch.zeros_like(cos).scatter_(1, target.view(-1, 1), AM_MARGIN)
    return F.cross_entropy(AM_SCALE * (cos - m), target,
                           label_smoothing=LABEL_SMOOTHING)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
speakers = np.array([speaker_of(f) for f in audio_files])


def make_folds():
    """CV_FOLDS test groups, speaker-disjoint, plus val speakers per fold.

    IEMOCAP: one session (2 speakers) per fold, which is the paper's
    leave-one-session-out. CREMA-D has no sessions -- 91 independent actors --
    so its actors are shuffled into CV_FOLDS groups instead. Different corpus
    structure, same guarantee: no speaker is in both train and test.
    """
    rng = np.random.RandomState(SEED)
    if DATASET == "iemocap":
        sess = np.array([session_of(f) for f in audio_files])
        names = sorted(set(sess))
        groups = [np.where(sess == s)[0] for s in names]
        if len(groups) != CV_FOLDS:
            print(f"  note: {len(groups)} sessions found, CV_FOLDS={CV_FOLDS}")
    else:
        uniq = np.unique(speakers)
        rng.shuffle(uniq)
        parts = np.array_split(uniq, CV_FOLDS)
        groups = [np.where(np.isin(speakers, p))[0] for p in parts]
        names = [f"g{k}" for k in range(len(parts))]

    folds = []
    for k, test_idx in enumerate(groups):
        rest = np.setdiff1d(np.arange(len(audio_files)), test_idx)
        # Held-out speakers for model selection, taken from the training
        # speakers and never from the test group, so the reported test score is
        # still leave-one-session-out.
        #
        # A fraction rather than a fixed count: IEMOCAP has 8 training speakers
        # so this is 1, but CREMA-D has ~73, and one actor there is ~82 clips
        # over 6 classes -- far too few to pick a checkpoint on.
        rest_spk = np.unique(speakers[rest])
        n_val = min(len(rest_spk) - 1, max(1, int(round(len(rest_spk) * VAL_FRAC))))
        val_spk = np.random.RandomState(SEED + k).permutation(rest_spk)[:n_val]
        in_val = np.isin(speakers[rest], val_spk)
        folds.append(dict(name=names[k], train=rest[~in_val], val=rest[in_val],
                          test=test_idx, val_speakers=len(val_spk),
                          val_speaker=",".join(map(str, sorted(val_spk)))))
    return folds


folds = make_folds()
print("\nCV: {} folds ({})".format(
    len(folds), "leave-one-session-out" if DATASET == "iemocap"
    else "actor-disjoint groups"))
for k, f in enumerate(folds):
    ov = set(speakers[f["train"]]) & set(speakers[f["test"]])
    print("  fold {} [{}]: train {} | val {} ({} spk) | test {} | overlap {}".format(
        k, f["name"], len(f["train"]), len(f["val"]), f["val_speakers"],
        len(f["test"]), len(ov)))

# ---------------------------------------------------------------------------
# Train / evaluate one fold
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, indices):
    """WA (accuracy) and UAR (mean per-class recall)."""
    frontend.eval()
    model.eval()
    preds = []
    for i in range(0, len(indices), BATCH_SIZE):
        idx = indices[i:i + BATCH_SIZE]
        preds.append(model(features(idx, train=False)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    tgts = all_labels[indices]
    wa = float((preds == tgts).mean())
    recalls = [float((preds[tgts == c] == c).mean())
               for c in range(NUM_CLASSES) if (tgts == c).any()]
    return wa, float(np.mean(recalls)), preds, tgts


def run_fold(k, fold):
    global frontend
    # Reseeded per fold, so every fold starts from a comparable initialisation
    # and the fold index keeps them from being five identical runs.
    set_seed(SEED + k)
    frontend = build_frontend()
    model = build_model()

    fe_trainable = [p for p in frontend.parameters() if p.requires_grad]
    # One optimizer, one autograd graph -- a learnable frontend trains with the
    # classifier. mel has no parameters at all, so it contributes no group.
    groups = [{"params": list(model.parameters()), "lr": LR,
               "weight_decay": WEIGHT_DECAY}]
    if fe_trainable:
        groups.append({"params": fe_trainable, "lr": LEAF_LR, "weight_decay": 0.0})
    optimizer = optim.AdamW(groups)

    def lr_scale(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / max(1, WARMUP_EPOCHS)
        p = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    train_idx, val_idx, test_idx = fold["train"], fold["val"], fold["test"]
    n_train = len(train_idx)
    num_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    best_path = os.path.join(RESULT_DIR, f"best_{TAG_ID}_fold{k}.pth")
    # Mid-fold resume. A capped job is killed without warning, and a fold cut
    # off at epoch 25 of 30 used to leave nothing behind -- the next submission
    # started it again from epoch 1. This file is rewritten every epoch and
    # removed once the fold's row reaches folds.csv, so at most one epoch is
    # ever lost.
    #
    # PATCH is in the name because TAG does not carry it, and two patch sizes
    # give different position-embedding shapes -- without it a 224/patch-8 run
    # would try to resume from a 224/patch-16 checkpoint.
    last_path = os.path.join(RESULT_DIR, f"last_{TAG_ID}p{PATCH}_fold{k}.pth")
    best_uar, log, start_epoch = -1.0, [], 0

    if RESUME_FOLD and os.path.exists(last_path):
        try:
            # weights_only=False: this checkpoint deliberately holds more
            # than tensors -- optimizer/scheduler state and the NumPy RNG
            # tuple -- and torch >= 2.6 refuses those under the default
            # weights_only=True. The file is one this script wrote itself.
            _ck = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(_ck["model"])
            frontend.load_state_dict(_ck["frontend"])
            optimizer.load_state_dict(_ck["optimizer"])
            scheduler.load_state_dict(_ck["scheduler"])
            best_uar, log, start_epoch = _ck["best_uar"], _ck["log"], _ck["epoch"]
            # Restored after set_seed() above, deliberately: the point is to
            # continue the interrupted run's stream, not to restart it.
            np.random.set_state(_ck["np_rng"])
            torch.set_rng_state(_ck["torch_rng"])
            if _ck.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(_ck["cuda_rng"])
            print("  resuming fold {} at epoch {}/{} (best val UAR {:.2f}%)".format(
                k, start_epoch + 1, NUM_EPOCHS, best_uar * 100))
        except Exception as _e:
            # A checkpoint from an incompatible configuration is not worth
            # crashing a multi-hour job over. Say so and train the fold fresh.
            print("  ignoring {}: {}: {}".format(
                os.path.basename(last_path), type(_e).__name__, _e))
            best_uar, log, start_epoch = -1.0, [], 0

    if k == 0:
        print("\nFrontend: {} | Model: {} | POOL={} LOSS={}".format(
            FRONTEND, MODEL, POOL, LOSS))
        print("  Trainable params -- frontend: {}, classifier: {}".format(
            sum(p.numel() for p in fe_trainable),
            sum(p.numel() for p in model.parameters())))
        if MODEL == "convnext":
            print("  ConvNeXt-{} (drop_path={}, pretrained={})".format(
                CONVNEXT_SIZE, DROP_PATH, CONVNEXT_PRETRAINED))
        if POOL == "attn":
            _bb = model.backbone
            print("  Attentive pooling: {} heads ({} spiky) over {} tokens "
                  "x {} dims -> {}".format(
                      ATT_HEADS, max(0, min(ATT_SHARP, ATT_HEADS)),
                      _bb.tokens, _bb.dim, model.pool.out_dim))
            # Attentive pooling has to have something to choose between. A 2x2
            # grid is ConvNeXt at TARGET_SIZE=64: four tokens for five heads,
            # which is barely distinguishable from the average it replaces.
            if _bb.tokens < 9:
                print("  WARNING: only {} tokens to pool over. The attentive "
                      "arm cannot".format(_bb.tokens))
                print("           differ much from POOL=cls here -- raise "
                      "TARGET_SIZE ({} gives".format(TARGET_SIZE * 2))
                print("           {} tokens) or lower PATCH before reading "
                      "this comparison.".format(
                          (TARGET_SIZE * 2 // 32) ** 2 if MODEL == "convnext"
                          else (TARGET_SIZE * 2 // PATCH) ** 2))
        print("  Methods: coord={} specaug={} norm={} pcen={}".format(
            COORD_CHANNELS, SPEC_AUGMENT, NORMALIZE, PCEN))

    print(f"\n{'=' * 66}\nFold {k} [{fold['name']}] -- {NUM_EPOCHS} epochs, "
          f"{n_train} train / {len(val_idx)} val / {len(test_idx)} test\n{'=' * 66}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        frontend.train()
        model.train()
        perm = train_idx[np.random.permutation(n_train)]
        epoch_loss, correct = 0.0, 0

        for b in range(num_batches):
            idx = perm[b * BATCH_SIZE:min((b + 1) * BATCH_SIZE, n_train)]
            y = torch.from_numpy(all_labels[idx]).to(device)
            out = model(features(idx, train=True))
            loss = am_softmax_loss(out, y) if LOSS == "amsoftmax" else ce(out, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss) * len(idx)
            correct += int((out.argmax(1) == y).sum())

        scheduler.step()
        val_wa, val_uar, _, _ = evaluate(model, val_idx)
        log.append(dict(fold=k, epoch=epoch + 1, loss=epoch_loss / n_train,
                        train_wa=correct / n_train * 100, val_wa=val_wa * 100,
                        val_uar=val_uar * 100))
        star = ""
        if val_uar > best_uar:
            best_uar = val_uar
            torch.save({"model": model.state_dict(),
                        "frontend": frontend.state_dict()}, best_path)
            star = "  *"
        print("  epoch {:3d}/{}  loss {:.4f}  train {:.2f}%  val WA {:.2f}%  "
              "UAR {:.2f}%  {:.0f}s{}".format(
                  epoch + 1, NUM_EPOCHS, epoch_loss / n_train,
                  correct / n_train * 100, val_wa * 100, val_uar * 100,
                  time.time() - t0, star))

        # Via a temporary file: a kill during the write must not leave a
        # truncated checkpoint where the last good one was.
        if RESUME_FOLD:
            _tmp = last_path + ".tmp"
            torch.save({"epoch": epoch + 1,
                        "model": model.state_dict(),
                        "frontend": frontend.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_uar": best_uar, "log": log,
                        "np_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": (torch.cuda.get_rng_state_all()
                                     if torch.cuda.is_available() else None)},
                       _tmp)
            os.replace(_tmp, last_path)

    # Test on the best-val checkpoint, never the last one.
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model"])
    frontend.load_state_dict(ck["frontend"])
    test_wa, test_uar, preds, tgts = evaluate(model, test_idx)
    train_wa, _, _, _ = evaluate(model, train_idx)
    print("  fold {} TEST: WA {:.2f}%  UAR {:.2f}%  (train WA {:.2f}%)".format(
        k, test_wa * 100, test_uar * 100, train_wa * 100))

    # The caller is about to write this fold's row, after which the resume
    # checkpoint is dead weight -- and a stale one would have a finished fold
    # trying to resume itself.
    for _f in (last_path, last_path + ".tmp"):
        if os.path.exists(_f):
            os.remove(_f)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(tgts, preds):
        cm[t][p] += 1
    return dict(fold=k, name=fold["name"], n_test=len(test_idx),
                val_speaker=fold["val_speaker"], best_val_uar=best_uar * 100,
                test_wa=test_wa * 100, test_uar=test_uar * 100,
                train_wa=train_wa * 100), cm, log


# ---------------------------------------------------------------------------
# Drive the CV
# ---------------------------------------------------------------------------
frontend = None
folds_path = os.path.join(RESULT_DIR, "folds.csv")
cm_path = os.path.join(RESULT_DIR, "confusion_{}_fold{}.csv")

# Identifies "the same experiment" for the fold-skip logic below, so it has to
# carry everything that changes the result -- epochs and input size included.
# Without them a NUM_EPOCHS=2 smoke test would leave 5 rows behind and the real
# 30-epoch run would skip every fold as already done.
TAG = "{}|{}{}|{}|c{}|pool{}|h{}|{}|e{}|t{}|s{}".format(
    FRONTEND, MODEL,
    # ConvNeXt-tiny and ConvNeXt-base are different experiments; without the
    # size in the tag the second would skip the first's folds as already done.
    ":" + CONVNEXT_SIZE if MODEL == "convnext" else "",
    IEMOCAP_CLASSES if DATASET == "iemocap" else "6",
    NUM_CLASSES, POOL, ATT_HEADS if POOL == "attn" else 0, LOSS,
    NUM_EPOCHS, TARGET_SIZE, SEED)
# Short, filesystem-safe stamp so one arm's checkpoints cannot be mistaken for
# another's -- every arm shares RESULT_DIR.
TAG_ID = hashlib.md5(TAG.encode()).hexdigest()[:8]

done = set()
if os.path.exists(folds_path):
    _prev = pd.read_csv(folds_path, dtype={"name": str})
    if "tag" in _prev.columns:
        done = set(_prev[_prev["tag"] == TAG]["fold"].astype(int))
    if done:
        print("\nAlready in folds.csv for this config ({}): folds {}".format(
            TAG, sorted(done)))

logs = []
for k in (range(len(folds)) if FOLD < 0 else [FOLD]):
    if k in done:
        print(f"\nFold {k}: already recorded, skipping (delete its row to redo)")
        continue
    row, cm, log = run_fold(k, folds[k])
    row["tag"] = TAG
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        cm_path.format(TAG_ID, k))
    logs.extend(log)
    # Written per fold, not at the end: a job that dies in fold 3 keeps 0..2.
    pd.DataFrame([row]).to_csv(folds_path, mode="a", index=False,
                               header=not os.path.exists(folds_path))

if logs:
    # Every arm writes into the same RESULT_DIR, so this file holds the epochs
    # of several runs end to end. Stamp each row with the arm it belongs to:
    # `fold` alone cannot separate them, and epoch restarts at 1 per fold.
    # TAG is the same string folds.csv is keyed by, so the two join on it.
    lp = os.path.join(RESULT_DIR, "training_log.csv")
    _ident = {"run": TAG, "run_id": TAG_ID, "frontend": FRONTEND, "model": MODEL,
              # loss_fn, not loss: the per-epoch rows already carry `loss`
              # as the numeric training loss, and merging would drop this one.
              "dataset": DATASET, "pool": POOL, "loss_fn": LOSS,
              "target_size": TARGET_SIZE, "patch": PATCH, "seed": SEED}
    _log_df = pd.DataFrame([{**_ident, **_r} for _r in logs])
    # Concat rather than mode="a": a plain append writes no header, so rows
    # written before this stamp existed would silently take the new columns'
    # places. Rewriting is cheap -- the file holds one row per epoch per fold.
    if os.path.exists(lp):
        _log_df = pd.concat([pd.read_csv(lp), _log_df], ignore_index=True)
    _log_df.to_csv(lp, index=False)

# The aggregate is only meaningful over the whole CV, so it waits for every
# fold. With FOLD=k job splitting, the last job to finish writes the row.
have = pd.read_csv(folds_path, dtype={"name": str})
have = have[have["tag"] == TAG].drop_duplicates("fold").sort_values("fold")

print("\n" + "=" * 66)
print("Run9 | {} | {} + {} | {} classes | POOL={} LOSS={}".format(
    DATASET, FRONTEND, MODEL, NUM_CLASSES, POOL, LOSS))
print("-" * 66)
for _, r in have.iterrows():
    print("  fold {} [{}]  WA {:.2f}%  UAR {:.2f}%  ({} clips)".format(
        int(r["fold"]), r["name"], r["test_wa"], r["test_uar"], int(r["n_test"])))

if len(have) < len(folds):
    print("-" * 66)
    print("{} of {} folds done -- run the rest before reading a CV number.".format(
        len(have), len(folds)))
    print("=" * 66)
    raise SystemExit(0)

wa_m, wa_s = have["test_wa"].mean(), have["test_wa"].std(ddof=0)
ua_m, ua_s = have["test_uar"].mean(), have["test_uar"].std(ddof=0)
print("-" * 66)
print("  {}-fold CV   WA {:.2f} +- {:.2f} %   UA {:.2f} +- {:.2f} %".format(
    len(have), wa_m, wa_s, ua_m, ua_s))
print("  (chance = {:.1f}%)".format(100 / NUM_CLASSES))
print("  mean train WA {:.2f}%  ->  generalization gap {:.1f} points".format(
    have["train_wa"].mean(), have["train_wa"].mean() - wa_m))
print("=" * 66)

# Pooled confusion matrix across folds -- per-fold matrices stay on disk too.
cms = [pd.read_csv(cm_path.format(TAG_ID, int(r["fold"])), index_col=0)
       for _, r in have.iterrows()
       if os.path.exists(cm_path.format(TAG_ID, int(r["fold"])))]
if cms:
    cm_df = pd.DataFrame(sum(c.values for c in cms),
                         index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.index.name = "True \\ Pred"
    print("\nConfusion Matrix (all folds pooled):")
    print(cm_df.to_string())
    cm_df.to_csv(os.path.join(RESULT_DIR, "confusion_matrix.csv"))

run = {"script": "Run9.py", "frontend": FRONTEND, "model": MODEL,
       "dataset": DATASET, "classes": NUM_CLASSES,
       "iemocap_classes": IEMOCAP_CLASSES if DATASET == "iemocap" else "",
       "cv": "session5fold" if DATASET == "iemocap" else "actor5fold",
       "folds": len(have), "pool": POOL,
       "att_heads": ATT_HEADS if POOL == "attn" else "",
       "att_sharp": ATT_SHARP if POOL == "attn" else "",
       "embed_dim": EMBED_DIM, "loss": LOSS,
       "am_margin": AM_MARGIN if LOSS == "amsoftmax" else "",
       "am_scale": AM_SCALE if LOSS == "amsoftmax" else "",
       "epochs": NUM_EPOCHS, "batch": BATCH_SIZE, "lr": LR, "leaf_lr": LEAF_LR,
       "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
       "dropout": DROPOUT, "emb_dropout": EMB_DROPOUT, "dim": DIM,
       "depth": DEPTH, "heads": HEADS, "mlp_dim": MLP_DIM, "patch": PATCH,
       "dim_head": DIM_HEAD, "cnn_channels": CNN_CHANNELS,
       "convnext_size": CONVNEXT_SIZE if MODEL == "convnext" else "",
       "drop_path": DROP_PATH if MODEL == "convnext" else "",
       "convnext_pretrained": CONVNEXT_PRETRAINED if MODEL == "convnext" else "",
       "convnext_22k": CONVNEXT_22K if MODEL == "convnext" else "",
       "target_size": TARGET_SIZE, "leaf_filters": LEAF_N_FILTERS,
       "learn_pooling": LEARN_POOLING, "pcen": PCEN,
       "coord_channels": COORD_CHANNELS, "spec_augment": SPEC_AUGMENT,
       "freq_mask": FREQ_MASK, "time_mask": TIME_MASK, "normalize": NORMALIZE,
       "fixed_seconds": FIXED_SECONDS, "seed": SEED, "clips": len(audio_files),
       "train_wa": have["train_wa"].mean(),
       "test_wa": wa_m, "test_wa_std": wa_s,
       "test_uar": ua_m, "test_uar_std": ua_s,
       "fold_uars": json.dumps([round(float(u), 2) for u in have["test_uar"]])}

sweep = os.path.join(RESULT_DIR, "sweep.csv")
# Merge on column names rather than appending blind: the column set can grow
# over time, and a plain append would file new values under the old header.
row = pd.DataFrame([run])
if os.path.exists(sweep):
    row = pd.concat([pd.read_csv(sweep), row], ignore_index=True)
row.to_csv(sweep, index=False)
print(f"\nSaved to {RESULT_DIR}  (CV result appended to sweep.csv)")
