import os
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn, optim

import tensorflow as tf


_gpus = tf.config.list_physical_devices("GPU")
if os.environ.get("LEAF_ON_CPU", "0") not in ("0", "", "false", "False"):
    tf.config.set_visible_devices([], "GPU")
    _gpus = []
    print("LEAF_ON_CPU=1 -> TensorFlow restricted to CPU (ViT stays on GPU)")

for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass
tf.get_logger().setLevel("ERROR")

import functools

import leaf_audio.frontend as leaf_frontend_mod
from leaf_audio import initializers
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

DATASET_DIR = "Dataset2"
RESULT_DIR = "./results/LeafTF_VitCnnGlobal"
SAMPLE_RATE = 16000


def _env(name, default, cast=float):
    return cast(os.environ.get(name, str(default)))


LEAF_N_FILTERS = _env("LEAF_N_FILTERS", 64, int)
TARGET_SIZE = _env("TARGET_SIZE", 64, int)
WINDOW_LEN = _env("WINDOW_LEN", 25, float)
LEAF_LR = _env("LEAF_LR", 1e-5)
LEAF_PREEMP = _env("LEAF_PREEMP", 1, int)        # TF LEAF supports pre-emphasis
LEARN_POOLING = _env("LEARN_POOLING", 0, int)

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
CNN_CHANNELS = _env("CNN_CHANNELS", 64, int)     # CNN stem width (the "improved" part)
DROPOUT = _env("DROPOUT", 0.2)
EMB_DROPOUT = _env("EMB_DROPOUT", 0.1)
RESUME = _env("RESUME", 0, int)

# Which classifier to put on top of the LEAF features:
#   vit            Model/VitGlobal      plain ViT, global attention, no CNN stem
#   vit_local      Model/VitLocal       plain ViT, windowed local attention
#   cnn_vit        Model/VitCnnGlobal   CNN stem + global attention  (default)
#   cnn_vit_local  Model/VitCnnLocal    CNN stem + local attention
MODEL = os.environ.get("MODEL", "cnn_vit").lower()

SPLIT_MODE = os.environ.get("SPLIT_MODE", "speaker")
TEST_FRAC = _env("TEST_FRAC", 0.2)
VAL_FRAC = _env("VAL_FRAC", 0.1)
SEED = _env("SEED", 99, int)
DATASET = os.environ.get("DATASET", "auto").lower()


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)


set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"TensorFlow {tf.__version__} | GPUs visible to TF: {len(_gpus)}")
if not _gpus:
    print("WARNING: TensorFlow sees NO GPU -- the LEAF frontend will run on CPU.")
    print("         Install a CUDA-enabled tensorflow build (see Dockerfile).")

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

# One fixed input length for every batch. 0 = auto (75th percentile, capped at
# 6s). Constant shapes also stop TensorFlow retracing the frontend each batch.
FIXED_SECONDS = _env("FIXED_SECONDS", 0.0)
if FIXED_SECONDS <= 0:
    FIXED_SECONDS = min(6.0, max(2.0, float(np.percentile(lengths, 75)) / SAMPLE_RATE))
FIXED_SAMPLES = int(round(FIXED_SECONDS * SAMPLE_RATE))
_trunc = int((lengths > FIXED_SAMPLES).sum())
print(f"Fixed input window: {FIXED_SECONDS:.2f}s ({FIXED_SAMPLES} samples) | "
      f"{_trunc} of {len(lengths)} clips cropped ({_trunc / len(lengths) * 100:.1f}%)")

# TF LEAF runs its complex conv + squared modulus at FULL waveform resolution
# before pooling to frames, so the peak intermediate is
#   batch x (2 * n_filters) x samples x 4 bytes
# and the backward pass holds several copies of it. This is the tensor that
# OOMs -- it scales with the input window, not with the 64x64 output.
_peak_gb = BATCH_SIZE * 2 * LEAF_N_FILTERS * FIXED_SAMPLES * 4 / 1e9
print(f"LEAF peak intermediate: ~{_peak_gb:.2f} GB/copy "
      f"({BATCH_SIZE} x {2 * LEAF_N_FILTERS} x {FIXED_SAMPLES})")
if _peak_gb > 0.5 and not _env("ALLOW_BIG_BATCH", 0, int):
    _fit = max(2, int(0.25e9 // (2 * LEAF_N_FILTERS * FIXED_SAMPLES * 4)))
    raise SystemExit(
        "\nRefusing to start: the LEAF intermediate is ~{:.2f} GB per copy and the\n"
        "backward pass holds several of them. This OOMs on a 12 GB GPU.\n"
        "\n"
        "  Why: TF LEAF runs its complex conv + squared modulus at FULL waveform\n"
        "  resolution before pooling, so memory scales with\n"
        "      BATCH_SIZE x (2 * LEAF_N_FILTERS) x samples x 4 bytes\n"
        "  = {} x {} x {} x 4 = {:.2f} GB\n"
        "\n"
        "  Fix: add   -e BATCH_SIZE={} -e FIXED_SECONDS=3\n"
        "  Override:  -e ALLOW_BIG_BATCH=1   (only on a larger GPU)\n"
        .format(_peak_gb, BATCH_SIZE, 2 * LEAF_N_FILTERS, FIXED_SAMPLES,
                _peak_gb, min(_fit, 8)))

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
    """(B, FIXED_SAMPLES) -- one constant shape for the whole run.

    Batch-max padding made epochs progressively slower: IEMOCAP clips run
    0.78s-17.3s so nearly every batch was a new shape. That hurts twice here --
    the CUDA allocator keeps a block per shape, and TensorFlow RETRACES the
    Keras frontend for every unseen input shape, growing its function cache all
    run. A single fixed length removes both.

    Long clips are cropped at a random offset while training (cheap
    augmentation) and from the start at eval; short clips are zero-padded.
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
# TF LEAF frontend (on GPU) + DLPack bridge
# ---------------------------------------------------------------------------
compression_fn = functools.partial(leaf_frontend_mod.log_compression, log_offset=1e-5)
complex_conv_init = initializers.GaborInit(
    sample_rate=SAMPLE_RATE, min_freq=60.0, max_freq=7800.0)

leaf = leaf_frontend_mod.Leaf(
    learn_pooling=bool(LEARN_POOLING),
    n_filters=LEAF_N_FILTERS,
    window_len=WINDOW_LEN,
    sample_rate=SAMPLE_RATE,
    preemp=bool(LEAF_PREEMP),
    compression_fn=compression_fn,
    complex_conv_init=complex_conv_init,
)
# Build the Keras layer so trainable_variables exists before the optimizer.
leaf(tf.zeros((1, SAMPLE_RATE), dtype=tf.float32), training=False)

def _leaf_is_finite():
    return all(bool(np.isfinite(v.numpy()).all()) for v in leaf.trainable_variables)


leaf_weights_path = os.path.join(RESULT_DIR, "leaf_weights.h5")
# RESUME defaults to 0: each run starts from a fresh mel initialization, so one
# diverged run cannot poison later ones and ablation runs stay independent.
if RESUME and os.path.exists(leaf_weights_path):
    leaf.load_weights(leaf_weights_path)
    if _leaf_is_finite():
        print(f"Loaded LEAF weights from {leaf_weights_path}")
    else:
        raise SystemExit(f"{leaf_weights_path} contains non-finite values; "
                         f"delete it or run without RESUME=1.")
elif os.path.exists(leaf_weights_path):
    print(f"Ignoring existing {os.path.basename(leaf_weights_path)} "
          f"(set RESUME=1 to continue from it).")

_use_dlpack = bool(_gpus) and device == "cuda"


def tf_to_torch(t):
    """TF tensor -> torch tensor, staying on GPU when DLPack is available."""
    if _use_dlpack:
        try:
            from torch.utils import dlpack as tdl
            return tdl.from_dlpack(tf.experimental.dlpack.to_dlpack(t)).clone()
        except Exception:
            pass
    return torch.from_numpy(t.numpy()).to(device)


def torch_to_tf(t):
    """torch tensor -> TF tensor, staying on GPU when DLPack is available."""
    t = t.contiguous()
    if _use_dlpack:
        try:
            from torch.utils import dlpack as tdl
            return tf.experimental.dlpack.from_dlpack(tdl.to_dlpack(t))
        except Exception:
            pass
    return tf.constant(t.detach().cpu().numpy(), dtype=tf.float32)


def leaf_forward(waveforms, training):
    """(B,T) numpy -> TF tape (or None) and TF features resized to the ViT input.

    TF LEAF emits (B, frames, n_filters); resizing to (TARGET, TARGET) keeps the
    original time-on-rows orientation used by the earlier runs.
    """
    batch_tf = tf.constant(waveforms)
    if training:
        tape = tf.GradientTape()
        with tape:
            feats = leaf(batch_tf, training=True)
            feats = tf.expand_dims(feats, -1)
            feats = tf.image.resize(feats, (TARGET_SIZE, TARGET_SIZE))
            feats = tf.squeeze(feats, -1)
        return tape, feats
    feats = leaf(batch_tf, training=False)
    feats = tf.expand_dims(feats, -1)
    feats = tf.image.resize(feats, (TARGET_SIZE, TARGET_SIZE))
    return None, tf.squeeze(feats, -1)


# ---------------------------------------------------------------------------
# Model: CNN-stem ViT
# ---------------------------------------------------------------------------
_MODELS = {"vit": VitGlobal, "vit_local": VitLocal,
           "cnn_vit": VitCnnGlobal, "cnn_vit_local": VitCnnLocal}
if MODEL not in _MODELS:
    raise SystemExit(f"MODEL={MODEL!r} unknown; choose one of {sorted(_MODELS)}")

# The four variants take different kwargs: only the cnn_* ones accept
# cnn_channels, and VitCnnLocal does not accept dim_head.
_kw = dict(image_size=(TARGET_SIZE, TARGET_SIZE), patch_size=(PATCH, PATCH),
           num_classes=NUM_CLASSES, dim=DIM, depth=DEPTH, heads=HEADS,
           mlp_dim=MLP_DIM, channels=1,
           dropout=DROPOUT, emb_dropout=EMB_DROPOUT)
if MODEL.startswith("cnn_"):
    _kw["cnn_channels"] = CNN_CHANNELS
if MODEL != "cnn_vit_local":
    _kw["dim_head"] = DIM_HEAD

vit_model = _MODELS[MODEL].ViT(**_kw).to(device)
print(f"Model: {MODEL} ({_MODELS[MODEL].__name__})")

print(f"Trainable params -- LEAF(tf): {sum(int(np.prod(v.shape)) for v in leaf.trainable_variables)}, "
      f"ViT+CNN: {sum(p.numel() for p in vit_model.parameters())}")

optimizer = optim.AdamW(vit_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
leaf_optimizer = tf.keras.optimizers.Adam(learning_rate=LEAF_LR)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def lr_scale(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / max(1, WARMUP_EPOCHS)
    p = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * p))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


@torch.no_grad()
def _predict(indices):
    vit_model.eval()
    preds = []
    for i in range(0, len(indices), BATCH_SIZE):
        idx = indices[i:i + BATCH_SIZE]
        _, feats_tf = leaf_forward(make_batch(idx), training=False)
        feat_t = tf_to_torch(feats_tf).float().unsqueeze(1).to(device)
        preds.append(vit_model(feat_t).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate(indices):
    preds = _predict(indices)
    tgts = all_labels[indices]
    wa = float((preds == tgts).mean())
    recalls = [float((preds[tgts == c] == c).mean())
               for c in range(NUM_CLASSES) if (tgts == c).any()]
    return wa, float(np.mean(recalls)), preds, tgts


# ---------------------------------------------------------------------------
# Training -- manual TF/torch chain rule
# ---------------------------------------------------------------------------
n_train = len(train_idx)
num_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\nTraining {NUM_EPOCHS} epoch(s) | {n_train} train samples | "
      f"batch_size={BATCH_SIZE} | {num_batches} batches/epoch")
print(f"TF<->torch transfer: {'DLPack (GPU-resident)' if _use_dlpack else 'numpy (CPU round trip)'}\n")

os.makedirs(RESULT_DIR, exist_ok=True)
best_path = os.path.join(RESULT_DIR, "best.pth")
train_log, best_uar = [], -1.0

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    vit_model.train()
    perm = train_idx[np.random.permutation(n_train)]
    epoch_loss, epoch_correct = 0.0, 0

    for b in range(num_batches):
        idx = perm[b * BATCH_SIZE:min((b + 1) * BATCH_SIZE, n_train)]
        bs = len(idx)

        # 1. frontend forward, recorded on the TF tape
        tape, feats_tf = leaf_forward(make_batch(idx, train=True), training=True)

        # 2. hand the features to PyTorch; this tensor is the seam between the
        #    two autograd systems, so it must be a leaf that requires grad.
        feat_t = tf_to_torch(feats_tf).float().unsqueeze(1).to(device)
        feat_t.requires_grad_(True)
        target_t = torch.tensor(all_labels[idx], dtype=torch.long, device=device)

        # 3. ViT forward/backward
        optimizer.zero_grad(set_to_none=True)
        output = vit_model(feat_t)
        loss = criterion(output, target_t)
        loss.backward()
        optimizer.step()

        # 4. push dL/dfeatures back across the seam so the tape can finish the
        #    chain rule into the Gabor filters.
        grad_tf = torch_to_tf(feat_t.grad.squeeze(1))
        leaf_grads = tape.gradient(feats_tf, leaf.trainable_variables,
                                   output_gradients=grad_tf)
        pairs = [(g, v) for g, v in zip(leaf_grads, leaf.trainable_variables)
                 if g is not None]
        if pairs:
            leaf_optimizer.apply_gradients(pairs)
        del tape

        epoch_loss += loss.item() * bs
        epoch_correct += (output.argmax(1) == target_t).sum().item()

    scheduler.step()
    # TF and torch each hold their own GPU cache; release torch's every epoch so
    # the two allocators do not slowly squeeze each other over a long run.
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
        if _leaf_is_finite():
            leaf.save_weights(leaf_weights_path)
        star = "  <- best"
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]  loss={avg_loss:.4f}  train={acc:.1f}%  "
          f"val_WA={val_wa * 100:.1f}%  val_UAR={val_uar * 100:.1f}%  "
          f"time={elapsed:.1f}s{star}")

# ---------------------------------------------------------------------------
# Test on the best-val checkpoint
# ---------------------------------------------------------------------------
vit_model.load_state_dict(torch.load(best_path, map_location=device)["vit"])
leaf.load_weights(leaf_weights_path)

test_wa, test_uar, test_preds, test_targets = evaluate(test_idx)
train_wa, _, _, _ = evaluate(train_idx)

print("\n" + "=" * 66)
print(f"TEST -- TF LEAF + VitCnnGlobal | {DATASET} | {SPLIT_MODE} split | "
      f"{len(test_idx)} clips")
print(f"  WA  {test_wa * 100:.2f}%")
print(f"  UAR {test_uar * 100:.2f}%   (chance = {100 / NUM_CLASSES:.1f}%)")
print(f"  train WA {train_wa * 100:.2f}%  ->  generalization gap "
      f"{(train_wa - test_wa) * 100:.1f} points")
print("=" * 66)

# ---------------------------------------------------------------------------
# Confusion matrix + logs
# ---------------------------------------------------------------------------
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

run = {"frontend": "leaf-audio-tf", "model": "VitCnnGlobal", "dataset": DATASET,
       "classes": NUM_CLASSES, "split_mode": SPLIT_MODE, "epochs": NUM_EPOCHS,
       "batch": BATCH_SIZE, "lr": LR, "leaf_lr": LEAF_LR,
       "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
       "dropout": DROPOUT, "emb_dropout": EMB_DROPOUT, "dim": DIM, "depth": DEPTH,
       "heads": HEADS, "mlp_dim": MLP_DIM, "patch": PATCH, "dim_head": DIM_HEAD,
       "cnn_channels": CNN_CHANNELS, "target_size": TARGET_SIZE,
       "fixed_seconds": FIXED_SECONDS,
       "leaf_filters": LEAF_N_FILTERS, "preemp": LEAF_PREEMP,
       "learn_pooling": LEARN_POOLING, "seed": SEED,
       "best_val_uar": best_uar * 100, "test_wa": test_wa * 100,
       "test_uar": test_uar * 100, "train_wa": train_wa * 100}
sweep = os.path.join(RESULT_DIR, "sweep.csv")
pd.DataFrame([run]).to_csv(sweep, mode="a", header=not os.path.exists(sweep), index=False)

print(f"\nSaved to {RESULT_DIR}  (results appended to sweep.csv)")
