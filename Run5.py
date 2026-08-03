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
from Model import VitGlobal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# CREMA-D ships 6 emotions and labels them lowercase.
CREMAD_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "neutral": 3,
    "sad": 4,
    "happy": 5,
}

# IEMOCAP: 5 classes, with excited kept SEPARATE from happy (the 4-class
# convention usually merges them). Frustration / Other / Surprise / Disgust /
# Fear are dropped -- not in this class set.
IEMOCAP_MAP = {
    "neutral": 0,
    "sad": 1,
    "anger": 2,
    "happy": 3,
    "excited": 4,
}

# IEMOCAP writes "Sadness"/"Happiness"/"Anger" and uses BOTH "Excited" and
# "Excitement" for the same class; CREMA-D writes "sad"/"happy".
LABEL_ALIASES = {
    "sadness": "sad",
    "happiness": "happy",
    "excitement": "excited",
    "angry": "anger",
    "neutral state": "neutral",
}

DATASET_DIR = "Dataset2"
RESULT_DIR = "./results/Leaf_VitGlobal"
SAMPLE_RATE = 16000


def _env(name, default, cast=float):
    return cast(os.environ.get(name, str(default)))


# Every knob is env-overridable so a sweep needs no image rebuild:
#   docker run ... -e LR=1e-4 -e DROPOUT=0.3 -e NUM_EPOCHS=80 model
LEAF_N_FILTERS = _env("LEAF_N_FILTERS", 64, int)
TARGET_SIZE = _env("TARGET_SIZE", 64, int)
WINDOW_LEN = _env("WINDOW_LEN", 25, float)
LEAF_LR = _env("LEAF_LR", 1e-5)          # frontend learns far slower than the head

NUM_EPOCHS = _env("NUM_EPOCHS", 60, int)
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
DROPOUT = _env("DROPOUT", 0.2)
EMB_DROPOUT = _env("EMB_DROPOUT", 0.1)

# speaker = actor-independent (correct for SER); random = the leaky version, kept
# so the inflation from speaker leakage can be measured rather than argued about.
SPLIT_MODE = os.environ.get("SPLIT_MODE", "speaker")
TEST_FRAC = _env("TEST_FRAC", 0.2)
VAL_FRAC = _env("VAL_FRAC", 0.1)
SEED = _env("SEED", 99, int)

# cremad | iemocap | auto (inferred from the label vocabulary in labels.csv)
DATASET = os.environ.get("DATASET", "auto").lower()


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no CUDA device visible -- run with `docker run --gpus all`.")

# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
labels_df = pd.read_csv(os.path.join(DATASET_DIR, "labels.csv"))
name_col = "Filename" if "Filename" in labels_df.columns else "filename"
labels_dict = dict(zip(labels_df[name_col], labels_df["Label"]))


def canon(label):
    """'Sadness' -> 'sad'. Case/alias normalization shared by both corpora."""
    s = str(label).strip().lower()
    return LABEL_ALIASES.get(s, s)


vocab = {canon(v) for v in labels_dict.values()}
if DATASET == "auto":
    # IEMOCAP is the only one of the two with excited/frustration in its vocabulary.
    DATASET = "iemocap" if vocab & {"excited", "frustration", "other"} else "cremad"

EMOTION_MAP = IEMOCAP_MAP if DATASET == "iemocap" else CREMAD_MAP
NUM_CLASSES = len(EMOTION_MAP)
# Per-dataset results dir: a 5-class IEMOCAP checkpoint must not overwrite a
# 6-class CREMA-D one, and their sweep rows must stay separable.
RESULT_DIR = os.path.join(RESULT_DIR, DATASET)


def speaker_of(fname):
    """Speaker id used to keep the split speaker-independent.

    CREMA-D  1001_DFA_ANG_XX.wav  -> '1001'          (91 actors)
    IEMOCAP  Ses01F_impro01_F000  -> 'Ses01F'        (10 speakers: 5 sessions x2)
             the trailing field's letter is the speaker's gender in that session,
             which is what distinguishes the two actors within a session.
    """
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

# Keep only files whose label belongs to this dataset's class set. IEMOCAP in
# particular carries Frustration/Other/etc. that would otherwise KeyError.
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

all_audio = []
all_labels = []

for fname in audio_files:
    base = os.path.splitext(fname)[0]
    label_idx = EMOTION_MAP[canon(labels_dict[base])]

    audio = sf.read(os.path.join(audio_dir, fname), dtype="float32")
    if isinstance(audio, tuple):
        audio = audio[0]
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]

    all_audio.append(audio)
    all_labels.append(label_idx)

all_labels = np.array(all_labels, dtype=np.int64)
lengths = np.array([a.shape[0] for a in all_audio])
print(
    f"Loaded {len(all_audio)} files | max length: {lengths.max()} samples | "
    f"total {lengths.sum() / SAMPLE_RATE:.1f}s"
)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
actors = np.array([speaker_of(f) for f in audio_files])
rng = np.random.RandomState(SEED)

if SPLIT_MODE == "speaker":
    uniq = np.unique(actors)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * TEST_FRAC)))
    n_val = max(1, int(round(len(uniq) * VAL_FRAC)))
    test_a = set(uniq[:n_test])
    val_a = set(uniq[n_test:n_test + n_val])
    test_idx = np.array([i for i, a in enumerate(actors) if a in test_a])
    val_idx = np.array([i for i, a in enumerate(actors) if a in val_a])
    train_idx = np.array([i for i, a in enumerate(actors)
                          if a not in test_a and a not in val_a])
    held_out = "{} test actors, {} val actors, {} train actors".format(
        len(test_a), len(val_a), len(uniq) - len(test_a) - len(val_a))
else:
    perm_all = rng.permutation(len(audio_files))
    n_test = int(round(len(perm_all) * TEST_FRAC))
    n_val = int(round(len(perm_all) * VAL_FRAC))
    test_idx, val_idx, train_idx = (perm_all[:n_test],
                                    perm_all[n_test:n_test + n_val],
                                    perm_all[n_test + n_val:])
    held_out = "random split -- speakers appear in BOTH train and test"

print("\nSplit ({}): train {} | val {} | test {}".format(
    SPLIT_MODE, len(train_idx), len(val_idx), len(test_idx)))
print("  " + held_out)
if SPLIT_MODE == "speaker":
    overlap = (set(actors[train_idx]) & set(actors[test_idx])) | \
              (set(actors[train_idx]) & set(actors[val_idx]))
    print("  speaker overlap train/eval: {} (must be 0)".format(len(overlap)))


def make_batch(indices):
    batch_list = [all_audio[i] for i in indices]
    bs = len(batch_list)
    batch_len = max(x.shape[0] for x in batch_list)
    batch = np.zeros((bs, batch_len), dtype=np.float32)
    for j, x in enumerate(batch_list):
        n = x.shape[0]
        batch[j, :n] = x
    return batch


def leaf_features(waveforms):
    """(B, T) numpy -> (B, 1, TARGET_SIZE, TARGET_SIZE) on `device`.

    leaf_pytorch emits (B, n_filters, frames) i.e. freq-major, while the old TF
    Leaf emitted (B, frames, n_filters). The transpose below keeps the image
    orientation the ViT was trained on (time on rows, frequency on columns).
    """
    x = torch.from_numpy(waveforms).to(device, non_blocking=True).unsqueeze(1)
    feats = leaf_frontend(x)                       # (B, 64, frames)
    feats = torch.log(feats + 1e-5)                # == TF log_compression(1e-5)
    feats = feats.transpose(1, 2).unsqueeze(1)     # (B, 1, frames, 64)
    return F.interpolate(feats, size=(TARGET_SIZE, TARGET_SIZE),
                         mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
leaf_frontend = Leaf(
    n_filters=LEAF_N_FILTERS,
    sample_rate=SAMPLE_RATE,
    window_len=WINDOW_LEN,
    preemp=False,           # see module docstring -- not implemented in leaf_pytorch
    init_min_freq=60.0,
    init_max_freq=7800.0,
    pcen_compression=False,  # log compression applied explicitly in leaf_features
).to(device)

# Old TF config used learn_pooling=False; freezing the Gaussian lowpass keeps
# only the Gabor filterbank learnable, as before.
for p in leaf_frontend._pooling.parameters():
    p.requires_grad_(False)

leaf_weights_path = os.path.join(RESULT_DIR, "leaf_weights.pth")
if os.path.exists(leaf_weights_path):
    leaf_frontend.load_state_dict(torch.load(leaf_weights_path, map_location=device))
    print(f"Loaded Leaf weights from {leaf_weights_path}")

vit_model = VitGlobal.ViT(
    image_size=(TARGET_SIZE, TARGET_SIZE),
    patch_size=(PATCH, PATCH),
    num_classes=NUM_CLASSES,
    dim=DIM,
    depth=DEPTH,
    heads=HEADS,
    mlp_dim=MLP_DIM,
    channels=1,
    dropout=DROPOUT,
    emb_dropout=EMB_DROPOUT,
).to(device)

leaf_trainable = [p for p in leaf_frontend.parameters() if p.requires_grad]
print(f"Trainable params -- leaf: {sum(p.numel() for p in leaf_trainable)}, "
      f"vit: {sum(p.numel() for p in vit_model.parameters())}")

# AdamW (decoupled weight decay) is the standard optimizer for a ViT trained from
# scratch; the frontend gets its own much smaller lr and no decay, since 512 Gabor
# parameters do not need regularizing and destabilize easily.
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
    """WA (plain accuracy) and UAR (mean per-class recall) over `indices`."""
    leaf_frontend.eval()
    vit_model.eval()
    preds = []
    for i in range(0, len(indices), BATCH_SIZE):
        idx = indices[i:i + BATCH_SIZE]
        out = vit_model(leaf_features(make_batch(idx)))
        preds.append(out.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    tgts = all_labels[indices]
    wa = float((preds == tgts).mean())
    # UAR: CREMA-D is near-balanced but neutral has ~15% fewer clips, and UAR is
    # what the SER literature reports.
    recalls = [float((preds[tgts == c] == c).mean())
               for c in range(NUM_CLASSES) if (tgts == c).any()]
    return wa, float(np.mean(recalls)), preds, tgts

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
num_samples = len(all_audio)
n_train = len(train_idx)
num_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
print(
    f"\nTraining {NUM_EPOCHS} epoch(s) | {n_train} train samples | batch_size={BATCH_SIZE} | {num_batches} batches/epoch\n"
)

os.makedirs(RESULT_DIR, exist_ok=True)
best_path = os.path.join(RESULT_DIR, "best.pth")
train_log = []
best_uar = -1.0

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    leaf_frontend.train()
    vit_model.train()

    perm = train_idx[np.random.permutation(n_train)]
    epoch_loss = 0.0
    epoch_correct = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_train)
        batch_idx_arr = perm[start:end]
        bs = len(batch_idx_arr)

        feat_t = leaf_features(make_batch(batch_idx_arr))
        target_t = torch.tensor(all_labels[batch_idx_arr], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        output = vit_model(feat_t)
        loss = criterion(output, target_t)
        loss.backward()          # flows through the ViT AND the Gabor filters
        optimizer.step()

        epoch_loss += loss.item() * bs
        epoch_correct += (output.argmax(1) == target_t).sum().item()

    scheduler.step()
    avg_loss = epoch_loss / n_train
    acc = epoch_correct / n_train * 100
    val_wa, val_uar, _, _ = evaluate(val_idx)
    elapsed = time.time() - t0

    train_log.append(
        {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_acc": acc,
            "val_wa": val_wa * 100,
            "val_uar": val_uar * 100,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(elapsed, 1),
        }
    )
    # Select on val UAR, never on train accuracy -- train accuracy only measures
    # memorization and will keep rising after the model stops generalizing.
    star = ""
    if val_uar > best_uar:
        best_uar = val_uar
        torch.save({"vit": vit_model.state_dict(),
                    "leaf": leaf_frontend.state_dict()}, best_path)
        star = "  <- best"
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]  "
        f"loss={avg_loss:.4f}  train={acc:.1f}%  "
        f"val_WA={val_wa * 100:.1f}%  val_UAR={val_uar * 100:.1f}%  "
        f"time={elapsed:.1f}s{star}"
    )

# ---------------------------------------------------------------------------
# Evaluation -- held-out test split, best-val checkpoint
# ---------------------------------------------------------------------------
state = torch.load(best_path, map_location=device)
vit_model.load_state_dict(state["vit"])
leaf_frontend.load_state_dict(state["leaf"])

test_wa, test_uar, all_preds, test_targets = evaluate(test_idx)
train_wa, train_uar, _, _ = evaluate(train_idx)

print("\n" + "=" * 62)
print(f"TEST ({SPLIT_MODE} split, {len(test_idx)} clips, best-val checkpoint)")
print(f"  WA  {test_wa * 100:.2f}%")
print(f"  UAR {test_uar * 100:.2f}%   (chance = {100 / NUM_CLASSES:.1f}%)")
print(f"  train WA {train_wa * 100:.2f}%  ->  generalization gap "
      f"{(train_wa - test_wa) * 100:.1f} points")
print("=" * 62)

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
class_names = list(EMOTION_MAP.keys())
num_classes = len(class_names)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for t, p in zip(test_targets, all_preds):
    conf_matrix[t][p] += 1

rev_map = {v: k for k, v in EMOTION_MAP.items()}
cm_rows = []
for i in range(num_classes):
    row = {rev_map[i]: conf_matrix[i][0]}
    for j in range(1, num_classes):
        row[rev_map[j]] = conf_matrix[i][j]
    cm_rows.append(row)

cm_df = pd.DataFrame(cm_rows, index=[rev_map[i] for i in range(num_classes)])
cm_df.index.name = "True \\ Pred"

print(f"\nConfusion Matrix:")
print(cm_df.to_string())

torch.save(vit_model.state_dict(), os.path.join(RESULT_DIR, "model_best.pth"))
torch.save(leaf_frontend.state_dict(), leaf_weights_path)
cm_df.to_csv(os.path.join(RESULT_DIR, "confusion_matrix.csv"))
pd.DataFrame(train_log).to_csv(
    os.path.join(RESULT_DIR, "training_log.csv"), index=False
)

# One row per run, appended -- so a parameter sweep is directly comparable.
run = {"dataset": DATASET, "classes": NUM_CLASSES,
       "split_mode": SPLIT_MODE, "epochs": NUM_EPOCHS, "batch": BATCH_SIZE,
       "lr": LR, "leaf_lr": LEAF_LR, "weight_decay": WEIGHT_DECAY,
       "label_smoothing": LABEL_SMOOTHING, "dropout": DROPOUT,
       "emb_dropout": EMB_DROPOUT, "dim": DIM, "depth": DEPTH, "heads": HEADS,
       "mlp_dim": MLP_DIM, "patch": PATCH, "target_size": TARGET_SIZE,
       "leaf_filters": LEAF_N_FILTERS, "seed": SEED,
       "best_val_uar": best_uar * 100, "test_wa": test_wa * 100,
       "test_uar": test_uar * 100, "train_wa": train_wa * 100}
sweep = os.path.join(RESULT_DIR, "sweep.csv")
pd.DataFrame([run]).to_csv(sweep, mode="a", header=not os.path.exists(sweep), index=False)

print(f"\nSaved to {RESULT_DIR}  (results appended to sweep.csv)")