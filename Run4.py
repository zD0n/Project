import os
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.config.set_visible_devices([], "GPU")
import functools
import leaf_audio.frontend as frontend
from leaf_audio import initializers

from torch import nn, optim
from Model import VitGlobal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMOTION_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "neutral": 3,
    "sad": 4,
    "happy": 5,
}

DATASET_DIR = "Dataset2"
LEAF_N_FILTERS = 64
TARGET_SIZE = 64
SAMPLE_RATE = 16000
WINDOW_LEN = 25
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "10"))
BATCH_SIZE = 32
LR = 1e-4
RESULT_DIR = "./results/Leaf_VitGlobal"
SEED = 99


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass


set_seed(SEED)

# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
labels_df = pd.read_csv(os.path.join(DATASET_DIR, "labels.csv"))
labels_dict = dict(zip(labels_df["Filename"], labels_df["Label"]))

audio_dir = os.path.join(DATASET_DIR, "audios")
audio_files = sorted(os.listdir(audio_dir))

print(f"Loading {len(audio_files)} audio files...")

all_audio = []
all_labels = []

for fname in audio_files:
    base = os.path.splitext(fname)[0]
    label_str = labels_dict[base]
    label_idx = EMOTION_MAP[label_str]

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


def make_batch(indices):
    batch_list = [all_audio[i] for i in indices]
    bs = len(batch_list)
    batch_len = max(x.shape[0] for x in batch_list)
    batch = np.zeros((bs, batch_len), dtype=np.float32)
    for j, x in enumerate(batch_list):
        n = x.shape[0]
        batch[j, :n] = x
    return batch


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
compression_fn = functools.partial(frontend.log_compression, log_offset=1e-5)
complex_conv_init = initializers.GaborInit(
    sample_rate=SAMPLE_RATE, min_freq=60.0, max_freq=7800.0
)

leaf_frontend = frontend.Leaf(
    learn_pooling=False,
    n_filters=LEAF_N_FILTERS,
    window_len=WINDOW_LEN,
    sample_rate=SAMPLE_RATE,
    preemp=True,
    compression_fn=compression_fn,
    complex_conv_init=complex_conv_init,
)

leaf_weights_path = os.path.join(RESULT_DIR, "leaf_weights.h5")
dummy_audio = tf.zeros((1, SAMPLE_RATE), dtype=tf.float32)
leaf_frontend(dummy_audio, training=False)
if os.path.exists(leaf_weights_path):
    leaf_frontend.load_weights(leaf_weights_path)
    print(f"Loaded Leaf weights from {leaf_weights_path}")

vit_model = VitGlobal.ViT(
    image_size=(TARGET_SIZE, TARGET_SIZE),
    patch_size=(8, 8),
    num_classes=len(EMOTION_MAP),
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=1024,
    channels=1,
    dropout=0.1,
    emb_dropout=0.1,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model = vit_model.to(device)
print(f"Device: {device}")

leaf_optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
vit_optimizer = optim.Adam(vit_model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
num_samples = len(all_audio)
num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
print(
    f"\nTraining {NUM_EPOCHS} epoch(s) | {num_samples} samples | batch_size={BATCH_SIZE} | {num_batches} batches/epoch\n"
)

train_log = []

for epoch in range(NUM_EPOCHS):
    t0 = time.time()

    perm = np.random.permutation(num_samples)
    epoch_loss = 0.0
    epoch_correct = 0

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, num_samples)
        batch_idx_arr = perm[start:end]
        bs = len(batch_idx_arr)

        # Batch tensors
        batch_audio = tf.constant(make_batch(batch_idx_arr))  # (bs, batch_len)
        batch_labels = all_labels[batch_idx_arr]

        # ---- Leaf forward (TF) inside tape ----
        with tf.GradientTape() as tape:
            features_tf = leaf_frontend(batch_audio, training=True)  # (bs, T, 64)
            features_4d = tf.expand_dims(features_tf, -1)  # (bs, T, 64, 1)
            features_resized = tf.image.resize(features_4d, (TARGET_SIZE, TARGET_SIZE))
            features_resized = tf.squeeze(features_resized, -1)  # (bs, 64, 64)

        # ---- Bridge: TF → PyTorch ----
        feat_t = torch.from_numpy(features_resized.numpy()).float()  # (bs, 64, 64)
        feat_t = feat_t.unsqueeze(1).to(device)  # (bs, 1, 64, 64)
        feat_t.requires_grad_(True)

        target_t = torch.tensor(batch_labels, dtype=torch.long, device=device)

        # ---- ViT forward + loss ----
        vit_optimizer.zero_grad()
        output = vit_model(feat_t)
        loss = criterion(output, target_t)

        # ---- ViT backward ----
        loss.backward()
        vit_optimizer.step()

        # ---- Leaf backward ----
        grad_np = feat_t.grad.detach().cpu().numpy()
        grad_tf = tf.constant(grad_np, dtype=tf.float32)
        leaf_grads = tape.gradient(
            features_resized,
            leaf_frontend.trainable_variables,
            output_gradients=grad_tf,
        )
        leaf_optimizer.apply_gradients(
            zip(leaf_grads, leaf_frontend.trainable_variables)
        )

        epoch_loss += loss.item() * bs
        epoch_correct += (
            (output.argmax(1).cpu() == torch.tensor(batch_labels)).sum().item()
        )

    avg_loss = epoch_loss / num_samples
    acc = epoch_correct / num_samples * 100
    elapsed = time.time() - t0
    train_log.append(
        {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": acc,
            "time_s": round(elapsed, 1),
        }
    )
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]  "
        f"loss={avg_loss:.4f}  acc={acc:.1f}%  "
        f"time={elapsed:.1f}s"
    )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
leaf_frontend.training = False
vit_model.eval()

all_preds = []

for i in range(0, num_samples, BATCH_SIZE):
    batch_audio = tf.constant(make_batch(range(i, min(i + BATCH_SIZE, num_samples))))
    batch_labels = all_labels[i : i + BATCH_SIZE]

    with tf.GradientTape():
        features = leaf_frontend(batch_audio, training=False)
        features_4d = tf.expand_dims(features, -1)
        features_resized = tf.image.resize(features_4d, (TARGET_SIZE, TARGET_SIZE))
        features_resized = tf.squeeze(features_resized, -1)

    feat_t = torch.from_numpy(features_resized.numpy()).float().unsqueeze(1).to(device)

    with torch.no_grad():
        out = vit_model(feat_t)
        pred = out.argmax(1).cpu().numpy()
        all_preds.extend(pred)

all_preds = np.array(all_preds)
correct = (all_preds == all_labels).sum()
print(
    f"\nInference on full dataset: {correct}/{num_samples} = {correct / num_samples * 100:.1f}%"
)

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
class_names = list(EMOTION_MAP.keys())
num_classes = len(class_names)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for t, p in zip(all_labels, all_preds):
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

os.makedirs(RESULT_DIR, exist_ok=True)

torch.save(vit_model.state_dict(), os.path.join(RESULT_DIR, "model_best.pth"))
leaf_frontend.save_weights(leaf_weights_path)
cm_df.to_csv(os.path.join(RESULT_DIR, "confusion_matrix.csv"))
pd.DataFrame(train_log).to_csv(
    os.path.join(RESULT_DIR, "training_log.csv"), index=False
)

print(f"\nSaved to {RESULT_DIR}")
