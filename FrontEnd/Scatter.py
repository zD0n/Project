#Scattering Transform
import os
import torch
from kymatio import Scattering1D
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset

def scatter(mapping,path2folder):
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

    for filename in os.listdir(path2folder):
        if not filename.lower().endswith(".wav"):
            continue

        audio_path = os.path.join(path2folder, filename)
        # print(f"Processing {filename} ...")

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

    return TensorDataset(X, y)
