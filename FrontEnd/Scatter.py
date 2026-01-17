#Scattering Transform
import os
import torch
from kymatio import Scattering1D
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import pandas as pd

def preprocess(mapping,path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    ---------
    """

    contents = os.listdir(path)
    for i in contents:
        if i.endswith(".csv"):
            label_file = f"{path}\{i}".replace("\\", "/")
        else:
            data_dir = f"{path}\{i}".replace("\\", "/")
    emotion_map = mapping

    labels_df = pd.read_csv(label_file)
    """
    ---------
    """
    X_list = []
    y_list = []

    T = 16000          # 1 second of audio at 16kHz
    J = 6              # scattering scale
    Q = 8              # wavelets per octave

    # Initialize scattering
    scattering = Scattering1D(J=J, shape=T, Q=Q, frontend='torch')
    scattering = scattering.to(device)



    for idx, row in labels_df.iterrows():
        filename = row['Filename'] + '.wav'
        print("Current File Working : ",filename)
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            # print(f"File not found: {file_path}")
            continue
        
            
        emotion = row['Label'].lower()
        if emotion not in emotion_map:
            # print(f"Skipping unknown emotion: {emotion}")
            continue

        y, sr = librosa.load(file_path, sr=16000)  # force 16kHz
        if len(y) < T:
            y = np.pad(y, (0, T - len(y)))
        else:
            y = y[:T]

        x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T]
        Sx = scattering(x)  # [1, channels, time]

        Sx = Sx.unsqueeze(1)  # [1, 1, channels, time]
        Sx_resized = F.interpolate(Sx, size=(64, 128), mode="bilinear")
        Sx_resized = Sx_resized.repeat(1, 1, 1, 1)  # [1, 3, 256, 256]
        Sx_resized = Sx_resized.squeeze(0)          # [3, 256, 256]

        X_list.append(Sx_resized.cpu())
        y_list.append(emotion_map[emotion])

        
    print("Total samples loaded:", len(X_list), len(y_list))
    # ===========================
    # Final dataset tensors
    # ===========================
    X = torch.stack(X_list)  # [num_samples, 3, 256, 256]
    y = torch.tensor(y_list)       # [num_samples]

    print("Final dataset shapes:")
    print("  X:", X.shape)
    print("  y:", y.shape)

    return TensorDataset(X, y)
