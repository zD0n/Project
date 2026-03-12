# Mel

import librosa
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset

import pandas as pd


def preprocess(mapping, path):
    print("Im Loading Please wait for awhile.")
    """
    ---------
    """
    # print(path)
    contents = os.listdir(path)
    # print(contents)
    for i in contents:
        if i.endswith(".csv"):
            label_file = rf"{path}\{i}".replace("\\", "/")
        else:
            data_dir = rf"{path}\{i}".replace("\\", "/")
    emotion_map = mapping

    """
    ---------
    """

    def extract_features(file_path, n_mels=64, max_len=128):
        y, sr = librosa.load(file_path, sr=None)

        # compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

        # convert to log scale (recommended)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # pad or crop time axis
        if mel_spec.shape[1] < max_len:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, max_len - mel_spec.shape[1])))
        else:
            mel_spec = mel_spec[:, :max_len]

        return mel_spec

    """
    ---------
    """
    # print(label_file)
    labels_df = pd.read_csv(label_file)

    X_list = []
    y_list = []

    for idx, row in labels_df.iterrows():
        filename = row["Filename"] + ".wav"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            # print(f"File not found: {file_path}")
            continue

        emotion = row["Label"].lower()
        if emotion not in emotion_map:
            # print(f"Skipping unknown emotion: {emotion}")
            continue

        features = extract_features(file_path)
        X_list.append(features)
        y_list.append(emotion_map[emotion])

    print("Total samples loaded:", len(X_list), len(y_list))

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    X = X.unsqueeze(1)
    y = torch.tensor(np.array(y_list), dtype=torch.long)

    return TensorDataset(X, y)
