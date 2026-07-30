import os
import torch
import pickle
import torchaudio
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset


def loadpretrain():
    from models.classifier import Classifier

    hparams_path = r"../weight/efficientnet_leaf/hparams.pickle"
    ckpt_path = r"../weight/efficientnet_leaf/epoch=100_tr_loss=0.067792_tr_acc=0.980434_val_acc=0.954013.pth"
    checkpoint = torch.load(ckpt_path, weights_only=False)
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    model = Classifier(hparams.cfg)
    print(model.load_state_dict(checkpoint["model_state_dict"]))

    return model, hparams


def preprocess(mapping, path):

    # old_path = os.getcwd()
    # new_path = rf"{old_path}\FrontEnd"
    # os.chdir(new_path)
    model, hparams = loadpretrain()
    print("Model Loaded...")
    frontend = model.features
    """
    ---------
    """
    # print(path)
    print("Looking for Files...")
    contents = os.listdir(path)
    # print(contents)
    for i in contents:
        if i.endswith(".csv"):
            label_file = rf"{path}\{i}".replace("\\", "/")
        else:
            data_dir = rf"{path}\{i}".replace("\\", "/")
    emotion_map = mapping

    labels_df = pd.read_csv(label_file)
    """
    ---------
    """
    X_list = []
    y_list = []
    print("Start Extracting")
    print("Notice : This Process May take awhile. Try go Grab some Coffee..")
    for idx, row in labels_df.iterrows():
        filename = row["Filename"] + ".wav"
        print("Current File Working : ",filename)
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            # print(f"File not found: {file_path}")
            continue

        emotion = row["Label"].lower()
        if emotion not in emotion_map:
            # print(f"Skipping unknown emotion: {emotion}")
            continue

        # ---------------------------
        # Load and preprocess audio
        # ---------------------------
        waveform, sample_rate = torchaudio.load(file_path)  # [channels, time]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        expected_sr = hparams.cfg["audio_config"]["sample_rate"]
        if sample_rate != expected_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=expected_sr
            )
            waveform = resampler(waveform)

        waveform = waveform.unsqueeze(0).float().cpu()  # [1, 1, time]

        # ---------------------------
        # Forward pass
        # ---------------------------
        with torch.no_grad():
            features = frontend(waveform)  # [1, n_features, time_frames]
            # print(features)
        # ---------------------------
        # Convert to ViT input
        # ---------------------------
        feat_tensor = features.unsqueeze(1)  # [1, 1, n_features, time_frames]
        feat_resized = F.interpolate(feat_tensor, size=(64), mode="bilinear")
        feat_resized = feat_resized.repeat(1, 1, 1, 1)  # [1, 3, 64, 128]
        feat_resized = feat_resized.squeeze(0).cpu()
        X_list.append(feat_resized)
        y_list.append(emotion_map[emotion])

    print("Total samples loaded:", len(X_list), len(y_list))
    # ===========================
    # Final dataset tensors
    # ===========================
    X = torch.stack(X_list)  # [num_samples, 3, 64, 128]
    y = torch.tensor(y_list)  # [num_samples]

    print("Final dataset shapes:")
    print("  X:", X.shape)
    print("  y:", y.shape)

    # os.chdir(old_path)

    return TensorDataset(X, y)
