import os
import torch
import pickle
import torchaudio
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset

old_path = os.getcwd()
print(old_path)
new_path = rf"{old_path}\FrontEnd"
os.chdir(new_path)

from models.classifier import Classifier

def loadpretrain():
    results_dir = r"C:\Users\Asus TUF Gaming A15\Documents\GitHub\RM\results"
    hparams_path = os.path.join(results_dir, r"C:\Coding\efficientnet-b0_default_leaf_bs1x256_adam_warmupcosine_wd_1e-4_rs8882_legacycomplex\hparams.pickle")
    ckpt_path = os.path.join(results_dir, "ckpts", r"C:\Coding\efficientnet-b0_default_leaf_bs1x256_adam_warmupcosine_wd_1e-4_rs8882_legacycomplex\ckpts\epoch=100_tr_loss=0.067792_tr_acc=0.980434_val_acc=0.954013.pth")
    checkpoint = torch.load(ckpt_path)
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    model = Classifier(hparams.cfg)
    print(model.load_state_dict(checkpoint['model_state_dict']))

    return model,hparams

def preprocess(mapping,path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model,hparams = loadpretrain()
    frontend = model.features
    """
    ---------
    """
    # print(path)
    contents = os.listdir(path)
    # print(contents)
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

        # ---------------------------
        # Load and preprocess audio
        # ---------------------------
        waveform, sample_rate = torchaudio.load(file_path)  # [channels, time]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        expected_sr = hparams.cfg['audio_config']['sample_rate']
        if sample_rate != expected_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sr)
            waveform = resampler(waveform)

        waveform = waveform.unsqueeze(0).float().cpu()  # [1, 1, time]

        # ---------------------------
        # Forward pass
        # ---------------------------
        with torch.no_grad():
            features = frontend(waveform)  # [1, n_features, time_frames]

        # ---------------------------
        # Convert to ViT input
        # ---------------------------
        feat_tensor = features.unsqueeze(1)                     # [1, 1, n_features, time_frames]
        feat_resized = F.interpolate(feat_tensor, size=(64, 128), mode="bilinear")  
        feat_resized = feat_resized.repeat(1, 1, 1, 1)          # [1, 3, 256, 256]
        feat_resized = feat_resized.squeeze(0).cpu()
        X_list.append(feat_resized)
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
    os.chdir(old_path)
    return TensorDataset(X, y)