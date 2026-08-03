import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch, random, sys
from torch import nn, optim
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from contextlib import contextmanager
import pandas as pd
import numpy as np

print("Setup Function...")


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass

    print(f"Seed set to {seed}")


set_seed(42)


@contextmanager
def my_chdir(des_path):
    old_path = os.getcwd()
    des_path = os.path.abspath(des_path)
    sys.path.insert(0, des_path)
    try:
        os.chdir(des_path)
        yield
    finally:
        os.chdir(old_path)
        if des_path in sys.path:
            sys.path.remove(des_path)


def Create_Loader(dataset, batch_size):

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(
    parameter, num_epochs, device, train_loader, val_loader, save_dir="./results"
):

    vit_model = parameter
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        vit_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = vit_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        vit_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = vit_model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(vit_model.state_dict(), f"{save_dir}/model_best.pth")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return vit_model


def test_model(model, test_loader, device, emotion_map, save_dir="./results"):

    vit_model = model.to(device)
    vit_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = vit_model(X_batch)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    # ---------------------------
    # Prepare data
    # ---------------------------
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    y_true = all_labels.numpy()
    y_pred = all_preds.numpy()

    unique_labels = sorted(torch.unique(all_labels).tolist())

    all_emotions = {v: k for k, v in emotion_map.items()}
    display_labels = [
        all_emotions[i] if i in all_emotions else f"Unknown({i})" for i in unique_labels
    ]

    # ---------------------------
    # Classification report
    # ---------------------------
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=display_labels,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(save_dir, "classification_report.csv"))

    # also save text version
    report_txt = classification_report(
        y_true, y_pred, labels=unique_labels, target_names=display_labels
    )

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    print("Classification Report saved")

    # ---------------------------
    # Confusion matrix
    # ---------------------------
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # save raw matrix
    np.savetxt(
        os.path.join(save_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d"
    )

    # plot & save figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    print("Confusion matrix saved")

    return report_df, cm


result_dir = "./results"
os.makedirs(result_dir, exist_ok=True)
save_dir = f"{result_dir}/Experiment{len(os.listdir(result_dir))}"
os.makedirs(save_dir, exist_ok=True)

"""
IEMCAP
Class distribution:
Class 0 neutral : 1768
Class 1 frustration : 2874
Class 2 sadness : 1253
Class 3 surprise : 125
Class 4 anger : 1284
Class 5 happiness : 708
Class 6 excited : 1883
Class 7 fear : 105
Class 8 disgust : 3
Class 9 other : 36

"""

# emotion_map = {
#     "neutral": 0,
#     "sadness": 1,
#     "anger": 2,
#     "happiness": 3,
#     "excitement": 4,
#     "excited": 4,
# }


"""
CREAMA D
Class distribution:
Class 0 neutral : 1087
Class 1 anger : 1271
Class 2 disgust : 1271
Class 3 fear : 1271  
Class 4 happy : 1271
Class 5 sad : 1271
"""


emotion_map = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "neutral": 3,
    "sad": 4,
    "happy": 5,
}

"""
Available

FrontEnd
|- Leaf.preprocess
|   |- from FrontEnd import Leaf
|- Mel.preprocess
|   |- from FrontEnd import Mel
|- Scatter.preprocess
    |- from FrontEnd import Scatter

Model
|- VitCnnGlobal.Vit
|   |- from Model import VitCnnGlobal
|- VitCnnLocal.Vit
|   |- from Model import VitCnnLocal
|- VitGlobal.Vit
|   |- from Model import VitGlobal
|- VitLocal.Vit
|   |- from Model import VitLocal

"""


"""
------------------ Replace Front End Here ------------------
"""
print("Setup preprocessor.")
from FrontEnd import Leaf

print("Preprocessing Datas")

new_path = os.path.join(os.getcwd(), "FrontEnd")

dataset_path = "../Dataset2"
processed_path = "./dataset_preprocessed/processed_dataset2_64_leaf.pt"

if os.path.exists(processed_path):
    print("Loading preprocessed dataset...")
    dataset = torch.load(processed_path, weights_only=False)
else:
    with my_chdir(new_path):
        dataset = Leaf.preprocess(mapping=emotion_map, path=dataset_path)
    print("Saving preprocessed dataset...")
    torch.save(dataset, processed_path)

"""
-------------------------------------------------------------
"""


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if (
            x.ndim == 4
            and x.shape[0] == 2
            and x.shape[1] == 1
            and x.shape[2] == 64
            and x.shape[3] == 64
        ):
            x = x.reshape(2, 64, 64).unsqueeze(1).repeat(2, 1, 1, 1).squeeze(1)
        elif x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 64 and x.shape[2] == 64:
            x = x.squeeze(0).unsqueeze(0).unsqueeze(0)
            x = torch.nn.functional.interpolate(
                x, size=(64, 64), mode="bilinear", align_corners=False
            ).squeeze(0)
            x = x.repeat(4, 1, 1)
        elif x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 64 and x.shape[2] == 64:
            x = x.repeat(4, 1, 1)
        return x, y

    def __len__(self):
        return len(self.subset)


print("Spliting Dataset")
train_loader, val_loader, test_loader = Create_Loader(dataset, batch_size=16)
train_loader = DataLoader(
    TransformDataset(train_loader.dataset), batch_size=16, shuffle=True
)
val_loader = DataLoader(
    TransformDataset(val_loader.dataset), batch_size=16, shuffle=False
)
test_loader = DataLoader(
    TransformDataset(test_loader.dataset), batch_size=16, shuffle=False
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
num_epochs = 50

"""
------------------ Replace Model Here ------------------
"""
print("Setup Model.")
from Model import VitCnnLocal

print("Setup Parameter for Model")
vit_model = VitCnnLocal.ViT(
    image_size=(64, 64),
    patch_size=(8, 8),
    num_classes=len(emotion_map),
    dim=256,
    depth=6,
    heads=8,
    # dim_head=64,
    cnn_channels=64,
    mlp_dim=1024,
    channels=4,
    dropout=0.1,
    emb_dropout=0.1,
).to(device)

# model.load_state_dict(torch.load("./result/model_best.pth"))
# model.to(device)
# model.eval()
"""
-------------------------------------------------------------
"""
print("Training Model.")
Model = train_model(
    parameter=vit_model,
    num_epochs=num_epochs,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    save_dir=save_dir,
)

print("Testing Model.")
test_model(
    model=Model,
    test_loader=test_loader,
    device=device,
    emotion_map=emotion_map,
    save_dir=save_dir,
)
