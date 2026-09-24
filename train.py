"""
Training pipeline for the flower classifier.

Reproduces model/flower_cnn.pth. Dataset layout:
    data/flowers/<class_name>/*.jpg

Techniques demonstrated (each is an interview talking point):
    * stratified train/val split
    * data augmentation on the TRAIN split only (leakage control)
    * compact VGG-style CNN: conv-conv-pool blocks + BatchNorm
    * global average pooling (fewer params than a big FC layer)
    * dropout for regularization
    * Adam optimizer + cosine LR schedule
    * early stopping: keep the BEST validation checkpoint, not the last
    * artifact contract: weights + config.json written together, so the API
      can reproduce training-time preprocessing exactly

Usage:
    python train.py --data data/flowers --epochs 40 --out model/
"""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Reuse the exact architecture + preprocessing the API serves.
from app.model import FlowerCNN, build_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/flowers")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out", default="model")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(data_dir, min_size=180):
    """Scan data_dir/<class>/ and keep only decodable, non-tiny images."""
    paths, labels, classes = [], [], []
    for i, cls in enumerate(sorted(os.listdir(data_dir))):
        folder = Path(data_dir) / cls
        if not folder.is_dir():
            continue
        classes.append(cls)
        for f in sorted(folder.iterdir()):
            try:
                with Image.open(f) as im:
                    im.verify()
                if min(im.size) >= min_size:
                    paths.append(str(f))
                    labels.append(i)
            except Exception:
                pass  # corrupt / unsupported file -> skip
    return paths, labels, classes


class FlowerDS(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.tf = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img, mean, std = args.img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    paths, labels, classes = load_dataset(args.data)
    assert len(paths) >= 10, f"Too few images found in {args.data}"

    # ---- stratified split (keeps class ratios in both sets) ----
    idx = np.arange(len(paths))
    tr, val = train_test_split(idx, test_size=0.2, random_state=args.seed,
                               stratify=labels)
    tr_paths = [paths[i] for i in tr];  tr_labels = [labels[i] for i in tr]
    va_paths = [paths[i] for i in val]; va_labels = [labels[i] for i in val]

    # ---- augmentation: train ONLY. Validation must reflect real photos. ----
    train_tf = T.Compose([
        T.RandomResizedCrop(img, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    val_tf = build_transform(img, mean, std)  # deterministic, shared with API

    train_dl = DataLoader(FlowerDS(tr_paths, tr_labels, train_tf),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(FlowerDS(va_paths, va_labels, val_tf),
                        batch_size=args.batch)

    # ---- model / optimizer / schedule ----
    model = FlowerCNN(n_classes=len(classes)).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    # ---- training loop with early stopping ----
    best_acc, best_state, wait = 0.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += len(yb)
        acc = correct / total
        print(f"epoch {epoch + 1:3d}/{args.epochs}  val_acc={acc:.3f}")
        if acc >= best_acc:                     # keep best checkpoint
            best_acc, best_state, wait = acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)

    # ---- final report ----
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            y_true += yb.tolist()
            y_pred += model(xb).argmax(1).tolist()
    print(classification_report(y_true, y_pred, target_names=classes,
                                zero_division=0))
    print("confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # ---- save artifacts (weights + config = the train/serve contract) ----
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), Path(args.out) / "flower_cnn.pth")
    with open(Path(args.out) / "config.json", "w") as f:
        json.dump({"classes": classes, "img_size": img,
                   "mean": mean, "std": std}, f, indent=2)
    print(f"saved -> {args.out}/flower_cnn.pth, config.json | best val acc={best_acc:.3f}")


if __name__ == "__main__":
    main()
