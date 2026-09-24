"""
Model definition + inference wrapper.

FlowerCNN      – the exact architecture used in training (must stay in sync
                 with train.py; the saved file only contains weights, so this
                 class IS the model when loading state_dict).
FlowerPredictor – loads weights + config once and exposes predict(pil_image).
                 Keeping preprocessing here (not in the route) guarantees the
                 API applies the SAME transform that validation used in
                 training -> no train/serve skew.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Architecture: VGG-style blocks (conv -> BN -> ReLU) x2, then MaxPool.
# BatchNorm speeds up training and adds mild regularization.
# Global average pooling (AdaptiveAvgPool2d(1)) collapses each feature map to
# a single value -> far fewer parameters than flattening into a big FC layer.
# ---------------------------------------------------------------------------
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FlowerCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32),      # 64 -> 32
            _block(32, 64),     # 32 -> 16
            _block(64, 128),    # 16 -> 8
            _block(128, 256),   # 8  -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),    # regularization for a small dataset
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_transform(img_size, mean, std):
    """Deterministic inference transform. Deliberately NOT augmented:
    preprocessing at serve time must match validation preprocessing."""
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class FlowerPredictor:
    """Singleton-style wrapper: construct once (at app startup), predict many."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)

        self.classes = cfg["classes"]
        self.transform = build_transform(cfg["img_size"], cfg["mean"], cfg["std"])

        # weights_only=True is the safe way to load state dicts
        state = torch.load(model_dir / "flower_cnn.pth", map_location="cpu",
                           weights_only=True)
        self.model = FlowerCNN(n_classes=len(self.classes))
        self.model.load_state_dict(state)
        self.model.eval()          # disable dropout / batchnorm updates

    @torch.no_grad()
    def predict(self, pil_image):
        """pil_image: PIL.Image (RGB). Returns the API response dict."""
        x = self.transform(pil_image.convert("RGB")).unsqueeze(0)  # add batch dim
        probs = torch.softmax(self.model(x), dim=1).squeeze(0)
        top = int(probs.argmax())
        return {
            "class_name": self.classes[top],
            "confidence": round(float(probs[top]), 4),
            "probabilities": {c: round(float(p), 4)
                              for c, p in zip(self.classes, probs)},
        }
