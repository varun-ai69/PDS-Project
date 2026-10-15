"""
SARIntelligence/atr_models/ground_atr_classifier.py
===================================================
10-Class Deep Convolutional Automatic Target Recognition (ATR)
trained on Sandia/AFRL MSTAR radar test dataset:
Classes: 2S1, BMP2, BRDM2, BTR60, BTR70, D7, T62, T72, ZIL131, ZSU_23_4
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models", "detection", "mstar_resnet18_10class.pt")

MSTAR_CLASSES = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU_23_4']
MSTAR_DISPLAY_NAMES = {
    'T72': 'T-72 Main Battle Tank',
    'T62': 'T-62 Main Battle Tank',
    'BMP2': 'BMP-2 Infantry Fighting Vehicle',
    'BRDM2': 'BRDM-2 Reconnaissance Scout',
    'BTR70': 'BTR-70 Armored Personnel Carrier',
    'BTR60': 'BTR-60 Armored Personnel Carrier',
    '2S1': '2S1 Gvozdika Self-Propelled Howitzer',
    'ZSU_23_4': 'ZSU-23-4 Anti-Aircraft Gun System',
    'ZIL131': 'ZIL-131 Logistics Convoy Truck',
    'D7': 'D7 Armored Combat Bulldozer'
}


class GroundATRClassifier:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(MSTAR_CLASSES))

        # Check weights path priority
        weights_paths = [
            os.path.join(ROOT_DIR, "SARIntelligence", "weights", "ground", "mstar_resnet18_10class.pt"),
            MODEL_WEIGHTS_PATH,
        ]
        loaded = False
        for p in weights_paths:
            if os.path.exists(p):
                try:
                    ckpt = torch.load(p, map_location=self.device)
                    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
                    self.model.load_state_dict(state_dict, strict=True)
                    loaded = True
                    break
                except Exception:
                    pass

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.18], std=[0.22])
        ])

    def predict_chip(self, chip_gray: np.ndarray) -> tuple:
        """Classifies a single cropped radar target chip."""
        if chip_gray.ndim == 3:
            chip_gray = cv2.cvtColor(chip_gray, cv2.COLOR_BGR2GRAY)

        tensor = self.transform(chip_gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            cls_idx = int(np.argmax(probs))
            conf = float(probs[cls_idx])

        pred_class = MSTAR_CLASSES[cls_idx]
        display_name = MSTAR_DISPLAY_NAMES.get(pred_class, pred_class)
        return pred_class, display_name, conf
