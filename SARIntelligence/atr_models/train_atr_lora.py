"""
SARIntelligence/atr_models/train_atr_lora.py
============================================
Parameter-Efficient Fine-Tuning (PEFT / LoRA) Adaptation
for Deep SAR Automatic Target Recognition (ATR) Heads.
Applies Low-Rank Adaptation (r=16, alpha=32) to linear projection layers.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def create_lora_mstar_model(num_classes: int = 10, r: int = 16, lora_alpha: int = 32):
    """
    Wraps ResNet18 ATR backbone with LoRA trainable low-rank adapters.
    Freezes base feature extractor while keeping LoRA matrices trainable.
    """
    base_model = models.resnet18(weights=None)
    base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)

    if PEFT_AVAILABLE:
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["fc"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(base_model, config)
        print("PEFT LoRA model created successfully!")
        model.print_trainable_parameters()
        return model
    else:
        # Standard head fine-tuning fallback if peft not installed
        for param in base_model.parameters():
            param.requires_grad = False
        for param in base_model.fc.parameters():
            param.requires_grad = True
        print("Standard Linear Head Fine-Tuning model created.")
        return base_model


if __name__ == "__main__":
    m = create_lora_mstar_model()
