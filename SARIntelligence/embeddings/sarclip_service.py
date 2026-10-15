"""
SARIntelligence/embeddings/sarclip_service.py
=============================================
Multimodal 768-Dimensional Vision & Language Encoder
using BiliSakura/SARCLIP-ViT-L-14 foundation model for SAR imagery.
"""

import os
import sys
import logging
from typing import List, Union, Optional
import numpy as np
from PIL import Image
import torch
import open_clip
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

logger = logging.getLogger("SARCLIPService")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "models", "sar_retrieval", "SARCLIP-ViT-L-14")
EMBEDDING_DIM = 768


class SARCLIPService:
    """
    Singleton service wrapper around BiliSakura/SARCLIP-ViT-L-14.
    Encodes radar prompts and SAR GeoTIFF/PNG rasters into 768-D normalized vectors.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        safetensors_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"SARCLIP checkpoint not found at: {safetensors_path}")

        logger.info("Loading SARCLIP ViT-L-14 on device: %s", self.device)
        state_dict = load_file(safetensors_path)

        # 1. Vision Model (ViT-L-14) via OpenCLIP
        self.vision_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained=None,
            device=self.device
        )
        vision_weights = {k.replace("visual.", ""): v for k, v in state_dict.items() if k.startswith("visual.")}
        self.vision_model.visual.load_state_dict(vision_weights, strict=False)
        self.vision_model.eval()

        # 2. Text Model via Hugging Face CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained(CHECKPOINT_DIR)
        self.text_model = CLIPTextModel.from_pretrained(CHECKPOINT_DIR)
        self.text_model.to(self.device)
        self.text_model.eval()

        # 3. Projection Layer for Text (to 768-D)
        self.text_projection = None
        if "text_projection" in state_dict:
            self.text_projection = state_dict["text_projection"].to(self.device).float()

        logger.info("SARCLIPService 768-D Vision-Language Engine initialized successfully!")

    def encode_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> List[float]:
        """Encodes an image (file path, PIL Image, or numpy array) into a 768-D L2-normalized vector."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = Image.fromarray(image_input)
        else:
            img = image_input

        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.vision_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().numpy().tolist()

    def encode_text(self, text: str) -> List[float]:
        """Encodes military SITREP or user natural language query into a 768-D L2-normalized vector."""
        inputs = self.tokenizer([text], padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            pooled = outputs.pooler_output  # (1, 768)
            if self.text_projection is not None:
                pooled = pooled @ self.text_projection
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return pooled.squeeze(0).cpu().numpy().tolist()


_INSTANCE = None


def get_sarclip_service() -> SARCLIPService:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SARCLIPService()
    return _INSTANCE
