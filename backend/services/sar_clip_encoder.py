"""
backend/services/sar_clip_encoder.py
====================================
SARCLIP ViT-L-14 Foundation Model Encoder Service for Synthetic Aperture Radar.
================================================================================
Provides a singleton `SARCLIPEncoder` using BiliSakura/SARCLIP-ViT-L-14:
- Pre-trained directly on Synthetic Aperture Radar (SAR) imagery backscatter
- 24-layer Vision Transformer Large (ViT-L-14) with 14x14 fine spatial patch resolution
- 768-dimensional L2-normalized multi-modal vision-language vector space
"""

import io
import logging
import os
from typing import List, Union, Optional
import numpy as np
from PIL import Image
import torch
import open_clip
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer
from safetensors.torch import load_file

logger = logging.getLogger("SARCLIPEncoder")

MODEL_REPO_ID = "BiliSakura/SARCLIP-ViT-L-14"
EMBEDDING_DIM = 768


class SARCLIPEncoder:
    """
    Service wrapper around BiliSakura/SARCLIP-ViT-L-14 multimodal foundation model.
    Encodes radar prompts and SAR GeoTIFF/PNG crops into 768-D normalized vectors.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        # 1. Device Selection
        if device is None:
            env_dev = (os.getenv("DEVICE") or "auto").strip().lower()
            if env_dev in ("cuda", "gpu"):
                try:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    self.device = "cpu"
            elif env_dev == "cpu":
                self.device = "cpu"
            else:
                try:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    self.device = "cpu"
        else:
            self.device = "cuda" if str(device).lower() in ("cuda", "gpu") and torch.cuda.is_available() else "cpu"

        # 2. Model Location Discovery
        if checkpoint_dir is None:
            checkpoint_dir = os.getenv("SARCLIP_CHECKPOINT_PATH")
            if not checkpoint_dir or not os.path.exists(checkpoint_dir):
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                candidates = [
                    os.path.join(base_dir, "models", "sar_retrieval", "SARCLIP-ViT-L-14"),
                    os.path.join("/app", "models", "sar_retrieval", "SARCLIP-ViT-L-14"),
                ]
                checkpoint_dir = None
                for c in candidates:
                    if os.path.exists(c) and os.path.exists(os.path.join(c, "model.safetensors")):
                        checkpoint_dir = c
                        break

        if not checkpoint_dir or not os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
            raise FileNotFoundError(
                f"SARCLIP model.safetensors not found in: {checkpoint_dir}. Please run download first."
            )

        logger.info("Initializing SARCLIP ViT-L-14 from '%s' on device '%s'...", checkpoint_dir, self.device)
        self.checkpoint_dir = checkpoint_dir
        safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")

        # 3. Load Checkpoint Tensors
        weights = load_file(safetensors_path)
        vis_weights = {k.replace("vision_model.", ""): v for k, v in weights.items() if k.startswith("vision_model.")}
        txt_weights = {k.replace("text_model.", ""): v for k, v in weights.items() if k.startswith("text_model.")}
        self.text_projection = weights.get("text_projection")
        if self.text_projection is not None:
            self.text_projection = self.text_projection.to(self.device)

        # 4. Load Visual Transformer (ViT-L-14)
        m, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L-14")
        m.visual.load_state_dict(vis_weights, strict=True)
        self.visual_model = m.visual.to(self.device).eval()

        # 5. Load Text Transformer & Tokenizer
        text_cfg = CLIPTextConfig.from_pretrained(checkpoint_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint_dir)
        self.text_model = CLIPTextModel(text_cfg)
        self.text_model.load_state_dict(txt_weights, strict=True)
        self.text_model = self.text_model.to(self.device).eval()

        logger.info("SARCLIP ViT-L-14 encoder initialized successfully (dim=%d)!", EMBEDDING_DIM)

    # -------------------------------------------------------------------------
    # TEXT ENCODING (768-D)
    # -------------------------------------------------------------------------
    def encode_text(
        self,
        text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode natural language intelligence prompt(s) into 768-D L2-normalized vector(s).
        """
        single_input = isinstance(text, str)
        text_list = [text] if single_input else text

        # Tactical prompt template enhancement for radar domain
        enhanced_prompts = []
        for q in text_list:
            q_clean = q.strip()
            if any(k in q_clean.lower() for k in ["sar", "radar", "satellite", "aerial"]):
                enhanced_prompts.append(q_clean)
            else:
                enhanced_prompts.append(f"a synthetic aperture radar image showing {q_clean}")

        inputs = self.tokenizer(
            enhanced_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            pooler_output = outputs.pooler_output
            if self.text_projection is not None:
                text_features = pooler_output @ self.text_projection
            else:
                text_features = pooler_output

            # L2 normalize vectors
            norm = text_features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            normalized_features = (text_features / norm).cpu().numpy().tolist()

        return normalized_features[0] if single_input else normalized_features

    # -------------------------------------------------------------------------
    # IMAGE ENCODING (768-D)
    # -------------------------------------------------------------------------
    def encode_image(
        self,
        image_input: Union[np.ndarray, Image.Image, str, bytes]
    ) -> List[float]:
        """
        Encode SAR raster / patch into 768-D L2-normalized vector.
        Accepts: PIL Image, NumPy array (H, W) or (H, W, 3), file path, or raw bytes.
        """
        pil_img = self._load_as_pil(image_input)
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.visual_model(img_tensor)
            # L2 normalize
            norm = image_features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            normalized_vec = (image_features / norm).cpu().numpy()[0].tolist()

        return normalized_vec

    def compute_similarity(self, text_prompt: str, image_input: Union[np.ndarray, Image.Image, str, bytes]) -> float:
        """Computes cosine similarity between query prompt and SAR image."""
        t_vec = np.array(self.encode_text(text_prompt), dtype=np.float32)
        i_vec = np.array(self.encode_image(image_input), dtype=np.float32)
        return float(np.dot(t_vec, i_vec))

    def _load_as_pil(self, image_input: Union[np.ndarray, Image.Image, str, bytes]) -> Image.Image:
        """Helper to safely standardize any image format into RGB PIL.Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB") if image_input.mode != "RGB" else image_input

        if isinstance(image_input, (str, bytes)):
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img = Image.open(image_input)
            else:
                img = Image.open(io.BytesIO(image_input))
            return img.convert("RGB")

        if isinstance(image_input, np.ndarray):
            arr = image_input
            # Single band SAR (dB or linear amplitude)
            if arr.ndim == 2:
                if arr.dtype != np.uint8:
                    p2, p98 = np.percentile(arr, (2, 98))
                    arr = np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3:
                if arr.shape[0] in (1, 2, 3) and arr.shape[0] < arr.shape[1]:
                    # CHW -> HWC
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                elif arr.shape[-1] == 2:
                    # e.g. VV and VH: compute composite
                    ratio = np.clip(arr[:, :, 0] / (arr[:, :, 1] + 1e-6), 0, 255)
                    arr = np.stack([arr[:, :, 0], arr[:, :, 1], ratio], axis=-1)
                if arr.dtype != np.uint8:
                    p2, p98 = np.percentile(arr, (2, 98))
                    arr = np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        raise ValueError(f"Unsupported image input type: {type(image_input)}")


# -----------------------------------------------------------------------------
# SINGLETON ACCESSOR
# -----------------------------------------------------------------------------
_global_sar_encoder: Optional[SARCLIPEncoder] = None


def get_sar_encoder() -> SARCLIPEncoder:
    """Obtain shared singleton instance of SARCLIPEncoder."""
    global _global_sar_encoder
    if _global_sar_encoder is None:
        _global_sar_encoder = SARCLIPEncoder()
    return _global_sar_encoder
