"""
SARIntelligence/preprocessing/rgb_compositor.py
================================================
Scientific Satellite SAR RGB Compositing:
- Dual-Pol (VV + VH) Natural Polarimetric Composition:
  * Water renders in deep midnight pitch black
  * Battleship hulls & superstructures shine brilliant white / specular amber
- Single-Pol (HH / VV) Natural Satellite Grayscale:
  * Sharp tactical contrast preserving all authentic radar speckle
- Multi-Temporal Co-Registered Change Composite:
  * Static terrain: Natural balanced grayscale (R = G = B)
  * Tactical escalations (new deployments): Vivid highlight (R=T2, G=T1, B=T1)
"""

import cv2
import numpy as np


def stretch_percentiles(channel: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Clips and normalizes an array based on percentile thresholds to [0, 255] uint8."""
    ch_f = channel.astype(np.float32)
    vmin, vmax = np.percentile(ch_f, (p_low, p_high))
    if vmax <= vmin:
        vmax = vmin + 1e-4
    stretched = np.clip((ch_f - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return stretched.astype(np.uint8)


def build_dualpol_rgb_composite(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Maritime Fleet Polarimetric RGB Composite:
    - Water stays pitch dark / deep black (no artificial noise boost)
    - Warship hulls and radar superstructures glow brilliant white
    - Preserves 100% crisp authentic radar resolution without blurring.
    """
    vv_norm = vv.copy()
    vh_norm = vh.copy()

    # Red: VV co-polarization (strong metallic surface bounce)
    r_ch = vv_norm.copy()
    # Green: Balanced mid-band (volume scatter + specular)
    g_ch = np.clip(0.6 * vv_norm.astype(np.float32) + 0.4 * vh_norm.astype(np.float32), 0, 255).astype(np.uint8)
    # Blue: Subtle midnight navy tint for water, glints on ships
    b_ch = np.clip(0.85 * vh_norm.astype(np.float32) + 0.25 * vv_norm.astype(np.float32), 0, 255).astype(np.uint8)

    # OpenCV BGR ordering
    return cv2.merge([b_ch, g_ch, r_ch])


def build_single_band_tactical_rgb(img: np.ndarray) -> np.ndarray:
    """
    Natural High-Definition Satellite Representation (Airbases, Armor, Ports):
    - Preserves 100% authentic crisp radar satellite dynamic range.
    - Zero blurring, zero fake watercolor smudges.
    """
    img_norm = stretch_percentiles(img, 0.5, 99.5)
    return cv2.merge([img_norm, img_norm, img_norm])


def build_temporal_change_rgb(
    t1_baseline: np.ndarray,
    t2_current: np.ndarray,
    diff_boost: float = 2.0
) -> np.ndarray:
    """
    Standard Multi-Temporal Change Detection RGB Composite:
    - Red Channel   = T2 (Current observation)
    - Green Channel = T1 (Baseline observation)
    - Blue Channel  = T1 (Baseline observation)
    
    Properties:
    - Unchanged terrain has T2 == T1 -> R == G == B -> 100% pure, natural, crisp grayscale!
    - Newly arrived targets (tanks, aircraft, warships) have T2 > T1 -> Glow vivid RED!
    - Departed assets have T1 > T2 -> Glow CYAN!
    """
    t1_norm = stretch_percentiles(t1_baseline, 0.5, 99.5)
    t2_norm = stretch_percentiles(t2_current, 0.5, 99.5)

    # In OpenCV BGR: B = T1, G = T1, R = T2
    return cv2.merge([t1_norm, t1_norm, t2_norm])
