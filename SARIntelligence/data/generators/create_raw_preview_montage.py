"""
SARIntelligence/data/generators/create_raw_preview_montage.py
=============================================================
Renders high-contrast optical-style RGB preview images of raw GeoTIFFs
to visually inspect ocean wakes, ground vehicle returns, and temporal escalation.
"""

import os
import cv2
import numpy as np
import rasterio

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "previews"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def render_preview(tif_path: str, out_png: str):
    with rasterio.open(tif_path) as src:
        if src.count >= 2:
            # Dual-pol composite (R=VV, G=VH, B=VV/VH)
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)
            ratio = vv / (vh + 1e-4)

            def norm(a):
                p2, p98 = np.percentile(a, (2, 99))
                return np.clip((a - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

            r = norm(vv)
            g = norm(vh)
            b = norm(ratio)
            rgb = cv2.merge([b, g, r])  # OpenCV BGR
        else:
            # Single band
            band = src.read(1).astype(np.float32)
            p2, p98 = np.percentile(band, (2, 99))
            gray = np.clip((band - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enh = clahe.apply(gray)
            rgb = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(out_png, rgb)
    return out_png


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw"))
    samples = [
        ("ocean/ocean_active_fleet_01.tif", "preview_ocean_fleet_01.png"),
        ("ocean/ocean_calm_waters_06.tif", "preview_ocean_calm_06.png"),
        ("ground/ground_armor_t72_01.tif", "preview_ground_armor_01.png"),
        ("ground/ground_airbase_alert_17.tif", "preview_ground_airbase_17.png"),
        ("multi_temporal/temporal_pair_01_T1_baseline.tif", "preview_temporal_01_T1.png"),
        ("multi_temporal/temporal_pair_01_T2_current.tif", "preview_temporal_01_T2.png"),
    ]

    for rel_tif, out_name in samples:
        t_path = os.path.join(base_dir, rel_tif)
        if os.path.exists(t_path):
            o_path = os.path.join(OUTPUT_DIR, out_name)
            render_preview(t_path, o_path)
            print(f"Generated preview: {out_name}")
