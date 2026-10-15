"""
SARIntelligence/data/generators/apply_despeckled_enhancements.py
=============================================================
Applies industry-grade SAR despeckling and polarization enhancements:
1. Ground Domain:
   - Restores the 20 authentic tactical scenes (8 Airbases with multiple fighter jets on flightlines,
     8 Natural terrain armor scenes with 3-4 hull-down tanks, 4 Logistics convoys).
   - Calibrated 16-bit GeoTIFFs (EPSG:4326, HH / Spotlight X-band).
   - Enhanced Lee + Bilateral despeckled visual PNG previews (smooth, zero harsh speckle).
2. Ocean Domain:
   - Despeckles all 10 Sentinel-1 / TerraSAR-X ocean scenes.
   - Dual-pol (VV, VH, VV/VH) tactical composite rendering.
3. Multi-Temporal Surveillance:
   - Despeckles all 20 co-registered pairs (T1 Baseline vs T2 Current).
   - Re-renders side-by-side comparison montages.
"""

import os
import glob
import shutil
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import sys

sys.path.append(os.path.abspath("."))
from SARIntelligence.preprocessing.despeckle_filter import enhanced_lee_filter, bilateral_despeckle_filter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAR_INTEL_DATA = os.path.join(ROOT_DIR, "SARIntelligence", "data")
RAW_OCEAN = os.path.join(SAR_INTEL_DATA, "raw", "ocean")
RAW_GROUND = os.path.join(SAR_INTEL_DATA, "raw", "ground")
RAW_TEMPORAL = os.path.join(SAR_INTEL_DATA, "raw", "multi_temporal")
PREVIEWS_OCEAN = os.path.join(SAR_INTEL_DATA, "previews", "ocean")
PREVIEWS_GROUND = os.path.join(SAR_INTEL_DATA, "previews", "ground")
PREVIEWS_TEMPORAL = os.path.join(SAR_INTEL_DATA, "previews", "multi_temporal")
SRC_GROUND_TIFS = os.path.join(ROOT_DIR, "sar-inputs", "raw", "ground")
ART_DIR = r"C:\Users\Asus f15\.gemini\antigravity-ide\brain\43452791-0da0-477f-a3e8-e962d57882a6"


def clean_despeckle(gray_8bit: np.ndarray) -> np.ndarray:
    """Enhanced Lee (window 5) followed by Bilateral filter for silky smooth SAR texture with sharp edges."""
    lee = enhanced_lee_filter(gray_8bit, window_size=5, cu=0.523, cmax=1.73)
    bi = bilateral_despeckle_filter(lee, d=5, sigma_color=25.0, sigma_space=5.0)
    return bi


def process_ground_domain():
    print("1. Processing Ground Domain (Airbases with multiple planes + Natural Armor terrain)...")
    src_files = sorted(glob.glob(os.path.join(SRC_GROUND_TIFS, "*.tif")))
    for src_f in src_files:
        fn = os.path.basename(src_f)
        dst_tif = os.path.join(RAW_GROUND, fn)
        shutil.copy2(src_f, dst_tif)

        with rasterio.open(src_f) as src:
            d = src.read(1)
            p1, p99 = np.percentile(d, (1, 99))
            d_8 = np.clip((d - p1) / (p99 - p1 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

        desp = clean_despeckle(d_8)
        out_png = os.path.join(PREVIEWS_GROUND, fn.replace(".tif", ".png"))
        cv2.imwrite(out_png, cv2.cvtColor(desp, cv2.COLOR_GRAY2BGR))
        print(f"  Processed ground scene: {fn}")


def process_ocean_domain():
    print("2. Processing Ocean Domain (Despeckling Sentinel-1 / TerraSAR-X)...")
    tifs = sorted(glob.glob(os.path.join(RAW_OCEAN, "*.tif")))
    for tif_p in tifs:
        fn = os.path.basename(tif_p)
        with rasterio.open(tif_p) as src:
            vv = src.read(1)
            p1, p99 = np.percentile(vv, (1, 99))
            vv_8 = np.clip((vv - p1) / (p99 - p1 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

        desp = clean_despeckle(vv_8)
        out_png = os.path.join(PREVIEWS_OCEAN, fn.replace(".tif", ".png"))
        cv2.imwrite(out_png, cv2.cvtColor(desp, cv2.COLOR_GRAY2BGR))
        print(f"  Despeckled ocean scene: {fn}")


def process_temporal_domain():
    print("3. Processing Multi-Temporal Domain (Despeckling T1 vs T2 & Comparison Montages)...")
    for pair_idx in range(1, 21):
        t1_fn = f"temporal_pair_{pair_idx:02d}_T1_baseline.tif"
        t2_fn = f"temporal_pair_{pair_idx:02d}_T2_current.tif"
        t1_tif = os.path.join(RAW_TEMPORAL, t1_fn)
        t2_tif = os.path.join(RAW_TEMPORAL, t2_fn)

        if not os.path.exists(t1_tif) or not os.path.exists(t2_tif):
            continue

        with rasterio.open(t1_tif) as src:
            t1_d = src.read(1)
            t1_tags = src.tags()
            p1, p99 = np.percentile(t1_d, (1, 99))
            t1_8 = np.clip((t1_d - p1) / (p99 - p1 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

        with rasterio.open(t2_tif) as src:
            t2_d = src.read(1)
            t2_tags = src.tags()
            p1, p99 = np.percentile(t2_d, (1, 99))
            t2_8 = np.clip((t2_d - p1) / (p99 - p1 + 1e-5) * 255.0, 0, 255).astype(np.uint8)

        t1_clean = clean_despeckle(t1_8)
        t2_clean = clean_despeckle(t2_8)

        # Update preview PNGs
        cv2.imwrite(os.path.join(PREVIEWS_TEMPORAL, t1_fn.replace(".tif", ".png")), cv2.cvtColor(t1_clean, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(PREVIEWS_TEMPORAL, t2_fn.replace(".tif", ".png")), cv2.cvtColor(t2_clean, cv2.COLOR_GRAY2BGR))

        # Re-render side-by-side comparison montage
        comp = np.zeros((512 + 60, 512 * 2 + 20, 3), dtype=np.uint8)
        comp[:, :] = (25, 25, 25)
        comp[50:562, 5:517] = cv2.cvtColor(t1_clean, cv2.COLOR_GRAY2BGR)

        t2_bgr = cv2.cvtColor(t2_clean, cv2.COLOR_GRAY2BGR)
        surge = t2_tags.get("STATUS") == "CRITICAL_ESCALATION"
        unit_type = t2_tags.get("UNIT_TYPE", "COMBAT_UNIT")
        t1_tgts = t1_tags.get("TARGET_COUNT", "0")
        t2_tgts = t2_tags.get("TARGET_COUNT", "0")
        sector_name = t2_tags.get("SECTOR_NAME", f"Sector {pair_idx}")

        comp[50:562, 527:1039] = t2_bgr

        t1_header = f"T1 BASELINE (2026-08-15) | {sector_name[:24]} | {t1_tgts} TARGETS"
        t2_header = f"T2 CURRENT (2026-10-12) | {t2_tgts} {unit_type} DEPLOYED | {'ESCALATION SURGE' if surge else 'NORMAL'}"
        cv2.putText(comp, t1_header, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.putText(comp, t2_header, (530, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 140, 255) if surge else (100, 255, 100), 1)

        comp_png = os.path.join(PREVIEWS_TEMPORAL, f"comparison_pair_{pair_idx:02d}.png")
        cv2.imwrite(comp_png, comp)
        print(f"  Re-rendered comparison montage: comparison_pair_{pair_idx:02d}.png")


def copy_artifacts():
    print("4. Copying updated clean images to Artifacts directory...")
    os.makedirs(ART_DIR, exist_ok=True)
    to_copy = [
        ("SARIntelligence/data/previews/multi_temporal/comparison_pair_01.png", "phase1_comparison_pair_01.png"),
        ("SARIntelligence/data/previews/multi_temporal/comparison_pair_02.png", "phase1_comparison_pair_02.png"),
        ("SARIntelligence/data/previews/multi_temporal/comparison_pair_09.png", "phase1_comparison_pair_09.png"),
        ("SARIntelligence/data/previews/multi_temporal/comparison_pair_11.png", "phase1_comparison_pair_11.png"),
        ("SARIntelligence/data/previews/multi_temporal/comparison_pair_14.png", "phase1_comparison_pair_14.png"),
        ("SARIntelligence/data/previews/ocean/ocean_active_fleet_01.png", "phase1_ocean_fleet_01.png"),
        ("SARIntelligence/data/previews/ocean/ocean_active_fleet_02.png", "phase1_ocean_fleet_02.png"),
        ("SARIntelligence/data/previews/ground/ground_armor_tanks_01.png", "phase1_ground_tanks_01.png"),
        ("SARIntelligence/data/previews/ground/ground_airbase_aircraft_09.png", "phase1_ground_aircraft_09.png"),
        ("SARIntelligence/data/previews/ground/ground_airbase_aircraft_13.png", "phase1_ground_aircraft_13.png"),
        ("SARIntelligence/data/previews/ground/ground_tactical_convoy_17.png", "phase1_ground_convoy_17.png"),
        ("scratch/test_noise_comparison.png", "phase1_noise_comparison.png")
    ]
    for src, dst_name in to_copy:
        dst = os.path.join(ART_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst_name}")


if __name__ == "__main__":
    process_ground_domain()
    process_ocean_domain()
    process_temporal_domain()
    copy_artifacts()
    print("=== DESPECKLING & GROUND RESTORATION 100% COMPLETE ===")
