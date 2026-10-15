"""
SARIntelligence/preprocessing/radiometric_calibration.py
========================================================
Scientific SAR Radiometric Calibration & DN Normalization:
- DN to physical Sigma-0 (linear & dB) power transformation
- Robust dynamic range percentile stretch preserving 100% authentic radar textures
- Zero blurring, zero artificial smudging
"""

import numpy as np


def calibrate_digital_numbers_to_sigma0_linear(
    dn: np.ndarray,
    cal_factor: float = 65535.0
) -> np.ndarray:
    """
    Standard SAR Radiometric Calibration (Linear Power):
    Sigma0_linear = (DN / cal_factor)^2
    """
    dn_f = dn.astype(np.float32)
    norm = np.clip(dn_f / float(cal_factor), 1e-6, 1.0)
    return norm ** 2


def calibrate_digital_numbers_to_sigma0_db(
    dn: np.ndarray,
    cal_factor: float = 65535.0,
    min_db: float = -35.0,
    max_db: float = 20.0
) -> np.ndarray:
    """
    Standard SAR Radiometric Calibration (Decibel Power):
    Sigma0_dB = 10 * log10(Linear_power + epsilon)
    """
    sigma0_linear = calibrate_digital_numbers_to_sigma0_linear(dn, cal_factor)
    sigma0_db = 10.0 * np.log10(np.clip(sigma0_linear, 1e-6, 1.0))
    return np.clip(sigma0_db, min_db, max_db)


def sigma0_db_to_normalized_dn(
    sigma0_db: np.ndarray,
    min_db: float = -35.0,
    max_db: float = 20.0
) -> np.ndarray:
    """
    Maps calibrated dB values back to normalized [0, 255] uint8 DN.
    """
    clipped = np.clip(sigma0_db, min_db, max_db)
    norm = (clipped - min_db) / (max_db - min_db) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def normalize_dn(
    data: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
    return_float: bool = False
) -> np.ndarray:
    """
    Standard dynamic range linear DN normalization:
    Clips extreme sensor spikes while preserving 100% natural radar speckle texture.
    """
    data_f = data.astype(np.float32)
    v_min, v_max = np.percentile(data_f, (p_low, p_high))
    if v_max <= v_min:
        v_max = v_min + 1e-4
    stretched = np.clip((data_f - v_min) / (v_max - v_min), 0.0, 1.0)
    if return_float:
        return stretched
    return (stretched * 255.0).astype(np.uint8)


def radiometric_calibrate_and_normalize(
    dn: np.ndarray,
    cal_factor: float = None,
    p_low: float = 0.5,
    p_high: float = 99.5
) -> dict:
    """
    Unified end-to-end radiometric calibration and DN normalization step.
    Automatically detects calibration factor based on data range if not supplied.
    """
    if cal_factor is None:
        if dn.max() > 255.0:
            cal_factor = 65535.0
        else:
            cal_factor = 255.0

    sigma0_lin = calibrate_digital_numbers_to_sigma0_linear(dn, cal_factor)
    sigma0_db = calibrate_digital_numbers_to_sigma0_db(dn, cal_factor)
    dn_norm = normalize_dn(dn, p_low=p_low, p_high=p_high)

    return {
        "sigma0_linear": sigma0_lin,
        "sigma0_db": sigma0_db,
        "dn_normalized": dn_norm,
        "cal_factor": cal_factor
    }
