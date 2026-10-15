"""
SARIntelligence Preprocessing Module:
- DN Normalization
- Radiometric Calibration (Sigma-0 Linear & dB)
- Sub-Pixel Band & Multi-Temporal Co-Registration
- High-Fidelity Tactical Satellite RGB Compositing
"""

from SARIntelligence.preprocessing.radiometric_calibration import (
    normalize_dn,
    calibrate_digital_numbers_to_sigma0_db,
    calibrate_digital_numbers_to_sigma0_linear,
    sigma0_db_to_normalized_dn,
    radiometric_calibrate_and_normalize,
)

from SARIntelligence.preprocessing.band_registration import (
    register_bands,
    co_register_temporal_pair,
    calculate_subpixel_shift,
)

from SARIntelligence.preprocessing.rgb_compositor import (
    build_dualpol_rgb_composite,
    build_single_band_tactical_rgb,
    build_temporal_change_rgb,
    stretch_percentiles,
)

__all__ = [
    "normalize_dn",
    "calibrate_digital_numbers_to_sigma0_db",
    "calibrate_digital_numbers_to_sigma0_linear",
    "sigma0_db_to_normalized_dn",
    "radiometric_calibrate_and_normalize",
    "register_bands",
    "co_register_temporal_pair",
    "calculate_subpixel_shift",
    "build_dualpol_rgb_composite",
    "build_single_band_tactical_rgb",
    "build_temporal_change_rgb",
    "stretch_percentiles",
]
