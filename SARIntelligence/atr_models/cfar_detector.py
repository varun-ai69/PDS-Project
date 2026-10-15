"""
SARIntelligence/atr_models/cfar_detector.py
===========================================
Two-Parameter Cell-Averaging Constant False Alarm Rate (CA-CFAR) Detector
for radar point target and cluster detection in SAR imagery.
"""

import cv2
import numpy as np


class CACFARDetector:
    def __init__(
        self,
        guard_cells: int = 4,
        training_cells: int = 12,
        k_factor: float = 3.8,
        min_area: int = 25,
        max_area: int = 4000,
        max_targets: int = 20
    ):
        self.guard = guard_cells
        self.training = training_cells
        self.k_factor = k_factor
        self.min_area = min_area
        self.max_area = max_area
        self.max_targets = max_targets

    def detect(self, img_gray: np.ndarray) -> list:
        """
        Executes CFAR detection on a grayscale radar raster.
        Suppresses diffuse clutter and isolates prominent metallic vehicles/structures.
        Returns list of bounding boxes: [(x, y, w, h), ...]
        """
        h, w = img_gray.shape
        img_f = img_gray.astype(np.float32)

        # High-confidence CFAR threshold against background speckle
        p98_8 = float(np.percentile(img_f, 98.8))
        mean_val = float(np.mean(img_f))
        std_val = float(np.std(img_f))
        cfar_thresh = max(p98_8, mean_val + self.k_factor * std_val)

        binary = (img_f >= cfar_thresh).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                
                # Check aspect ratio (reject extreme thin lines / sensor streaks)
                aspect = max(bw, bh) / (min(bw, bh) + 1e-3)
                if aspect > 5.5:
                    continue

                crop = img_f[y:y+bh, x:x+bw]
                peak_rcs = float(np.max(crop)) if crop.size > 0 else 0.0

                candidates.append({
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "centroid": [float(cx), float(cy)],
                    "area_pixels": int(area),
                    "peak_rcs": peak_rcs
                })

        # Sort by peak RCS intensity descending and keep top targets
        candidates.sort(key=lambda d: d["peak_rcs"], reverse=True)
        return candidates[:self.max_targets]
