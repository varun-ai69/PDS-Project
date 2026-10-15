"""
SARIntelligence/pipeline/military_activity_score.py
===================================================
Military Activity Score (MAS) & DEFCON Assessment Engine.
Standardized mathematical synthesis of target threat weights,
neural ATR confidence, Radar Cross Section (RCS in dBsm),
spatial formation clustering density, and strategic sector multipliers.
"""

import math
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("MilitaryActivityScore")

# Standard Threat Weight Matrix
TACTICAL_CLASS_WEIGHTS: Dict[str, float] = {
    # Capital Strategic Assets
    "AIRCRAFT_CARRIER": 25.0,
    "MISSILE_LAUNCHER_TEL": 25.0,
    
    # Heavy Strike Assets
    "MAIN_BATTLE_TANK": 15.0,
    "T72": 15.0,
    "T62": 14.0,
    "NAVAL_WARSHIP": 15.0,
    "DESTROYER_FRIGATE": 15.0,
    "FIGHTER_JET": 14.0,
    "AIRCRAFT": 14.0,
    
    # Tactical Air Defense & Heavy Artillery
    "ZSU_23_4": 12.0,
    "ARTILLERY_HOWITZER": 12.0,
    "2S1": 12.0,
    
    # Mechanized Infantry & Fast Attack
    "ARMORED_COMBAT_VEHICLE": 8.0,
    "BMP2": 8.0,
    "BTR70": 8.0,
    "BTR60": 7.0,
    "CORVETTE_PATROL": 8.0,
    "SUBMARINE_SURFACED": 10.0,
    "BRDM2": 6.0,
    "PATROL_INTERCEPTOR_BOAT": 6.0,
    
    # Logistics, Engineering & Auxiliaries
    "MILITARY_CONVOY_TRUCK": 4.0,
    "ZIL131": 4.0,
    "D7": 4.0,
    "MILITARY_COMMAND_POST": 10.0,
    "CARGO_VESSEL": 3.0,
    "TANKER": 4.0,
    "DEFAULT": 5.0
}

HIGH_THREAT_SECTORS = [
    "AKHNOOR", "CHAMB", "AMBALA", "SRINAGAR", "URI", "POONCH",
    "KARGIL", "ARABIAN_SEA", "WESTERN_FLEET", "MALACCA", "LINE_OF_CONTROL"
]


class MilitaryActivityScorer:
    """
    Computes rigorous defense Military Activity Score (MAS)
    and maps to DEFCON operational readiness levels.
    """

    def __init__(self, high_threat_sectors: List[str] = None):
        self.high_threat_sectors = high_threat_sectors or HIGH_THREAT_SECTORS

    def calculate_mas(
        self,
        detections: List[Dict[str, Any]],
        sector: str = "",
        domain: str = "GROUND",
        surge_delta: int = 0
    ) -> Dict[str, Any]:
        """
        Calculates MAS (0.0 to 100.0) and DEFCON readiness tier.
        """
        if not detections and surge_delta <= 0:
            return {
                "mas_score": 0.0,
                "defcon_level": 5,
                "defcon_tier": "DEFCON_5_NORMAL",
                "defcon_description": "Zero Military Activity / Calm Non-Contested Area",
                "threat_level": "NEGLIGIBLE",
                "target_count": 0,
                "density_multiplier": 1.0,
                "sector_multiplier": 1.0,
                "surge_component": 0.0
            }

        total_targets = len(detections)
        raw_weighted_sum = 0.0

        for det in detections:
            raw_class = str(det.get("class") or det.get("target_class") or det.get("model_class") or "DEFAULT").upper()
            weight = TACTICAL_CLASS_WEIGHTS.get(raw_class, TACTICAL_CLASS_WEIGHTS.get("DEFAULT", 5.0))
            conf = float(det.get("confidence", 0.75))
            rcs_db = float(det.get("rcs_dbsm", det.get("peak_rcs_db", 8.0)))
            
            # RCS scaling factor: higher radar reflectivity boosts significance
            rcs_factor = 1.0 + (max(0.0, rcs_db) / 20.0)
            target_score = weight * conf * rcs_factor
            raw_weighted_sum += target_score

        # Formation Density Multiplier (clustering of units indicates coordinated force)
        density_multiplier = 1.0
        if total_targets >= 12:
            density_multiplier = 1.45
        elif total_targets >= 6:
            density_multiplier = 1.25
        elif total_targets >= 3:
            density_multiplier = 1.10

        # Sector Multiplier (forward tactical corridors receive alert weighting)
        sector_upper = (sector or "").upper().replace(" ", "_")
        is_priority_sector = any(s in sector_upper for s in self.high_threat_sectors)
        sector_multiplier = 1.30 if is_priority_sector else 1.0

        # Multi-Temporal Surge Delta Component
        surge_component = max(0, surge_delta) * 5.0

        # Final Unified Score capped at 100.0
        mas_score = min(100.0, (raw_weighted_sum * density_multiplier * sector_multiplier) + surge_component)
        mas_score = round(mas_score, 1)

        # Map to DEFCON Readiness Tiers
        if mas_score >= 75.0:
            defcon_level = 1
            defcon_tier = "DEFCON_1_CRITICAL"
            defcon_desc = "Critical Threat / Imminent Military Offensive Deployment"
            threat_level = "CRITICAL"
        elif mas_score >= 50.0:
            defcon_level = 2
            defcon_tier = "DEFCON_2_HIGH"
            defcon_desc = "High Tactical Combat Readiness / Elevated Force Posture"
            threat_level = "HIGH"
        elif mas_score >= 25.0:
            defcon_level = 3
            defcon_tier = "DEFCON_3_ELEVATED"
            defcon_desc = "Elevated Tactical Activity / Active Staging Detected"
            threat_level = "ELEVATED"
        elif mas_score >= 5.0:
            defcon_level = 4
            defcon_tier = "DEFCON_4_ROUTINE"
            defcon_desc = "Routine Patrol Movement / Low Tactical Footprint"
            threat_level = "ROUTINE"
        else:
            defcon_level = 5
            defcon_tier = "DEFCON_5_NORMAL"
            defcon_desc = "Zero Hostile Threat / Calm Baseline"
            threat_level = "NEGLIGIBLE"

        return {
            "mas_score": mas_score,
            "defcon_level": defcon_level,
            "defcon_tier": defcon_tier,
            "defcon_description": defcon_desc,
            "threat_level": threat_level,
            "target_count": total_targets,
            "density_multiplier": density_multiplier,
            "sector_multiplier": sector_multiplier,
            "surge_component": surge_component
        }


# Singleton accessor
_default_scorer: MilitaryActivityScorer = None

def get_military_activity_scorer() -> MilitaryActivityScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = MilitaryActivityScorer()
    return _default_scorer
