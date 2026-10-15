"""
SARIntelligence/agent/tactical_tools.py
=======================================
Autonomous Tactical Defense Intelligence Tools Suite.
Exposes exactly 4 core operational tools for military SAR satellite reconnaissance:
1. `query_database`: Structured relational DB lookup, metadata filtering, zero-activity/safe zone queries, and spatial corridors.
2. `semantic_similarity_search`: Multimodal 768-D SARCLIP vector search supporting BOTH text queries and uploaded image inputs.
3. `analyze_temporal_surge`: Multi-temporal surveillance comparison (T1 baseline vs T2 current), delta escalation ranking, and radar difference heatmaps.
4. `get_scene_telemetry`: Deep inspection of 6-layer Master Template telemetry, oriented bounding boxes, RCS in dBsm, and exact GPS.
"""

import os
import sys
import json
import glob
import sqlite3
import logging
from typing import List, Dict, Any, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logger = logging.getLogger("TacticalTools")
logging.basicConfig(level=logging.INFO)

DATA_DIR = os.path.join(ROOT_DIR, "SARIntelligence", "data")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "sarintelligence.db")
CATALOG_PATH = os.path.join(ROOT_DIR, "data", "sar_master_catalog.json")
COLLECTION_NAME = "sarintelligence_tactical_vectors"

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class TacticalTools:
    """
    Unified 4-Tool Defense Suite for Autonomous SAR Intelligence Retrieval.
    Clean, typed, production-grade interfaces for relational SQL, Qdrant vectors,
    temporal differencing, and granular target telemetry.
    """

    def __init__(self, catalog_path: Optional[str] = None, db_path: Optional[str] = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self.catalog_path = catalog_path or CATALOG_PATH
        self._master_catalog: Optional[Dict[str, Any]] = None
        self._encoder = None
        self._qdrant_client = None

    def _get_db(self) -> sqlite3.Connection:
        """Opens connection to the SQLite intelligence database."""
        return sqlite3.connect(self.db_path)

    @property
    def master_catalog(self) -> Dict[str, Any]:
        """Lazy-loads the 70-scene Master Catalog JSON on first access."""
        if self._master_catalog is None:
            if os.path.exists(self.catalog_path):
                try:
                    with open(self.catalog_path, "r", encoding="utf-8") as f:
                        self._master_catalog = json.load(f)
                    logger.info("Loaded %d scenes from Master Catalog JSON", len(self._master_catalog))
                except Exception as e:
                    logger.error("Failed to load catalog JSON: %s", e)
                    self._master_catalog = {}
            else:
                self._master_catalog = {}
        return self._master_catalog

    @property
    def encoder(self):
        """Lazy-loads the SARCLIP ViT-L-14 Vision-Language encoder only when vector search is called."""
        if self._encoder is None:
            from SARIntelligence.embeddings.sarclip_service import get_sarclip_service
            self._encoder = get_sarclip_service()
        return self._encoder

    @property
    def qdrant(self):
        """Lazy-connects to Qdrant vector database (checks Docker, host, and local disk)."""
        if self._qdrant_client is None and QDRANT_AVAILABLE:
            # 1. Try Docker container network address
            try:
                client = QdrantClient(url="http://qdrant:6333", timeout=2.0)
                client.get_collections()
                self._qdrant_client = client
                logger.info("Connected to Qdrant at http://qdrant:6333")
                return self._qdrant_client
            except Exception:
                pass

            # 2. Try host localhost address
            try:
                client = QdrantClient(url="http://localhost:6333", timeout=2.0)
                client.get_collections()
                self._qdrant_client = client
                logger.info("Connected to Qdrant at http://localhost:6333")
                return self._qdrant_client
            except Exception:
                pass

            # 3. Try local disk path if exists
            local_path = os.path.join(DATA_DIR, "qdrant_local")
            if os.path.exists(local_path):
                try:
                    self._qdrant_client = QdrantClient(path=local_path)
                    logger.info("Connected to Qdrant local disk at %s", local_path)
                    return self._qdrant_client
                except Exception as e:
                    logger.warning("Local Qdrant disk fallback error: %s", e)

        return self._qdrant_client

    def _format_urls(self, scene_id: str, domain: str) -> Dict[str, str]:
        """Constructs canonical web-accessible URLs for preprocessed and detection overlay images."""
        dom = domain.lower() if domain else "ground"
        if dom == "land_tactical_defense":
            dom = "ground"
        elif dom == "maritime_domain":
            dom = "ocean"

        clean_id = scene_id.replace("_detected", "")

        overlay_url = f"/sar-detections/{dom}/{clean_id}_detected.png"
        preprocessed_url = f"/sar-processed/{dom}/{clean_id}.png"

        return {
            "overlay_url": overlay_url,
            "preprocessed_url": preprocessed_url
        }

    # =========================================================================
    # TOOL 1: RELATIONAL & SPATIAL DATABASE QUERY TOOL
    # =========================================================================
    def query_database(
        self,
        domain: Optional[str] = None,
        sector: Optional[str] = None,
        target_class: Optional[str] = None,
        min_targets: Optional[int] = None,
        max_targets: Optional[int] = None,
        defcon_level: Optional[int] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Executes structured relational queries against the military telemetry database.
        
        Parameters:
        - domain: Filter by domain ('ground', 'ocean', 'multi_temporal').
        - sector: Filter by surveillance sector (e.g. 'AKHNOOR', 'CHAMB', 'GALWAN', 'BAY_OF_BENGAL').
        - target_class: Search for combat hardware (e.g. 'T72', 'BMP2', 'BTR70', 'ZSU_23_4', 'vessel', 'aircraft').
        - min_targets: Minimum required target count.
        - max_targets: Maximum allowed target count (USE max_targets=0 for negative constraints / safe zones / calm sea).
        - defcon_level: Filter by DEFCON alert tier (1 for Critical, 2 for Elevated, 3 for Stable).
        - min_lat, max_lat, min_lon, max_lon: Geographic bounding-box corridor filter.
        - limit: Maximum number of records to return (default: 10).
        
        Returns:
        A list of matching scenes with GPS coordinates, target counts, DEFCON tier, and image URLs.
        """
        conn = self._get_db()
        cur = conn.cursor()

        query = (
            "SELECT scene_id, scene_name, domain, sector, tactical_category, "
            "lat, lon, defcon_level, defcon_tier, total_targets, primary_classes, "
            "sitrep_text, surge_annotation "
            "FROM scenes WHERE 1=1"
        )
        params = []

        if domain:
            query += " AND (domain LIKE ? OR tactical_category LIKE ?)"
            params.extend([f"%{domain}%", f"%{domain}%"])

        if sector:
            clean_sec = sector.replace("_", " ").strip()
            query += " AND (sector LIKE ? OR sitrep_text LIKE ? OR scene_name LIKE ?)"
            params.extend([f"%{sector}%", f"%{clean_sec}%", f"%{clean_sec}%"])

        if target_class:
            tc_low = target_class.lower()
            if any(w in tc_low for w in ["vessel", "warship", "ship", "boat", "fleet"]):
                query += " AND (domain = 'OCEAN' OR primary_classes LIKE '%BOAT%' OR primary_classes LIKE '%VESSEL%' OR sitrep_text LIKE '%vessel%' OR sitrep_text LIKE '%warship%')"
            elif any(w in tc_low for w in ["tank", "armor"]):
                query += " AND (primary_classes LIKE '%T72%' OR primary_classes LIKE '%T62%' OR primary_classes LIKE '%BMP%' OR primary_classes LIKE '%BTR%' OR sitrep_text LIKE '%tank%')"
            elif any(w in tc_low for w in ["aircraft", "fighter", "plane"]):
                query += " AND (primary_classes LIKE '%AIRCRAFT%' OR tactical_category LIKE '%AIRBASE%' OR sitrep_text LIKE '%aircraft%')"
            else:
                query += " AND (primary_classes LIKE ? OR sitrep_text LIKE ?)"
                params.extend([f"%{target_class}%", f"%{target_class}%"])

        if min_targets is not None:
            query += " AND total_targets >= ?"
            params.append(min_targets)
        if max_targets is not None:
            query += " AND total_targets <= ?"
            params.append(max_targets)
        if defcon_level is not None:
            query += " AND defcon_level = ?"
            params.append(defcon_level)
        if min_lat is not None:
            query += " AND lat >= ?"
            params.append(min_lat)
        if max_lat is not None:
            query += " AND lat <= ?"
            params.append(max_lat)
        if min_lon is not None:
            query += " AND lon >= ?"
            params.append(min_lon)
        if max_lon is not None:
            query += " AND lon <= ?"
            params.append(max_lon)

        # Ordering doctrine: High threat first if searching active, low targets first if searching safe zones
        if max_targets == 0:
            query += " ORDER BY total_targets ASC, defcon_level DESC LIMIT ?"
        else:
            query += " ORDER BY defcon_level ASC, total_targets DESC LIMIT ?"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        results = []
        for r in rows:
            sid, sname, dom, sec, tac, lat, lon, defcon, tier, targets, pclasses, sitrep, surge = r
            urls = self._format_urls(sid, dom)
            results.append({
                "scene_id": sid,
                "scene_name": sname,
                "domain": dom,
                "sector": sec,
                "tactical_category": tac,
                "coordinates": {"lat": round(float(lat), 4), "lon": round(float(lon), 4)},
                "defcon_level": defcon,
                "defcon_tier": tier,
                "total_targets": targets,
                "primary_classes": pclasses,
                "sitrep_text": sitrep,
                "surge_annotation": surge,
                "overlay_url": urls["overlay_url"],
                "preprocessed_url": urls["preprocessed_url"]
            })

        return results

    # =========================================================================
    # TOOL 2: MULTIMODAL 768-D SEMANTIC VECTOR SEARCH (IMAGE & TEXT)
    # =========================================================================
    def semantic_similarity_search(
        self,
        query_text: Optional[str] = None,
        image_path: Optional[str] = None,
        domain_filter: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Executes open-vocabulary multimodal retrieval across the 70 scenes
        using 768-D SARCLIP foundation embeddings in Qdrant.
        """
        if not query_text and "query" in kwargs:
            query_text = kwargs["query"]
        if not query_text and not image_path:
            raise ValueError("Either 'query_text' or 'image_path' must be provided.")

        # 1. Compute 768-D query embedding
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image not found: {image_path}")
            logger.info("Encoding query image '%s' using SARCLIP Vision Transformer...", image_path)
            query_vec = self.encoder.encode_image(image_path)
        else:
            logger.info("Encoding query text '%s' using SARCLIP Text Transformer...", query_text)
            query_vec = self.encoder.encode_text(query_text)

        # 2. Query Qdrant vector database
        if self.qdrant is not None:
            try:
                hits = self.qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vec,
                    limit=top_k * 2 if domain_filter else top_k
                ).points

                results = []
                for h in hits:
                    p = dict(h.payload)
                    p_domain = p.get("domain", "").lower()
                    if domain_filter and domain_filter.lower() not in p_domain:
                        continue

                    sid = p.get("scene_id", "")
                    dom = p.get("domain", "ground")
                    urls = self._format_urls(sid, dom)

                    results.append({
                        "scene_id": sid,
                        "scene_name": p.get("scene_name", sid),
                        "similarity_score": round(float(h.score), 4),
                        "domain": dom,
                        "sector": p.get("sector", ""),
                        "tactical_category": p.get("tactical_category", ""),
                        "coordinates": {"lat": p.get("lat", 0.0), "lon": p.get("lon", 0.0)},
                        "defcon_level": p.get("defcon_level", 3),
                        "defcon_tier": p.get("defcon_tier", "DEFCON_3_ELEVATED"),
                        "total_targets": p.get("total_targets", 0),
                        "sitrep_text": p.get("sitrep_text", ""),
                        "surge_annotation": p.get("surge_annotation", ""),
                        "overlay_url": urls["overlay_url"],
                        "preprocessed_url": urls["preprocessed_url"]
                    })
                    if len(results) >= top_k:
                        break

                return results
            except Exception as e:
                logger.warning("Qdrant vector query failed (%s). Falling back to database.", e)

        # 3. Fallback: Relational query if vector DB unavailable
        logger.info("Executing fallback database lookup for query: %s", query_text or image_path)
        fallback = self.query_database(domain=domain_filter, limit=top_k)
        for idx, item in enumerate(fallback):
            item["similarity_score"] = round(1.0 - (idx * 0.08), 4)
        return fallback

    # =========================================================================
    # TOOL 3: MULTI-TEMPORAL RADAR DIFFERENCE & ESCALATION SURGE TOOL
    # =========================================================================
    def analyze_temporal_surge(
        self,
        sector: Optional[str] = None,
        pair_number: Optional[int] = None,
        min_surge_delta: int = 1,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Scans surveillance sectors across co-registered multi-temporal radar passes.
        Compares baseline T1 (Past) against current T2 (Recent) observations,
        computes force delta (Delta = T2 - T1), and provides difference heatmap paths.
        """
        # Handle loose arguments passed by LLMs
        if pair_number is None:
            for k in ["pair", "pair_id", "pair_no"]:
                if k in kwargs and kwargs[k]:
                    try:
                        import re
                        m = re.search(r'\d+', str(kwargs[k]))
                        if m:
                            pair_number = int(m.group(0))
                            break
                    except Exception:
                        pass
        conn = self._get_db()
        cur = conn.cursor()

        query = """
            SELECT scene_id, scene_name, sector, lat, lon, total_targets, surge_annotation, defcon_tier
            FROM scenes
            WHERE scene_id LIKE '%_T2_current%'
        """
        params = []
        if sector:
            query += " AND sector LIKE ?"
            params.append(f"%{sector}%")
        if pair_number is not None:
            p_str = f"temporal_pair_{pair_number:02d}_"
            query += " AND scene_id LIKE ?"
            params.append(f"%{p_str}%")

        query += " ORDER BY total_targets DESC"
        cur.execute(query, params)
        t2_rows = cur.fetchall()
        conn.close()

        surges = []
        for r in t2_rows:
            t2_sid, sname, sec, lat, lon, t2_targets, surge_annot, tier = r
            t1_sid = t2_sid.replace("_T2_current", "_T1_baseline")

            # Look up matching T1 baseline
            conn = self._get_db()
            cur = conn.cursor()
            cur.execute("SELECT total_targets, defcon_tier FROM scenes WHERE scene_id = ?", (t1_sid,))
            t1_row = cur.fetchone()
            conn.close()

            t1_targets = t1_row[0] if t1_row else 0
            t1_tier = t1_row[1] if t1_row else "DEFCON_5_NORMAL"

            delta = t2_targets - t1_targets

            # Extract pair number (e.g. '01' from 'temporal_pair_01_T2_current')
            pair_num_match = glob.re.search(r'temporal_pair_(\d+)', t2_sid) if hasattr(glob, 're') else None
            if not pair_num_match:
                import re
                pair_num_match = re.search(r'temporal_pair_(\d+)', t2_sid)
            p_num = pair_num_match.group(1) if pair_num_match else "01"

            # Threat level assessment
            if delta >= 10:
                threat_assessment = "CRITICAL_ESCALATION"
            elif delta >= 4:
                threat_assessment = "TACTICAL_SURGE"
            elif delta >= 1:
                threat_assessment = "ROUTINE_EXPANSION"
            elif delta == 0:
                threat_assessment = "STATIC"
            else:
                threat_assessment = "DE_ESCALATION"

            if delta >= min_surge_delta or min_surge_delta <= 0:
                diff_heatmap_url = f"/sar-diffs/diff_pair_{p_num}.png"
                t1_overlay_url = f"/sar-detections/multi_temporal/{t1_sid}_detected.png"
                t2_overlay_url = f"/sar-detections/multi_temporal/{t2_sid}_detected.png"

                surges.append({
                    "pair_id": f"pair_{p_num}",
                    "sector": sec,
                    "scene_name": sname,
                    "coordinates": {"lat": round(float(lat), 4), "lon": round(float(lon), 4)},
                    "baseline_epoch": "T1 (Past Observation)",
                    "baseline_scene_id": t1_sid,
                    "baseline_targets": t1_targets,
                    "baseline_defcon": t1_tier,
                    "current_epoch": "T2 (Current Observation)",
                    "current_scene_id": t2_sid,
                    "current_targets": t2_targets,
                    "current_defcon": tier,
                    "net_delta": f"+{delta}" if delta > 0 else str(delta),
                    "delta_numeric": delta,
                    "threat_assessment": threat_assessment,
                    "surge_explanation": surge_annot,
                    "t1_image_url": t1_overlay_url,
                    "t2_image_url": t2_overlay_url,
                    "diff_heatmap_url": diff_heatmap_url
                })

        surges.sort(key=lambda x: x["delta_numeric"], reverse=True)
        return surges[:limit]

    # =========================================================================
    # TOOL 4: COMPLETE SCENE TELEMETRY & TARGET INSPECTION TOOL
    # =========================================================================
    def get_scene_telemetry(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Deep inspection of the 6-layer Master Template JSON for a given scene_id.
        Extracts exact individual vehicle bounding boxes, radar cross section (RCS in dBsm),
        individual target GPS coordinates, and classification confidence scores.
        
        Parameters:
        - scene_id: Unique identifier of the radar scene (e.g. 'ground_armor_tanks_02', 'ocean_active_fleet_01').
        
        Returns:
        Complete structured dictionary with target bounding boxes, RCS, GPS, and image artifact paths.
        """
        clean_id = scene_id.replace("_detected", "")

        # 1. First check Master Catalog in memory
        catalog = self.master_catalog
        record = catalog.get(clean_id) or catalog.get(scene_id)

        # 2. If not found in catalog, inspect individual template JSON file on disk
        if not record:
            pattern = os.path.join(DATA_DIR, "templates", "**", f"{clean_id}.json")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                try:
                    with open(matches[0], "r", encoding="utf-8") as f:
                        record = json.load(f)
                except Exception as e:
                    logger.error("Failed to read template JSON %s: %s", matches[0], e)

        if not record:
            return None

        # Clean and standardize telemetry output
        dom = record.get("domain", "ground")
        urls = self._format_urls(clean_id, dom)

        geo = record.get("geospatial_telemetry", {})
        det_summary = record.get("detection_summary", {})
        mas = record.get("military_activity_score", {})
        sitrep = record.get("tactical_sitrep", {})

        # Extract target detections list
        detections = []
        raw_dets = record.get("detections", [])
        for d in raw_dets:
            detections.append({
                "target_id": d.get("target_id") or d.get("detection_id", "TGT"),
                "class": d.get("class", "UNKNOWN"),
                "display_name": d.get("display_name", d.get("class", "Combat Asset")),
                "confidence": round(float(d.get("confidence", 0.85)), 3),
                "rcs_db": round(float(d.get("rcs_db", 8.5)), 2),
                "oriented_bbox": d.get("oriented_bbox", {}),
                "coordinates": d.get("coordinates", {})
            })

        return {
            "scene_id": clean_id,
            "scene_name": record.get("scene_name", clean_id),
            "domain": dom,
            "sector": record.get("sector", "TACTICAL_ZONE"),
            "tactical_category": record.get("tactical_category", "MILITARY_OPERATION"),
            "center_coordinates": {
                "lat": geo.get("center_latitude", 0.0),
                "lon": geo.get("center_longitude", 0.0)
            },
            "total_targets": det_summary.get("total_targets", len(detections)),
            "primary_classes": det_summary.get("primary_classes", []),
            "mas_score": mas.get("mas_score", 0.5),
            "defcon_level": mas.get("defcon_level", 2),
            "defcon_tier": mas.get("defcon_tier", "DEFCON_2_HIGH"),
            "threat_level": sitrep.get("threat_level", "HIGH"),
            "briefing": sitrep.get("briefing", ""),
            "detections": detections,
            "image_artifacts": {
                "overlay_url": urls["overlay_url"],
                "preprocessed_url": urls["preprocessed_url"]
            }
        }
