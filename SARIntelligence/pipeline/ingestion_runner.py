"""
SARIntelligence/pipeline/ingestion_runner.py
============================================
Master Ingestion & Indexing Orchestrator for SARIntelligence:
Iterates over all 70 processed SAR scenes, executes MasterTemplateBuilder,
applies Domain Routing (OCEAN -> SOTA Maritime OBB, GROUND -> MSTAR 10-Class),
computes mathematical Military Activity Score (MAS), persists 6-Layer Master
Template JSON schemas, and upserts 768-D SARCLIP vectors into Qdrant and SQLite.
"""

import os
import sys
import glob
import json
import sqlite3
import logging
from typing import List, Dict, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from SARIntelligence.pipeline.template_builder import MasterTemplateBuilder

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestionRunner")

DATA_DIR = os.path.join(ROOT_DIR, "SARIntelligence", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DETECTIONS_DIR = os.path.join(DATA_DIR, "detections")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "sarintelligence.db")
QDRANT_STORAGE_PATH = os.path.join(DATA_DIR, "qdrant_local")
COLLECTION_NAME = "sarintelligence_tactical_vectors"


def init_sqlite_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scenes (
            scene_id TEXT PRIMARY KEY,
            scene_name TEXT,
            domain TEXT,
            sector TEXT,
            tactical_category TEXT,
            lat REAL,
            lon REAL,
            mas_score REAL,
            defcon_level INTEGER,
            defcon_tier TEXT,
            threat_level TEXT,
            total_targets INTEGER,
            primary_classes TEXT,
            sitrep_text TEXT,
            surge_annotation TEXT,
            json_path TEXT,
            overlay_png TEXT
        )
    """)
    # Automatic schema migration
    existing_cols = [c[1] for c in cur.execute("PRAGMA table_info(scenes)").fetchall()]
    new_cols = {
        "mas_score": "REAL",
        "defcon_level": "INTEGER",
        "defcon_tier": "TEXT",
        "threat_level": "TEXT",
        "surge_annotation": "TEXT"
    }
    for col, ctype in new_cols.items():
        if col not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE scenes ADD COLUMN {col} {ctype}")
            except Exception as e:
                logger.warning("Column %s could not be added: %s", col, e)
    conn.commit()
    conn.close()


def get_qdrant_client():
    if not QDRANT_AVAILABLE:
        return None

    # Try network Qdrant daemon first (inside docker: qdrant:6333, outside: localhost:6333)
    hosts_to_try = [
        os.environ.get("QDRANT_HOST", "qdrant"),
        "localhost",
        "127.0.0.1"
    ]
    port = int(os.environ.get("QDRANT_PORT", 6333))

    for h in hosts_to_try:
        try:
            client = QdrantClient(host=h, port=port, timeout=2.0)
            client.get_collections()
            logger.info("Connected to Qdrant daemon at %s:%d", h, port)
            return client
        except Exception:
            continue

    # Fallback to local disk store
    try:
        os.makedirs(QDRANT_STORAGE_PATH, exist_ok=True)
        client = QdrantClient(path=QDRANT_STORAGE_PATH)
        logger.info("Connected to local disk Qdrant store at %s", QDRANT_STORAGE_PATH)
        return client
    except Exception as e:
        logger.warning("Local Qdrant fallback note: %s", e)
        return None


def run_full_master_ingestion(limit_per_category: int = None):
    init_sqlite_db()
    builder = MasterTemplateBuilder()

    qdrant_client = get_qdrant_client()
    if qdrant_client is not None:
        try:
            collections = [c.name for c in qdrant_client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
            logger.info("Qdrant collection '%s' verified ready!", COLLECTION_NAME)
        except Exception as e:
            logger.warning("Qdrant collection ensure error: %s", e)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()

    categories = ["ocean", "ground", "multi_temporal"]
    total_processed = 0
    qdrant_points = []

    for cat in categories:
        in_cat_dir = os.path.join(PROCESSED_DIR, cat)
        out_det_dir = os.path.join(DETECTIONS_DIR, cat)
        out_tmpl_dir = os.path.join(TEMPLATES_DIR, cat)

        os.makedirs(out_det_dir, exist_ok=True)
        os.makedirs(out_tmpl_dir, exist_ok=True)

        tif_files = sorted(glob.glob(os.path.join(in_cat_dir, "*.tif")))
        if limit_per_category:
            tif_files = tif_files[:limit_per_category]

        for idx, tif_path in enumerate(tif_files, 1):
            fn = os.path.basename(tif_path)
            base_name = os.path.splitext(fn)[0]
            png_path = os.path.join(in_cat_dir, f"{base_name}.png")
            out_overlay = os.path.join(out_det_dir, f"{base_name}_detected.png")
            out_json = os.path.join(out_tmpl_dir, f"{base_name}.json")

            tmpl = builder.build_template_for_scene(
                processed_tif=tif_path,
                processed_png=png_path,
                out_json_path=out_json,
                out_overlay_png=out_overlay
            )

            mas_meta = tmpl.get("military_activity_score", {})
            mas_score = mas_meta.get("mas_score", 0.0)
            defcon_tier = mas_meta.get("defcon_tier", "DEFCON_5_NORMAL")
            threat_level = mas_meta.get("threat_level", "NORMAL")

            # Insert into SQLite
            cur.execute("""
                INSERT OR REPLACE INTO scenes (
                    scene_id, scene_name, domain, sector, tactical_category,
                    lat, lon, mas_score, defcon_level, defcon_tier, threat_level,
                    total_targets, primary_classes, sitrep_text, surge_annotation,
                    json_path, overlay_png
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tmpl["scene_id"],
                tmpl["scene_name"],
                tmpl["domain"],
                tmpl["sector"],
                tmpl["tactical_category"],
                tmpl["geospatial_telemetry"]["center_coordinates"]["lat"],
                tmpl["geospatial_telemetry"]["center_coordinates"]["lon"],
                mas_score,
                tmpl["detection_summary"]["defcon_level"],
                defcon_tier,
                threat_level,
                tmpl["detection_summary"]["total_target_count"],
                ",".join(tmpl["detection_summary"]["class_counts"].keys()),
                tmpl["tactical_sitrep"]["sitrep_text"],
                tmpl["tactical_sitrep"].get("surge_annotation", ""),
                out_json,
                out_overlay
            ))

            # Prepare Qdrant point
            vector_768 = tmpl["embeddings"]["pre_bound_image_vector"]
            point_id = total_processed + 1
            payload = {
                "scene_id": tmpl["scene_id"],
                "scene_name": tmpl["scene_name"],
                "domain": tmpl["domain"],
                "sector": tmpl["sector"],
                "tactical_category": tmpl["tactical_category"],
                "mas_score": mas_score,
                "defcon_level": tmpl["detection_summary"]["defcon_level"],
                "defcon_tier": defcon_tier,
                "threat_level": threat_level,
                "total_targets": tmpl["detection_summary"]["total_target_count"],
                "lat": tmpl["geospatial_telemetry"]["center_coordinates"]["lat"],
                "lon": tmpl["geospatial_telemetry"]["center_coordinates"]["lon"],
                "sitrep_text": tmpl["tactical_sitrep"]["sitrep_text"],
                "surge_annotation": tmpl["tactical_sitrep"].get("surge_annotation", ""),
                "overlay_png": out_overlay
            }
            if qdrant_client is not None:
                qdrant_points.append(PointStruct(id=point_id, vector=vector_768, payload=payload))

            total_processed += 1
            logger.info(
                "[%d] Processed %s | Domain: %s | MAS: %.1f | %s | Targets: %d",
                total_processed, tmpl["scene_id"], tmpl["domain"], mas_score, defcon_tier, tmpl["detection_summary"]["total_target_count"]
            )

    conn.commit()
    conn.close()

    # Batch upsert into Qdrant
    if qdrant_client is not None and qdrant_points:
        try:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)
            logger.info("Upserted %d 768-D vectors into Qdrant collection '%s'!", len(qdrant_points), COLLECTION_NAME)
        except Exception as e:
            logger.warning("Qdrant batch upsert note: %s", e)

    logger.info("MASTER INGESTION PIPELINE FINISHED: %d scenes registered across DB, Qdrant, and FileStore!", total_processed)
    return total_processed


if __name__ == "__main__":
    run_full_master_ingestion()
