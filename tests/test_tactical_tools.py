"""
tests/test_tactical_tools.py
============================
Automated Test Suite for the 4 Core Defense Tactical Tools:
1. query_database (Relational, spatial, negative/safe-zone search)
2. semantic_similarity_search (Multimodal text & image retrieval)
3. analyze_temporal_surge (Temporal pass differencing, buildup delta, heatmaps)
4. get_scene_telemetry (Granular target bounding boxes, RCS, GPS coordinates)
"""

import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from SARIntelligence.agent.tactical_tools import TacticalTools


@pytest.fixture(scope="module")
def tools():
    return TacticalTools()


# =============================================================================
# TOOL 1 TESTS: query_database
# =============================================================================

def test_query_database_ground_tanks(tools):
    """Test 1A: Search for T-72 tanks in Akhnoor sector."""
    results = tools.query_database(sector="AKHNOOR", target_class="T72")
    assert len(results) > 0, "Expected at least 1 scene in Akhnoor with T-72"
    scene = results[0]
    assert "AKHNOOR" in scene["sector"]
    assert scene["total_targets"] > 0
    assert "lat" in scene["coordinates"] and "lon" in scene["coordinates"]
    assert scene["overlay_url"].startswith("/sar-detections/")
    print(f"\n[TEST 1A PASS] Retrieved {scene['scene_id']} with {scene['total_targets']} targets in {scene['sector']}")


def test_query_database_negative_constraint(tools):
    """Test 1B: Negative constraint test for zero military activity / safe zones."""
    results = tools.query_database(max_targets=0, limit=5)
    assert len(results) > 0, "Expected at least 1 scene with zero military activity"
    for r in results:
        assert r["total_targets"] == 0, f"Expected 0 targets but got {r['total_targets']}"
    print(f"\n[TEST 1B PASS] Retrieved {len(results)} zero-activity / safe-zone scenes. Sample: {results[0]['scene_id']}")


def test_query_database_defcon_level(tools):
    """Test 1C: Filter by DEFCON alert tier."""
    results = tools.query_database(defcon_level=1, limit=5)
    assert len(results) > 0, "Expected DEFCON-1 critical scenes"
    for r in results:
        assert r["defcon_level"] == 1
    print(f"\n[TEST 1C PASS] Retrieved {len(results)} DEFCON-1 critical alert scenes.")


def test_query_database_spatial_corridor(tools):
    """Test 1D: Spatial corridor bounding box filter."""
    # Jammu & Kashmir / Northern Sector corridor: Lat 32.0 to 35.0, Lon 74.0 to 79.0
    results = tools.query_database(min_lat=32.0, max_lat=35.0, min_lon=74.0, max_lon=79.0, limit=5)
    assert len(results) > 0, "Expected scenes in J&K corridor"
    for r in results:
        assert 32.0 <= r["coordinates"]["lat"] <= 35.0
        assert 74.0 <= r["coordinates"]["lon"] <= 79.0
    print(f"\n[TEST 1D PASS] Spatial bounding box matched {len(results)} scenes in operational corridor.")


# =============================================================================
# TOOL 2 TESTS: semantic_similarity_search
# =============================================================================

def test_semantic_search_text(tools):
    """Test 2A: Open-vocabulary text retrieval."""
    query = "naval warships in maritime sector"
    results = tools.semantic_similarity_search(query_text=query, top_k=3)
    assert len(results) > 0
    top = results[0]
    assert "similarity_score" in top
    assert top["similarity_score"] > 0.0
    assert "overlay_url" in top
    print(f"\n[TEST 2A PASS] Text vector search matched: {top['scene_id']} (Score: {top['similarity_score']})")


def test_semantic_search_image(tools):
    """Test 2B: Image-to-Image similarity retrieval."""
    # Test using an authentic ground preprocessed image
    test_img = os.path.join(ROOT_DIR, "sar-processed", "ground", "ground_armor_tanks_01.png")
    if not os.path.exists(test_img):
        test_img = os.path.join(ROOT_DIR, "SARIntelligence", "data", "processed", "ground", "ground_armor_tanks_01.png")

    if os.path.exists(test_img):
        results = tools.semantic_similarity_search(image_path=test_img, top_k=3)
        assert len(results) > 0
        top = results[0]
        assert "similarity_score" in top
        print(f"\n[TEST 2B PASS] Image-to-Image vector search matched: {top['scene_id']} (Score: {top['similarity_score']})")
    else:
        print("\n[TEST 2B SKIP] Test image not found on host path.")


# =============================================================================
# TOOL 3 TESTS: analyze_temporal_surge
# =============================================================================

def test_analyze_temporal_surge_ranking(tools):
    """Test 3A: Ranked list of sectors with military activity surges."""
    surges = tools.analyze_temporal_surge(min_surge_delta=1, limit=5)
    assert len(surges) > 0, "Expected at least 1 sector with positive troop surge"
    top_surge = surges[0]
    assert top_surge["delta_numeric"] >= 1
    assert "diff_heatmap_url" in top_surge
    assert "t1_image_url" in top_surge
    assert "t2_image_url" in top_surge
    print(f"\n[TEST 3A PASS] Top surge in {top_surge['sector']}: {top_surge['net_delta']} units ({top_surge['threat_assessment']})")


def test_analyze_temporal_surge_specific_pair(tools):
    """Test 3B: Specific temporal pair 01 surge evaluation."""
    surges = tools.analyze_temporal_surge(pair_number=1)
    assert len(surges) > 0, "Expected Pair 01 results"
    p1 = surges[0]
    assert p1["pair_id"] == "pair_01"
    assert p1["baseline_targets"] >= 0
    assert p1["current_targets"] >= 0
    assert p1["diff_heatmap_url"] == "/sar-diffs/diff_pair_01.png"
    print(f"\n[TEST 3B PASS] Pair 01 delta: T1={p1['baseline_targets']} -> T2={p1['current_targets']} (Net: {p1['net_delta']})")


# =============================================================================
# TOOL 4 TESTS: get_scene_telemetry
# =============================================================================

def test_get_scene_telemetry(tools):
    """Test 4A: Deep inspection of granular vehicle bounding boxes and RCS."""
    telemetry = tools.get_scene_telemetry("ground_armor_tanks_02")
    assert telemetry is not None, "Expected telemetry for ground_armor_tanks_02"
    assert telemetry["scene_id"] == "ground_armor_tanks_02"
    assert telemetry["total_targets"] > 0
    assert len(telemetry["detections"]) > 0

    first_target = telemetry["detections"][0]
    assert "class" in first_target
    assert "rcs_db" in first_target
    assert "oriented_bbox" in first_target
    assert "coordinates" in first_target
    print(f"\n[TEST 4A PASS] Telemetry for {telemetry['scene_id']}: {telemetry['total_targets']} targets. Sample target: {first_target['class']} (RCS: {first_target['rcs_db']} dBsm)")


if __name__ == "__main__":
    t = TacticalTools()
    print("="*60)
    print("RUNNING TACTICAL TOOLS DIRECT VERIFICATION")
    print("="*60)
    test_query_database_ground_tanks(t)
    test_query_database_negative_constraint(t)
    test_query_database_defcon_level(t)
    test_query_database_spatial_corridor(t)
    test_semantic_search_text(t)
    test_analyze_temporal_surge_ranking(t)
    test_analyze_temporal_surge_specific_pair(t)
    test_get_scene_telemetry(t)
    print("\n" + "="*60)
    print("ALL 4 DEFENSE TOOLS PASSED VERIFICATION WITH ZERO ERRORS!")
    print("="*60)
