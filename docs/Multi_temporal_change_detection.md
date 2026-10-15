# 🛰️ Multi-Temporal Change Detection & False-Alarm Suppression Specification

> [!IMPORTANT]
> **Problem Statement**: SIH-2026 PS26227 — Sections 2.2.2 & 2.2.3 (Ministry of Defence / Indian Army)
> This document is the complete, fully-specified implementation guide for the Multi-Temporal Change Analysis sub-system.

---

## Table of Contents

1. [Problem Statement Requirements Targeted](#1-problem-statement-requirements-targeted)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Stage 1 — Temporal Tile Staging & Quality Gate Engine](#3-stage-1--temporal-tile-staging--quality-gate-engine)
4. [Stage 2 — Two-Tier False-Alarm Suppression Pipeline](#4-stage-2--two-tier-false-alarm-suppression-pipeline)
5. [Stage 3 — Sub-Pixel Phase Correlation Co-Registration](#5-stage-3--sub-pixel-phase-correlation-co-registration)
6. [Stage 4 — Prithvi-EO-2.0-300M Patch Change Detection](#6-stage-4--prithvi-eo-20-300m-patch-change-detection)
7. [Stage 5 — Semantic Change Classification (Multi-Spectral)](#7-stage-5--semantic-change-classification-multi-spectral)
8. [Stage 6 — Analyst Review & Provenance Workflow](#8-stage-6--analyst-review--provenance-workflow)
9. [Stage 7 — Temporal Backward Traversal (Date Estimation)](#9-stage-7--temporal-backward-traversal-date-estimation)
10. [PostgreSQL Schema Reference](#10-postgresql-schema-reference)
11. [API Endpoints](#11-api-endpoints)
12. [End-to-End Data Flow](#12-end-to-end-data-flow)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. Problem Statement Requirements Targeted

This pipeline directly addresses Ministry of Defence (MoD) / Indian Army Problem Statement **SIH-26227 Sections 2.2.2 and 2.2.3**:

| PS Section | Capability Mandate | Implementation Status |
|---|---|---|
| **2.2.2** | **Multi-Temporal Change Analysis**<br>• For user-specified AOI and time window, identify physical, structural, and environmental changes (appearance, disappearance, expansion, contraction).<br>• Classify detected changes into tactical categories: construction, land clearance, water-extent variation, road development.<br>• Estimate earliest available observation date at which change is visible via temporal backward traversal. | ✅ **Implemented** |
| **2.2.3** | **False-Alarm Suppression & Quality Handling (Tier-2)**<br>• Suppress false changes caused by seasonal variation, sun angle, illumination differences, clouds, shadows, and imperfect co-registration.<br>• Utilize pixel-level validity masks (`bad_mask_before \| bad_mask_after`) and dynamic radiometric normalization.<br>• Enforce strict usable ground area thresholding (≥ 30% clean pixels) and quality confidence gates. | ✅ **Implemented** |

---

## 2. System Architecture Overview

```
User selects Target Tile on /change.html
              │
              ▼
┌──────────────────────────────────────────────────┐
│  STAGE 1: Temporal Staging & Quality Gate        │  change_stager.py
│  PostgreSQL → site_key → all temporal epochs     │
│  QualityCheckEngine → STAY / DROPPED per tile    │
│  manifest.json audit log                         │
└──────────────────────┬───────────────────────────┘
                       │ Clean STAY tiles only
                       ▼
┌──────────────────────────────────────────────────┐
│  STAGE 2A: Tier-1 Cloud & Shadow Masking         │  quality_auditor.py
│  bad_mask.tif pixel audit → bad_pixel_pct        │
│  Tiles > 15% bad pixels → DROPPED                │
│                                                  │
│  STAGE 2B: Radiometric Normalization             │  calibrator.py
│  0.5%–99.5% percentile stretch per valid pixel   │
│  QB-Norm Inverted Softmax hubness suppression    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STAGE 3: Sub-Pixel Phase Correlation            │  patch_change_detector.py
│  skimage.registration.phase_cross_correlation    │
│  NIR/Red band registration (upsample_factor=10)  │
│  Bicubic warp of T2 to eliminate jitter          │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STAGE 4: Prithvi-EO-2.0-300M Patch Embeddings  │  patch_change_detector.py
│  8×8 spatial grid → 64 patches                  │  prithvi_encoder.py
│  Per-patch QA filter (> 20% bad → drop)         │
│  GPU CUDA FP16 inference → 1024-D features       │
│  Cosine Distance → 1-Sided Left-MAD Z-scoring    │
│  8-Neighbor Connected Component Cluster Filter   │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STAGE 5: Semantic Change Classification         │  semantic_change_classifier.py
│  NDVI / NDWI / NDBI / BSI per candidate patch    │
│  Before → After land cover transition labelling  │
│  Spectral stability filter (Δ < 0.002 → drop)    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STAGE 6: Analyst Review & Provenance            │  analyst_workflow.py
│  Human-in-the-loop confirm / reject / unsure     │
│  PostgreSQL audit trail (analyst_decisions)      │
│  GeoJSON + PDF SITREP export                     │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  STAGE 7: Temporal Backward Traversal            │  temporal_splitter.py
│  Bisect-search across available epoch dates      │
│  Earliest change visibility date estimation      │
└──────────────────────────────────────────────────┘
```

---

## 3. Stage 1 — Temporal Tile Staging & Quality Gate Engine

**Module**: `backend/services/change_stager.py`

### `get_all_regions_grid_map()`
Queries PostgreSQL to build the full mosaic canvas for the `change.html` frontend:
1. Fetches all ingested `ingestion_coverage` regions (geometry, bbox, dates)
2. Fetches all Sentinel-2 tiles (excluding Maxar high-res) with spatial bounds
3. Groups tiles by `site_key` (geographic footprint identifier)
4. Computes temporal depth (`epoch_count`) per site — how many distinct dates exist per ground location
5. Returns a complete `{ regions, tiles }` map for the interactive Leaflet change workbench

### `stage_tile_temporal_series(tile_id, staging_dir_name)`
Core staging orchestrator for a selected target tile:

```python
# Execution flow:
# 1. Query PostgreSQL for all tiles sharing the same site_key (same ground footprint)
# 2. Sort by acquisition_date ascending (oldest T1 → newest Tn)
# 3. Run QualityCheckEngine on each tile's _mask.tif
# 4. STAY: Copy clean tiles to data/change_staging/<site_key>/T<idx>_<date>/
# 5. DROPPED: Record rejection reason in manifest.json
# 6. Generate manifest.json with full audit log:
#    {
#      "site_key": ...,
#      "selected_tiles": [...],  # STAY
#      "dropped_tiles": [...],   # DROPPED with reasons
#      "quality_audit": {...}
#    }
```

#### Staging Directory Structure
```
data/change_staging/<site_key>/
├── T0_2022-03-15/
│   ├── tile.tif          ← Clean multi-band GeoTIFF (T1 Baseline)
│   ├── mask.tif          ← Cloud/shadow mask
│   └── thumbnail.jpg     ← RGB preview
├── T1_2024-11-08/
│   ├── tile.tif          ← Clean multi-band GeoTIFF (T2 Current)
│   ├── mask.tif
│   └── thumbnail.jpg
└── manifest.json         ← Full audit log
```

---

## 4. Stage 2 — Two-Tier False-Alarm Suppression Pipeline

### Tier 1 — Pixel-Level Cloud & Shadow Masking

**Module**: `backend/services/quality_auditor.py` — `QualityCheckEngine`

The Quality Check Engine reads each tile's `_mask.tif` (generated by `s2cloudless` + directional shadow ray-casting during ingestion) and produces a `TileQualityReport`:

```python
@dataclass
class TileQualityReport:
    tile_id: str
    site_key: str
    acquisition_date: str
    total_pixels: int
    bad_pixels_count: int       # Pixels flagged in mask.tif
    good_pixels_count: int
    bad_pixel_pct: float        # % of bad pixels
    usable_surface_pct: float   # % of clean valid surface
    quality_confidence: float   # 1.0 - bad_pixel_pct/100
    status: str                 # "STAY" or "DROPPED"
    is_dropped: bool
    rejection_reasons: List[str]
```

#### Quality Gates
| Threshold | Value | Action |
|---|---|---|
| `max_bad_pixel_pct` | `15.0%` | If `bad_pixel_pct > 15%` → tile is **DROPPED** |
| `min_usable_pct` | `85.0%` | If `usable_surface_pct < 85%` → tile is **DROPPED** |

This directly implements PS 2.2.3's mandate: _"strict usable ground area thresholding (≥ 30% clean pixels)"_ — the platform enforces a stricter 85% clean threshold.

#### Mask Composition
The `bad_mask` used by the quality engine is a union of:
```
bad_mask = cloud_mask | cloud_shadow_mask | sensor_dropout_mask | nodata_mask

where:
  cloud_mask        → s2cloudless gradient-boosted tree classification
  cloud_shadow_mask → directional shadow ray-tracing from cloud centroid projections
  sensor_dropout_mask → detector failure / striping artefacts
  nodata_mask       → image border zero-padding or missing scan lines
```

---

### Tier 2 — Radiometric Normalization & Hubness Suppression

**Module**: `backend/services/calibrator.py` — `RetrievalCalibrator`

Implements **QB-Norm (Querybank Normalization)** with **Dynamic Inverted Softmax (DIS)**:

```
Reference: Bogolin et al., "Cross Modal Retrieval with Querybank Normalisation", CVPR 2022.
```

#### QB-Norm Formula
```
DIS_score(q, d) = sim(q, d) / Σᵢ exp(β × sim(q, bankᵢ))

where:
  q      = query embedding vector (512-D RemoteCLIP)
  d      = document (tile) embedding vector
  bankᵢ  = one of 512 RSICD query bank reference embeddings
  β      = inverse temperature (default: 20.0)
```

#### Hubness Detection
- Computes `hub_topk=5` most retrieved tiles across all queries
- If top-1 result is a **hub** (frequently retrieved regardless of query): applies Inverted Softmax dampening
- If top-1 result is **not a hub**: raw cosine similarity preserved
- Prevents single high-energy tiles from dominating all retrieval results

#### Radiometric Percentile Normalization
Applied during ingestion (not at query time):
```
For each spectral band:
  p_low  = 0.5th percentile (computed ONLY over valid unmasked pixels)
  p_high = 99.5th percentile (computed ONLY over valid unmasked pixels)
  normalized = clip((pixel - p_low) / (p_high - p_low), 0.0, 1.0)
```

This masks out cloud/shadow pixels before computing stretch bounds, preventing atmospheric outliers from compressing the valid surface dynamic range.

---

## 5. Stage 3 — Sub-Pixel Phase Correlation Co-Registration

**Module**: `backend/services/patch_change_detector.py` — `align_subpixel_misregistration()`

Satellite orbital jitter and multi-pass viewing geometry differences between T1 and T2 acquisitions introduce sub-pixel spatial offsets. These offsets generate **systematic false alarms** at tile edges and along high-contrast structural boundaries.

### Algorithm
```
1. Select best-contrast reference band (NIR preferred > Red > Band index 3)
2. Crop central 70% of tile (removes border zero-padding and edge artefacts)
3. Compute phase cross-correlation (FFT-based frequency domain):
   shifts, error, _ = phase_cross_correlation(
       T1_band_crop,
       T2_band_crop,
       upsample_factor=10    # 0.1px resolution
   )
4. Only apply correction if 0.1px ≤ |shift| ≤ 3.0px
   (< 0.1px = already aligned; > 3.0px = gross geometric error, not jitter)
5. Warp all T2 bands using scipy.ndimage.shift (order=1 bicubic)
6. Warp T2 mask using order=0 (nearest neighbour, preserves binary flag)
```

### Output Metadata
```json
{
  "applied": true,
  "shift_y_pixels": -0.342,
  "shift_x_pixels": 0.187,
  "error": 0.0023,
  "status": "aligned"
}
```

### Status Codes
| Status | Meaning |
|---|---|
| `aligned` | Correction applied successfully |
| `already_aligned` | Shift < 0.1px, no correction needed |
| `shift_out_of_bounds` | Shift > 3.0px, flagged as gross geometry error |
| `insufficient_texture` | Band variance < 1e-4, cannot compute correlation |
| `shape_mismatch` | T1/T2 arrays have different dimensions |
| `skimage_unavailable` | `skimage.registration` not installed |

---

## 6. Stage 4 — Prithvi-EO-2.0-300M Patch Change Detection

**Module**: `backend/services/patch_change_detector.py` — `PrithviPatchChangeDetector`
**Encoder**: `backend/services/prithvi_encoder.py`

### 4a. Spatial Grid Partitioning
```
512×512 tile → 8×8 grid → 64 patches of 64×64 pixels each

Per patch:
  bad_frac_T1 = mean(mask_before[patch_y0:y1, x0:x1] > 0)
  bad_frac_T2 = mean(mask_after[patch_y0:y1, x0:x1] > 0)

Drop patch if:
  bad_frac_T1 > 20% OR
  bad_frac_T2 > 20% OR
  mean(crop) < 1e-4 (NoData / black edge)
```

### 4b. Prithvi-EO-2.0-300M Latent Feature Encoding

**Prithvi-EO-2.0-300M** is a 300-million parameter geospatial foundation model jointly developed by **NASA** and **IBM Research**, pre-trained on 4.2M+ Sentinel-2 multi-temporal patches covering 8 spectral bands over 5 temporal frames.

```
Architecture: Vision Transformer (ViT) with temporal attention
Input:        Multi-band patch (C×H×W) — handles 4-band to 9-band Sentinel-2
Output:       1024-dimensional latent feature vector per patch

GPU Inference: CUDA FP16 TensorCore mixed precision
Batch Size:    32 patches per GPU forward pass
Memory Mode:   eval() + torch.no_grad() + autocast(fp16)
```

All 64 before-patches and 64 after-patches are encoded in a **single joint GPU batch** (128 forward passes) to maximize GPU utilization:
```python
all_crops   = crops_before + crops_after  # 128 crops total
all_vectors = encoder.encode_multiband_patches(all_crops, batch_size=32)
vecs_before = all_vectors[:64]
vecs_after  = all_vectors[64:]
```

### 4c. Cosine Distance Scoring
```
For each valid patch pair (i):
  similarity_i = dot(vec_before_i, vec_after_i)   # Both L2-normalized
  distance_i   = 1.0 - similarity_i               # Cosine distance

distance = 0.0  → identical semantic content (no change)
distance = 2.0  → maximally different semantic content (extreme change)
```

### 4d. Robust 1-Sided Left-MAD Z-Scoring

Standard Z-score is broken for change detection because genuine anomalies inflate the standard deviation, making Z-scores conservative and missing true changes.

**Left-MAD** computes dispersion strictly from the left side of the median (values ≤ median), using only the null/stable distribution:

```
median  = median(distances)
left_distances = distances[distances <= median]
left_MAD = median(|left_distances - median|)
sigma_left = 1.4826 × left_MAD      # Robust sigma estimator

Z_score_i = (distance_i - median) / (sigma_left + ε)

Mark as candidate if:
  Z_score_i >= z_threshold (default: 2.0)  AND
  distance_i >= min_dist   (default: 0.01)
```

> [!NOTE]
> The factor `1.4826` is the consistency constant making MAD equivalent to standard deviation under a Gaussian distribution.

### 4e. 8-Neighbor Connected Component Cluster Filtering

Isolated single-patch anomalies are almost always false alarms (cloud edges, sensor artefacts). The system uses `scipy.ndimage.label` with an 8-connectivity structuring element to cluster adjacent candidate patches:

```python
binary_grid[r, c] = 1  if patch is candidate
                  = 0  otherwise

labeled_grid, n_clusters = scipy.ndimage.label(
    binary_grid,
    structure=np.ones((3, 3))  # 8-neighbor connectivity
)

For each cluster:
  if cluster_size < min_cluster_size (default: 2):
    mark all patches in cluster as is_isolated_noise = True
    → NOT reported as confirmed candidates

Only spatially connected clusters of ≥ 2 patches are reported.
```

### 4f. Visual Overlay Generation

Change candidates are rendered as clean bounding boxes on the satellite imagery without full blue grid overlays:
- **Highlight color**: Bold crimson-red `(239, 68, 68)` — no blue grid, single highlight color
- **Box style**: Solid 2px border + semi-transparent fill
- **No text labels**: Strictly spatial boundary — no Z-score or class text drawn on image
- **Output**: Side-by-side Before / After images saved to `output_dir/`

---

## 7. Stage 5 — Semantic Change Classification (Multi-Spectral)

**Module**: `backend/services/semantic_change_classifier.py` — `SemanticChangeClassifier`

For each confirmed candidate patch, the semantic classifier computes multi-spectral indices and assigns land cover transitions.

### Multi-Spectral Index Formulae

| Index | Formula | Detects |
|---|---|---|
| **NDVI** | `(NIR - Red) / (NIR + Red)` | Vegetation density |
| **NDWI** | `(Green - NIR) / (Green + NIR)` | Open water bodies |
| **NDBI** | `(SWIR - NIR) / (SWIR + NIR)` | Built-up / impervious surfaces |
| **BSI** | `((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))` | Bare soil / exposed ground |

### Land Cover Hierarchy (Sentinel-2 L2A)

| Class ID | Name | Thresholds | Color |
|---|---|---|---|
| 1 | Water Body | `NDWI ≥ 0.20` AND `NDVI < 0.20` | `#0284c7` (blue) |
| 2 | Dense Vegetation | `NDVI ≥ 0.40` AND `NDWI < 0.20` | `#16a34a` (green) |
| 3 | Sparse Vegetation | `0.20 ≤ NDVI < 0.40` AND `NDWI < 0.20` | `#84cc16` (lime) |
| 4 | Built-Up Area | `NDVI < 0.25` AND (`NDBI ≥ 0.10` OR `NDBI ≥ BSI`) | `#ef4444` (red) |
| 5 | Bare Soil | `NDVI < 0.25` AND (`BSI > NDBI` OR `NDBI < 0.10`) | `#d97706` (amber) |
| 6 | Mixed Land Cover | Conflicting / transition signals | `#a855f7` (purple) |

### Band Mapping

Handles Sentinel-2 L2A configurations automatically:

| Channel Count | Band Mapping |
|---|---|
| **≥ 9 bands** | Full S2 L2A: `[B01, B02(Blue), B03(Green), B04(Red), B05, B08(NIR), B8A, B11(SWIR), B12]` |
| **5 bands** | `[Blue, Green, Red, NIR, SWIR]` |
| **4 bands** | `[Blue, Green, Red, NIR]` (SWIR derived from NIR fallback) |

### Spectral Stability Filter

Candidate patches are **NOT reported** if all three spectral delta magnitudes fall below the stability threshold:

```
Drop if:
  |ΔNDVI| < 0.002  AND
  |ΔNDWI| < 0.002  AND
  |ΔNDBI| < 0.002

→ Patches that are virtually spectrally identical (sensor noise only)
   are eliminated before analyst review
```

### Transition Taxonomy (PS 2.2.2 Tactical Categories)

The before → after land cover pair determines the tactical change category:

| Before → After | Tactical Category |
|---|---|
| Sparse Vegetation → Built-Up | **Construction / Infrastructure Expansion** |
| Dense Vegetation → Built-Up | **Land Clearance & Development** |
| Dense Vegetation → Sparse Vegetation | **Vegetation Disturbance / Deforestation** |
| Any → Water Body | **Water Extent Expansion / Flooding** |
| Water Body → Any | **Water Extent Contraction / Reclamation** |
| Bare Soil → Built-Up | **Construction on Cleared Land** |
| Built-Up → Dense Vegetation | **Demolition / Revegetation** |

---

## 8. Stage 6 — Analyst Review & Provenance Workflow

**Module**: `backend/services/analyst_workflow.py`

### Human-in-the-Loop Review

Every confirmed change candidate enters the analyst verification queue:

```
Analyst Actions per Candidate:
  ✅ CONFIRMED   → Physical real change, added to provenance export
  ❌ REJECTED    → False alarm, excluded from outputs
  ❓ UNSURE      → Flagged for secondary review
```

### Severity Scoring
```python
if total_candidates >= 6:
    severity_level = "CRITICAL"
    severity_score = min(99.0, max(88.0, 78.0 + total_candidates × 2.2))
elif total_candidates >= 2:
    severity_level = "HIGH"
    severity_score = min(87.9, max(75.0, 68.0 + total_candidates × 2.0))
elif total_candidates >= 1:
    severity_level = "MODERATE"
else:
    severity_level = "NONE"
```

### PostgreSQL Audit Tables

```sql
-- Change run registry
CREATE TABLE change_runs (
    run_id              UUID PRIMARY KEY,
    target_tile_id      TEXT,
    site_key            TEXT,
    region_id           TEXT,
    centroid_lat        REAL,
    centroid_lon        REAL,
    total_timeline_epochs INTEGER,
    available_dates     TEXT[],
    status              TEXT,        -- 'pending' | 'complete' | 'failed'
    created_at          TIMESTAMPTZ
);

-- Per-patch candidates
CREATE TABLE change_candidate_patches (
    candidate_id        UUID PRIMARY KEY,
    run_id              UUID REFERENCES change_runs(run_id),
    patch_id            INTEGER,
    z_score             REAL,
    distance            REAL,
    cluster_id          INTEGER,
    cluster_size        INTEGER,
    before_class        TEXT,
    after_class         TEXT,
    change_category     TEXT,
    delta_ndvi          REAL,
    delta_ndwi          REAL,
    delta_ndbi          REAL,
    bbox_pixels         INTEGER[],
    overlay_url         TEXT
);

-- Analyst decisions (append-only audit log)
CREATE TABLE analyst_decisions (
    decision_id         UUID PRIMARY KEY,
    run_id              UUID,
    candidate_id        UUID,
    analyst_id          TEXT,
    decision            TEXT,        -- 'confirmed' | 'rejected' | 'unsure'
    notes               TEXT,
    decided_at          TIMESTAMPTZ
);
```

### Export Formats

| Format | Description |
|---|---|
| **GeoJSON** | All confirmed candidates with centroid coordinates, spectral deltas, transition labels |
| **PDF SITREP** | `pdf_report_generator.py` — formatted intelligence report with satellite imagery, spectral charts, and analyst decisions |

---

## 9. Stage 7 — Temporal Backward Traversal (Date Estimation)

**Module**: `backend/services/temporal_splitter.py`

Once a change is confirmed, the system estimates **when the change first became visible** using bisect-style temporal traversal across all available epochs for that site:

```
Available dates: [2020-03-01, 2021-06-15, 2022-09-20, 2023-11-10, 2024-08-05]
Confirmed change between T_baseline (2020-03-01) and T_current (2024-08-05)

Algorithm:
  1. Binary bisect across epoch dates
  2. Run patch detection at mid-epoch
  3. If change detected at mid → traverse earlier half
  4. If change NOT detected → traverse later half
  5. Converge to earliest date at which change is first visible

Output: { "earliest_change_date": "2022-09-20", "confidence": 0.87 }
```

This satisfies PS 2.2.2: _"Estimate earliest available observation date at which change is visible via temporal backward traversal."_

---

## 10. PostgreSQL Schema Reference

All spatial data stored in **PostgreSQL 16 + PostGIS 3.4**:

```sql
-- Core tile registry (every ingested 512×512 patch)
CREATE TABLE tiles (
    tile_id             TEXT PRIMARY KEY,
    scene_id            TEXT,
    site_key            TEXT,           -- Geographic footprint identifier (groups epochs)
    sensor              TEXT,           -- 'Sentinel-2' | 'Maxar WorldView'
    acquisition_date    DATE,
    centroid_lat        REAL,
    centroid_lon        REAL,
    cloud_pct           REAL,
    quality_confidence  REAL,
    mean_ndvi           REAL,
    mean_ndwi           REAL,
    mean_ndbi           REAL,
    file_path           TEXT,
    bad_mask_path       TEXT,           -- Path to _mask.tif
    thumbnail_path      TEXT,
    geometry            GEOMETRY(POLYGON, 4326)   -- PostGIS bounding polygon
);

-- Ingestion coverage regions
CREATE TABLE ingestion_coverage (
    region_id           TEXT PRIMARY KEY,
    region_name         TEXT,
    status              TEXT,
    tile_count          INTEGER,
    last_updated        TIMESTAMPTZ,
    geometry            GEOMETRY(POLYGON, 4326)
);
```

---

## 11. API Endpoints

All routes under `POST /api/v1/change/`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/change/grid-map` | Fetch all regions + tiles for the change workbench canvas |
| `POST` | `/api/v1/change/stage` | Stage temporal series for a target tile → manifest.json |
| `POST` | `/api/v1/change/detect` | Run Prithvi-EO-2.0 patch detection on staged T1/T2 pair |
| `GET` | `/api/v1/change/runs` | List all change runs with review progress summary |
| `GET` | `/api/v1/change/runs/{run_id}/candidates` | Get all candidate patches for a run |
| `POST` | `/api/v1/change/decide` | Submit analyst decision (confirm/reject/unsure) |
| `GET` | `/api/v1/change/export/geojson/{run_id}` | Export confirmed changes as GeoJSON |
| `GET` | `/api/v1/change/export/pdf/{run_id}` | Generate PDF SITREP intelligence report |

---

## 12. End-to-End Data Flow

```
PostgreSQL (tiles + site_key grouping)
        │
        ▼  get_all_regions_grid_map()
┌───────────────────────────────────┐
│   change.html Leaflet Workbench   │
│   User selects target tile        │
└──────────────┬────────────────────┘
               │  POST /stage {tile_id}
               ▼
stage_tile_temporal_series()
  → QualityCheckEngine.audit_mask_file()  ← _mask.tif
  → STAY tiles copied to data/change_staging/
  → manifest.json written
               │
               ▼  POST /detect {staging_dir}
PrithviPatchChangeDetector.run_detection()
  ├── load_multiband_and_mask()      ← tile.tif + mask.tif
  ├── align_subpixel_misregistration()  ← phase correlation
  ├── 8×8 grid patch extraction
  ├── per-patch QA filter
  ├── encoder.encode_multiband_patches()  ← GPU Prithvi CUDA FP16
  ├── cosine_distance per patch
  ├── left_MAD Z-scoring
  ├── 8-neighbor connected components
  └── SemanticChangeClassifier.classify()  ← NDVI/NDWI/NDBI/BSI
               │
               ▼
PostgreSQL (change_runs + change_candidate_patches)
               │
               ▼  /review.html
analyst_workflow.py
  → confirm / reject / unsure
  → analyst_decisions audit log
               │
               ▼
Export: GeoJSON | PDF SITREP | Temporal Date Estimation
```

---

## 13. Configuration Reference

| Parameter | Default | File | Description |
|---|---|---|---|
| `max_bad_pixel_pct` | `15.0%` | `quality_auditor.py` | Tile-level cloud threshold |
| `min_usable_pct` | `85.0%` | `quality_auditor.py` | Min clean surface requirement |
| `grid_size` | `8` (8×8=64 patches) | `patch_change_detector.py` | Spatial patch grid resolution |
| `quality_thresh` | `0.20` (20%) | `patch_change_detector.py` | Per-patch cloud threshold |
| `z_threshold` | `2.0` | `patch_change_detector.py` | Minimum Left-MAD Z-score to flag candidate |
| `min_dist` | `0.01` | `patch_change_detector.py` | Minimum cosine distance threshold |
| `min_cluster_size` | `2` | `patch_change_detector.py` | Min connected patches to confirm change |
| `stability_threshold` | `0.002` | `semantic_change_classifier.py` | Min ΔNDVI/NDWI/NDBI to consider unstable |
| `upsample_factor` | `10` | `patch_change_detector.py` | Phase correlation sub-pixel resolution |
| `max_shift` | `3.0px` | `patch_change_detector.py` | Max acceptable orbital jitter correction |
| `beta` | `20.0` | `calibrator.py` | QB-Norm Inverted Softmax temperature |
| `hub_topk` | `5` | `calibrator.py` | Hubness detection top-K window |

---

> [!TIP]
> **Running a Change Detection End-to-End**
> ```bash
> # 1. Start the full stack
> docker compose up -d
>
> # 2. Navigate to the Change Detection workbench
> open http://localhost:8000/change.html
>
> # 3. Click any tile with epoch_count > 1, select T1 and T2, click "Detect Changes"
>
> # 4. Review candidates at /review.html
>
> # 5. Export confirmed changes
> curl http://localhost:8000/api/v1/change/export/geojson/{run_id}
> ```

> [!NOTE]
> All multi-spectral index computations, quality masks, and change scoring happen entirely **on-premises** with zero external API dependencies. The Prithvi-EO-2.0-300M model runs on local GPU (CUDA) or CPU fallback.
