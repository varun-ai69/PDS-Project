# 🛡️ SARIntelligence — Defence SAR Implementation Deep-Dive

> **Problem Statement**: SIH-2026 PS26227 — Semantic Retrieval & Multi-Temporal Analysis of Satellite Imagery (Ministry of Defence / Indian Army)
>
> This document covers the complete end-to-end Synthetic Aperture Radar (SAR) intelligence pipeline implemented inside `SARIntelligence/`. The system is purpose-built for **defence reconnaissance**, enabling autonomous analysis of military SAR satellite imagery for threat detection, vessel tracking, ground vehicle identification, and temporal surge analysis.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1 — SAR Ingestion & Preprocessing Pipeline](#2-phase-1--sar-ingestion--preprocessing-pipeline)
3. [Phase 2 — ATR Models & Domain-Specific Detection](#3-phase-2--atr-models--domain-specific-detection)
4. [Phase 3 — Master Template & Military Activity Scoring](#4-phase-3--master-template--military-activity-scoring)
5. [Phase 4 — Embedding & Vector Indexing (SARCLIP)](#5-phase-4--embedding--vector-indexing-sarclip)
6. [Phase 5 — Autonomous Agent Layer (LangGraph)](#6-phase-5--autonomous-agent-layer-langgraph)
7. [Phase 6 — Pluggable LLM Layer](#7-phase-6--pluggable-llm-layer)
8. [Directory Reference](#8-directory-reference)
9. [Data Flow Diagram](#9-data-flow-diagram)

---

## 1. Architecture Overview

```
RAW SAR GeoTIFF Input
        │
        ▼
┌─────────────────────────────────────┐
│  PHASE 1: Preprocessing Pipeline    │  ← SARIntelligence/preprocessing/
│  DN Normalization → Sigma-0 Cal     │
│  Band Registration → RGB Composite  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 2: Domain Router + ATR       │  ← SARIntelligence/preprocessing/domain_router.py
│  OCEAN → YOLOv8-OBB Maritime Det.  │  ← SARIntelligence/atr_models/maritime_detector.py
│  GROUND → MSTAR 10-Class ATR       │  ← SARIntelligence/atr_models/ground_atr_classifier.py
│  OCEAN (raw) → CFAR Detector       │  ← SARIntelligence/atr_models/cfar_detector.py
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 3: Master Template Builder   │  ← SARIntelligence/pipeline/template_builder.py
│  6-Layer JSON Schema Generation     │
│  Military Activity Score (MAS)      │  ← SARIntelligence/pipeline/military_activity_score.py
│  DEFCON Tier Classification         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 4: SARCLIP Embedding + Index │  ← SARIntelligence/embeddings/sarclip_service.py
│  768-D ViT-L-14 Vector Encoding     │
│  Qdrant Local Vector DB + SQLite    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 5: Autonomous Agent Layer    │  ← SARIntelligence/agent/langgraph_orchestrator.py
│  LangGraph State Machine            │
│  4 Tactical Intelligence Tools      │  ← SARIntelligence/agent/tactical_tools.py
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PHASE 6: Pluggable LLM Brain       │  ← SARIntelligence/agent/llm_interface.py
│  Gemini / Groq / Qwen / Air-Gapped  │
│  ReAct-style SITREP Synthesis       │
└─────────────────────────────────────┘
```

---

## 2. Phase 1 — SAR Ingestion & Preprocessing Pipeline

### Entry Point
**`SARIntelligence/pipeline/ingestion_runner.py`** — Master Ingestion & Indexing Orchestrator

The ingestion runner is the top-level orchestrator. It:
- Iterates over all processed SAR scenes in `SARIntelligence/data/processed/`
- Executes the full detection pipeline per scene via `MasterTemplateBuilder`
- Applies domain routing (OCEAN vs GROUND)
- Computes the Military Activity Score (MAS)
- Persists 6-Layer Master Template JSON schemas to `SARIntelligence/data/templates/`
- Upserts 768-D SARCLIP vectors into Qdrant and SQLite

### SQLite Intelligence Database Schema

```sql
CREATE TABLE IF NOT EXISTS scenes (
    scene_id        TEXT PRIMARY KEY,
    scene_name      TEXT,
    domain          TEXT,
    sector          TEXT,
    tactical_category TEXT,
    lat             REAL,
    lon             REAL,
    mas_score       REAL,
    defcon_level    INTEGER,
    defcon_tier     TEXT,
    threat_level    TEXT,
    total_targets   INTEGER,
    primary_classes TEXT,
    sitrep_text     TEXT,
    surge_annotation TEXT,
    json_path       TEXT,
    overlay_png     TEXT
);
```

---

### 2a. Preprocessing Sub-Pipeline

**Module**: `SARIntelligence/preprocessing/pipeline.py` — `SARUniversalPreprocessor`

#### Stage 1 — Digital Number (DN) Normalization
```
Raw SAR DN values → Percentile Stretch (0.5%–99.5%) → [0,255] uint8
```
- Eliminates sensor-specific gain offsets
- Preserves authentic radar texture and speckle statistics
- **File**: `preprocessing/radiometric_calibration.py → normalize_dn()`

#### Stage 2 — Radiometric Calibration (Sigma-0)
```
DN → Sigma-0 Linear Power → Sigma-0 dB
σ⁰_dB = 10 × log₁₀(σ⁰_linear + ε)
```
- Converts digital numbers to absolute physical radar backscatter coefficients
- Critical for calibrated RCS estimation in target classification
- **File**: `preprocessing/radiometric_calibration.py → calibrate_digital_numbers_to_sigma0_db()`

#### Stage 3 — Band & Multi-Temporal Sub-Pixel Co-Registration
```
Multi-Band SAR → Phase Correlation → Co-registered bands
Temporal T1, T2 → Phase Correlation → Pixel-accurate co-registration
```
- FFT-based frequency domain phase correlation for sub-pixel accuracy
- Eliminates false alarms from temporal misalignment
- **File**: `preprocessing/band_registration.py → register_bands(), co_register_temporal_pair()`

#### Stage 4 — RGB Polarimetric Composite Generation
```
VV + VH → Dual-Pol RGB (R=VV, G=VH, B=VV-VH)
Single-Band → Tactical Grayscale RGB (3-channel)
T1 + T2 → Temporal Change RGB (R=T2, G=T1, B=Δmag)
```
- **File**: `preprocessing/rgb_compositor.py`

#### Domain Router
- **File**: `preprocessing/domain_router.py`
- Classifies scene into `OCEAN` or `GROUND` via metadata tags + image statistics
- Routes to corresponding ATR detection model

---

## 3. Phase 2 — ATR Models & Domain-Specific Detection

### 3a. Maritime Domain — SOTA YOLOv8-OBB Vessel Detector
**File**: `SARIntelligence/atr_models/maritime_detector.py`

Uses **YOLOv8x-OBB** (Oriented Bounding Box) for maritime vessel detection.

#### Vessel Classes
| Class | Description |
|---|---|
| `AIRCRAFT_CARRIER` | Capital warship with flight deck |
| `DESTROYER_FRIGATE` | Guided missile surface combatant |
| `CORVETTE_PATROL` | Fast attack missile corvette |
| `SUBMARINE_SURFACED` | Surfaced submarine combatant |
| `CARGO_VESSEL` | Commercial container / cargo ship |
| `TANKER` | Fleet replenishment oil tanker |
| `NAVAL_WARSHIP` | Generic naval warship |

#### Detection Output
```json
{
  "class": "DESTROYER_FRIGATE",
  "confidence": 0.94,
  "obb": [cx, cy, width, height, angle_degrees],
  "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "length_m": 155.0,
  "beam_m": 18.0,
  "rcs_dbsm": 32.4,
  "wake_velocity_kt": 14.2
}
```

### 3b. Ground Domain — MSTAR 10-Class ATR
**File**: `SARIntelligence/atr_models/ground_atr_classifier.py`

| Class | Platform |
|---|---|
| `T72` | Main Battle Tank |
| `T62` | Soviet-era MBT |
| `2S1` | Self-Propelled Artillery |
| `BMP2` | Infantry Fighting Vehicle |
| `BTR70` | Armoured Personnel Carrier |
| `BRDM2` | Armoured Reconnaissance Vehicle |
| `ZSU234` | Shilka Anti-Aircraft Gun |
| `D7` | Military Bulldozer |
| `ZIL131` | Military Utility Truck |
| `SLICY` | Radar Calibration Target |

### 3c. CFAR Detector
**File**: `SARIntelligence/atr_models/cfar_detector.py`
- Constant False Alarm Rate sliding-window detector
- Used as pre-filter and fallback for low-confidence deep ATR detections

---

## 4. Phase 3 — Master Template & Military Activity Scoring

### 6-Layer Master Template JSON
**File**: `SARIntelligence/pipeline/template_builder.py`

```
Layer 1: Scene Metadata     → scene_id, lat/lon, timestamp, sensor, domain
Layer 2: Preprocessing Stats → sigma0_mean_db, speckle_index, snr_db
Layer 3: Detection Records   → all targets with OBB + class + RCS + confidence
Layer 4: Military Activity   → MAS score, DEFCON level, threat assessment
Layer 5: Temporal Analysis   → T1/T2 deltas, surge_annotation, activity_delta
Layer 6: SARCLIP Embedding   → 768-D float32 vector for Qdrant indexing
```

### Military Activity Score (MAS)
**File**: `SARIntelligence/pipeline/military_activity_score.py`

```
MAS = w₁×(N_combat/N_max) + w₂×(Σ RCS/RCS_ref) + w₃×(surge_delta) + w₄×(domain_weight)
```

#### DEFCON Classification
| MAS Range | DEFCON | Tier |
|---|---|---|
| 0.85–1.00 | 1 | MAXIMUM READINESS — Imminent Threat |
| 0.65–0.84 | 2 | HIGH ALERT — Elevated Military Activity |
| 0.40–0.64 | 3 | ELEVATED — Moderate Observed Activity |
| 0.15–0.39 | 4 | GUARDED — Low Activity Nominal |
| 0.00–0.14 | 5 | NORMAL — No Significant Threat |

---

## 5. Phase 4 — Embedding & Vector Indexing (SARCLIP)

**File**: `SARIntelligence/embeddings/sarclip_service.py`

- **Backbone**: OpenCLIP `ViT-L-14` → **768-dimensional** joint embedding space
- **SAR Adaptation**: Prompt templates tuned for radar vocabulary
- **Collection**: `sarintelligence_tactical_vectors` in Qdrant local
- **Storage**: `SARIntelligence/data/qdrant_local/` (embedded, no server needed)
- **Dual Persistence**: Qdrant (vector search) + SQLite (structured queries)

---

## 6. Phase 5 — Autonomous Agent Layer (LangGraph)

**File**: `SARIntelligence/agent/langgraph_orchestrator.py`

### State Graph
```
[START] → [Planner] → [CALL_TOOL] → [Tool Executor] → [Evaluator] ─┐
                    ↓                                               │
               [Synthesizer] ◄──────────────────────────────────────┘
                    ↓
                  [END]
```

### 4 Tactical Tools (`SARIntelligence/agent/tactical_tools.py`)

| Tool | Type | Description |
|---|---|---|
| `query_database` | SQL | Structured SQLite queries: domain filter, spatial corridor, DEFCON filter |
| `semantic_similarity_search` | Vector | 768-D SARCLIP text/image search via Qdrant |
| `analyze_temporal_surge` | Analysis | T1 vs T2 delta maps, escalation ranking, heatmaps |
| `get_scene_telemetry` | Telemetry | OBB, RCS dBsm, GPS, full 6-layer Master Template |

---

## 7. Phase 6 — Pluggable LLM Layer

**File**: `SARIntelligence/agent/llm_interface.py`

| Provider | Class | Notes |
|---|---|---|
| Google Gemini | `GeminiProvider` | `gemini-2.5-flash` primary |
| Groq | `GroqProvider` | LLaMA-3.3-70B, ultra-low latency |
| OpenAI-Compatible | `OpenAICompatibleProvider` | Qwen, DeepSeek, vLLM, Ollama |
| Sovereign Air-Gapped | `SovereignAirGappedProvider` | 100% offline, zero external calls |

Set `SOVEREIGN_MODE=true` in `.env` to force the air-gapped engine on classified servers.

### SITREP Output Format
```
CLASSIFICATION: [UNCLASSIFIED / RESTRICTED / CONFIDENTIAL]
SITREP — [SUBJECT]  DTG: [DD HHmmZ MON YY]

1. SITUATION SUMMARY
2. DETECTED ASSETS   (Class | Confidence | RCS dBsm | GPS)
3. THREAT ASSESSMENT (MAS | DEFCON | Threat Level)
4. TEMPORAL INTELLIGENCE (Surge deltas, timeline)
5. COMMANDER'S ASSESSMENT
```

---

## 8. Directory Reference

```
SARIntelligence/
├── agent/
│   ├── langgraph_orchestrator.py   # LangGraph ReAct state machine
│   ├── llm_interface.py            # Pluggable LLM abstraction layer
│   ├── tactical_tools.py           # 4 core intelligence tools
│   └── react_orchestrator.py       # Lightweight ReAct fallback
│
├── atr_models/
│   ├── maritime_detector.py        # YOLOv8x-OBB vessel detector (6 classes)
│   ├── ground_atr_classifier.py    # MSTAR 10-class ground vehicle ATR
│   ├── cfar_detector.py            # CFAR statistical anomaly detector
│   ├── train_atr_lora.py           # LoRA fine-tuning adapter
│   └── train_sota_maritime_model.py
│
├── embeddings/
│   └── sarclip_service.py          # SARCLIP ViT-L-14 768-D encoder
│
├── pipeline/
│   ├── ingestion_runner.py         # Master ingestion orchestrator
│   ├── template_builder.py         # 6-Layer Master Template builder
│   └── military_activity_score.py  # MAS scoring + DEFCON classification
│
├── preprocessing/
│   ├── pipeline.py                 # SARUniversalPreprocessor
│   ├── radiometric_calibration.py  # DN → Sigma-0 calibration
│   ├── band_registration.py        # Phase correlation co-registration
│   ├── rgb_compositor.py           # Dual-pol / temporal RGB composites
│   ├── despeckle_filter.py         # Speckle reduction
│   ├── domain_router.py            # OCEAN / GROUND classifier
│   └── dualpol_compositor.py       # Dual-polarisation compositor
│
├── data/
│   ├── sarintelligence.db          # SQLite intelligence database
│   ├── qdrant_local/               # Qdrant embedded vector store
│   ├── templates/                  # 6-Layer Master Template JSONs
│   ├── processed/                  # Calibrated GeoTIFFs
│   ├── detections/                 # ATR detection overlay PNGs
│   ├── temporal_diffs/             # Temporal delta heatmaps
│   ├── previews/                   # RGB thumbnails
│   └── generators/                 # Synthetic SAR dataset generators
│
└── weights/
    └── maritime/
        ├── yolov8x-obb.pt          # Primary maritime detector weights
        └── yolov8m-obb.pt          # Fallback maritime detector weights
```

---

## 9. Data Flow Diagram

```
INPUT: Sentinel-1 GRD / ERS / Synthetic MSTAR GeoTIFFs
        │
        ▼
┌─────────────────────────────────────────────────┐
│  SARUniversalPreprocessor                       │
│  DN Normalize → Sigma-0 Cal → Band Reg → RGB   │
└─────────────────────┬───────────────────────────┘
                      │
              ┌───────▼────────┐
              │  Domain Router  │
              └───┬────────┬───┘
                  │        │
        ┌─────────▼──┐  ┌──▼──────────────┐
        │   OCEAN    │  │     GROUND       │
        │ YOLOv8-OBB │  │  MSTAR 10-Class  │
        │  + CFAR    │  │   ATR Classify   │
        └─────┬──────┘  └──────┬───────────┘
              └──────┬─────────┘
                     │ Detection Records
                     ▼
        ┌────────────────────────────────┐
        │    Master Template Builder     │
        │  6-Layer JSON + MAS + DEFCON   │
        └──────────────┬─────────────────┘
                       │
              ┌────────▼────────┐
              │  SARCLIP Encode  │
              │  ViT-L-14 768-D  │
              └──────┬─────┬────┘
                     │     │
              ┌──────▼──┐ ┌▼─────────────┐
              │ SQLite   │ │ Qdrant Local │
              │ Intel DB │ │ Vector Index │
              └──────┬───┘ └──────┬───────┘
                     └─────┬──────┘
                           │
        ┌──────────────────▼──────────────────────┐
        │       LangGraph Autonomous Agent         │
        │  Planner → Tool Dispatch → Evaluator     │
        │                                          │
        │  query_database          (SQL)           │
        │  semantic_similarity_search (Vector)     │
        │  analyze_temporal_surge  (Delta)         │
        │  get_scene_telemetry     (Telemetry)     │
        │                                          │
        │  → Synthesizer (LLM)                     │
        └──────────────────┬──────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │  Pluggable LLM Brain      │
              │  Gemini / Groq / Qwen /   │
              │  Sovereign Air-Gapped     │
              └────────────┬─────────────┘
                           │
                           ▼
               MILITARY SITREP OUTPUT
```

---

> **Classification Note**: This system is designed for sovereign defence use. All SARIntelligence data stays on-premises. The `SovereignAirGappedProvider` ensures zero external communication in classified environments.
