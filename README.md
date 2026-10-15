# 🛰️ Canopus — Semantic Retrieval & Multi-Temporal Change Analysis of Satellite Imagery

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC2626?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![PostGIS](https://img.shields.io/badge/PostGIS-3.4-5B9BD5?style=for-the-badge&logo=postgis&logoColor=white)](https://postgis.net/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900?style=for-the-badge&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![GDAL](https://img.shields.io/badge/GDAL-Geospatial-489849?style=for-the-badge&logo=osgeo&logoColor=white)](https://gdal.org/)
[![MinIO](https://img.shields.io/badge/MinIO-S3%20Storage-C72C48?style=for-the-badge&logo=minio&logoColor=white)](https://min.io/)
[![RemoteCLIP](https://img.shields.io/badge/RemoteCLIP-ViT--B%2F32-8A2BE2?style=for-the-badge)](https://huggingface.co/chendelong/RemoteCLIP)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-10m%20Copernicus-003399?style=for-the-badge)](https://sentinel.esa.int/)
[![SARCLIP](https://img.shields.io/badge/SARCLIP-ViT--L%2F14%20768D-B91C1C?style=for-the-badge)](https://huggingface.co/)
[![YOLOv8-OBB](https://img.shields.io/badge/YOLOv8--OBB-Maritime%20ATR-FF6B00?style=for-the-badge)](https://ultralytics.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Autonomous%20Agent-6D28D9?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google)](https://ai.google.dev/)

**AI-Powered Earth Observation Intelligence Platform — Multimodal Semantic Retrieval, Multi-Temporal Change Detection & Autonomous SAR Defence Reconnaissance**

[Key Features](#-key-features-completed) • [SAR Intelligence](#️-sar-intelligence-module-defence) • [Quickstart Guide](#️-quickstart--execution-guide-docker) • [Service Endpoints](#-service-access-endpoints) • [Repository Structure](#-repository-directory-structure) • [API Contract](#-api-specification)

</div>

---

## Executive Summary

Modern Earth Observation (EO) archives capture terabytes of satellite imagery daily across diverse orbits, resolutions, and sensor modalities. However, extracting actionable intelligence from these massive multi-temporal archives presents two core operational bottlenecks:

1. **The Semantic Discovery Bottleneck**: Traditional catalog search systems are restricted to rigid metadata filters — querying exclusively by geographic bounding coordinates, sensor IDs, and capture timestamps. Analysts cannot search archives by **semantic meaning** or visual intent (e.g., querying *"airfield runways with hangars near water"*, *"cargo vessels docked at berths"*, *"new military encampments"*, or *"vegetated river valleys"*).
2. **The Multi-Temporal Change Bottleneck**: Identifying landscape transformations across temporal epochs ($T_1$ vs. $T_2$ across multiple years) traditionally requires intensive manual photo-interpretation. Furthermore, automated change detection is notoriously plagued by **high false-alarm rates** caused by ephemeral atmospheric interference — including cloud contamination, cloud shadow projection, seasonal vegetation swings, off-nadir sun angle variations, and sub-pixel spatial misalignments.

### The Canopus Platform Vision

**Canopus** is an end-to-end, sovereign platform built to solve **Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery** (Problem Statement SIH-26227). The system unifies natural language zero-shot retrieval, multi-spectral change intelligence, and unsupervised pattern discovery into a streamlined analytical workflow:

* **Natural Language & Visual Semantic Retrieval**: Powered by fine-tuned vision-language foundation models (RemoteCLIP ViT-B-32) and high-dimensional vector indexing (Qdrant), analysts can search planetary archives using plain conversational text prompts, exemplar satellite image patches, or multimodal queries with sub-second response times.
* **Multi-Temporal Change Analysis & Verification**: Tracks structural, infrastructural, and environmental evolution across multi-year temporal intervals. It isolates real physical changes (construction, earthworks, road development, water boundary shifts), estimates the earliest emergence date via temporal traversal, and provides an analyst review queue with exportable provenance.
* **Atmospheric Calibration & False-Positive Suppression**: Preprocessing pipelines eliminate false positives through automated `s2cloudless` machine learning cloud detection, directional shadow ray-tracing, and mask-aware radiometric percentile contrast stretching — preserving physical surface reflectance.
* **Multi-Spectral & Multi-Sensor Intelligence**: Evaluates and fuses multi-spectral band signatures (NDVI, NDWI, NDBI) from Sentinel-2 L2A with high-resolution optical reconnaissance imagery from Maxar WorldView, generating rule-based terrain assessments.
* **Unsupervised Clustering & Cross-Site Discovery**: Employs density-based vector clustering (**HDBSCAN**) to partition the high-dimensional latent space without arbitrary cluster count assumptions, isolating representative medoids and enabling one-click cross-site similarity discovery across distinct geographic sectors.
* **100% On-Premises & Incrementally Scalable**: Operates entirely within air-gapped, network-isolated environments with zero external API dependencies. Newly ingested imagery is indexed incrementally into spatial and vector tables without requiring full archive re-computation.

---

## 🚀 Key Features (Completed)

### Phase 1 — Ingestion & Multi-Spectral Preprocessing Pipeline
* **Dual Ingestion Entry Points**:
  - **AOI Search & Ingestion**: Users can draw interactive bounding polygons on a Leaflet satellite map or upload GeoJSON boundaries, specify multi-temporal date intervals, and trigger automated catalog queries.
  - **Offline Raster Ingestion**: Direct ingestion of local georeferenced GeoTIFF files with custom multi-spectral band mappings without external network requirements.
* **Multi-Sensor Satellite Support**:
  - **Sentinel-2 L2A**: Multi-spectral imagery streaming (Blue `B02`, Green `B03`, Red `B04`, NIR `B08`, SWIR `B11`, and 10m True Color `TCI`).
  - **Maxar WorldView / Wayback**: High-resolution optical reconnaissance imagery integration with multi-year temporal alignment.
* **Automated Cloud & Directional Shadow Masking**:
  - Integrates `s2cloudless` gradient-boosted tree classification and directional shadow ray-casting to generate pixel-level validity masks (`bad_mask`).
* **Mask-Aware Radiometric Normalization**:
  - Applies 0.5%–99.5% dynamic percentile contrast stretching strictly over clean ground pixels, preserving radiometric fidelity and multi-spectral index validity.
* **Equidistant Non-Overlapping 512×512 Tiling**:
  - Partitions large rasters into uniform 512×512 patches with boundary padding and georeferenced bounding boxes (`_get_grid_steps()`).
* **Multi-Spectral Index Extraction**:
  - **NDVI** (Normalized Difference Vegetation Index): $(\text{NIR} - \text{Red}) / (\text{NIR} + \text{Red})$
  - **NDWI** (Normalized Difference Water Index): $(\text{Green} - \text{NIR}) / (\text{Green} + \text{NIR})$
  - **NDBI** (Normalized Difference Built-Up Index): $(\text{SWIR} - \text{NIR}) / (\text{SWIR} + \text{NIR})$
  - **VARI** (Visible Atmospherically Resistant Index): $(\text{Green} - \text{Red}) / (\text{Green} + \text{Red} - \text{Blue})$ for optical RGB rasters.
* **Atomic Dual-Storage Persistence**:
  - Persists spatial polygons and index metadata to **PostgreSQL 16 + PostGIS 3.4**, and saves GeoTIFF files, visual thumbnails, and JSON manifests to disk and **MinIO S3** storage.

---

### Phase 2 — Multimodal Semantic Retrieval & Intelligent Chat System
* **Fine-Tuned RemoteCLIP Vision-Language Embedding Engine**:
  - Embeds satellite tiles into a shared 512-dimensional vector space using RemoteCLIP ViT-B-32 with prompt template ensembling (`"satellite imagery of {prompt}"`, `"aerial view of {prompt}"`).
* **Dual Query Modalities**:
  - **Natural Language Text Queries**: Search using unstructured descriptive text prompts.
  - **Visual Similarity Search (Image-to-Image)**: Upload any satellite image patch to retrieve visually and contextually similar ground locations across the archive.
* **Persistent ChatGPT/Gemini-Style Conversational Interface**:
  - Multi-turn conversation sessions with persistent history saved to PostgreSQL (`chat_conversations` and `chat_messages`).
  - Collapsible history sidebar with search sessions, chat renaming, deletion, and seamless conversation switching.
  - User session isolation ensuring independent analyst workspaces.
* **Compound Vector Filtering & Real-Time Parameter Sliders**:
  - Pre-filters vectors by **Sensor** (`Sentinel-2`, `Maxar WorldView`, or `All`), **Date Range**, and **Spatial Bounding Box**.
  - Interactive UI sliders for **Confidence Threshold** ($0.0$ to $1.0$) and **Cloud Threshold** with live updates without re-triggering full page reloads.
* **Spatial Deduplication & Multi-Spectral Ground Truth Explanations**:
  - Enforces spatial separation ($\sim 200\text{m}$) across Top-K results to prevent clustering around a single image frame.
  - Synthesizes rule-based spectral explanations combining $\text{NDVI} \times \text{NDWI} \times \text{NDBI}$ profiles.
* **Tile Inspection & Lineage Modal**:
  - Click-to-inspect modal displaying high-res visual preview, dynamic spectral meter bars, coordinates, sensor metadata, and one-click GeoTIFF download.

---

### Phase 3 — Unsupervised HDBSCAN Clustering, Medoid Extraction & Cross-Site Discovery
* **Unsupervised Density-Based Vector Clustering (HDBSCAN)**:
  - Groups 512-dimensional RemoteCLIP embeddings across the entire archive using **HDBSCAN** (`sklearn.cluster.HDBSCAN`).
  - Automatically identifies natural cluster boundaries based on vector density without requiring pre-set or arbitrary cluster counts ($K$).
* **Outlier & Noise Rejection**:
  - Flags atypical or anomalous tiles as noise points (`-1`), preventing distortion of cohesive cluster centroids.
* **Representative Medoid Extraction**:
  - Computes the geometric and mathematical **medoid** (the member tile closest to the cluster center) to serve as the visual and semantic representative for each cluster.
* **Automated Multi-Spectral Cluster Labeling**:
  - Analyzes aggregated NDVI, NDWI, and NDBI distributions across member tiles to assign human-interpretable terrain classifications (e.g., *Vegetative & Agricultural Fields*, *Urban Core & Infrastructure*, *Open Arid & Transition Terrain*).
* **Interactive Clustering Workbench (`/clustering.html`)**:
  - Executive metric HUD cards (Total Discovered Clusters, Clustered Archive Tiles, Embedding Dimension, HDBSCAN Cluster Engine).
  - Categorical filter pills (All Clusters, Vegetative & Fields, Water & Coastal, Urban & Built-Up, Arid & Transition) with real-time text search.
  - Cluster cards showing synchronized cluster badges, member tile counts, medoids, and direct "Pin on Map" actions.
* **Deep Member Tiles Drill-Down**:
  - Explores all member tiles within a selected cluster, highlighting the active medoid and displaying spectral indices.
* **Cross-Site "Find Similar" Discovery Engine**:
  - Allows users to branch off from any tile (within search results, inspection modals, or cluster member drill-down) by clicking **"Find Similar"** to discover visually and semantically related sites across all ingested regions.

---

## 🛡️ SAR Intelligence Module (Defence)

The **SARIntelligence** module is the dedicated defence-grade SAR (Synthetic Aperture Radar) reconnaissance sub-system built for the Ministry of Defence / Indian Army problem statement. It operates as a fully autonomous intelligence pipeline alongside the optical EO platform.

### What It Does

| Capability | Description |
|---|---|
| **Maritime Vessel Detection** | YOLOv8x-OBB detects and classifies 6 warship / merchant vessel classes with Oriented Bounding Boxes, calibrated RCS (dBsm), and wake velocity kinematics |
| **Ground Military ATR** | MSTAR-trained 10-class Automatic Target Recognition for MBTs (T-72, T-62), IFVs (BMP-2), APCs (BTR-70), artillery, anti-aircraft guns and logistics vehicles |
| **Military Activity Scoring** | Composite MAS (0.0–1.0) + DEFCON 1–5 tier classification derived from target count, RCS totals, temporal surge deltas and domain weights |
| **Temporal Surge Analysis** | Sub-pixel phase-correlated T1 vs T2 comparison with pixel-level delta heatmaps and escalation ranking across all indexed scenes |
| **768-D SARCLIP Semantic Search** | SAR-domain ViT-L-14 vision-language encoder enabling natural language AND image-to-image search across the entire SAR archive via Qdrant |
| **Autonomous Agent (LangGraph)** | Multi-step ReAct agent with 4 tactical tools — SQL lookup, vector search, temporal surge, and full telemetry retrieval |
| **Pluggable LLM Brain** | Gemini 2.5 Flash / Groq LLaMA-3.3-70B / OpenAI-compatible (Qwen, vLLM) / Sovereign Air-Gapped engine for classified deployments |
| **Military SITREP Output** | Structured intelligence reports with GPS coordinates, OBB detections, threat classification, and commander's assessment |

### Defence-Specific Design Choices

- **Zero External Dependencies in Sovereign Mode**: `SOVEREIGN_MODE=true` disables all external LLM/API calls — the system uses a deterministic rule-based planner and purely local Qdrant + SQLite for fully air-gapped classified deployment.
- **Calibrated Radar Physics**: All detections report physically calibrated metrics — Sigma-0 (σ⁰) backscatter in dB, RCS in dBsm, vessel beam/length in metres — enabling analyst verification without black-box outputs.
- **Sub-Pixel Co-Registration**: Phase correlation alignment of T1/T2 temporal pairs eliminates false change alarms from spatial misalignment — critical for detecting genuine military build-up vs. satellite orbit drift artefacts.
- **DEFCON Tier Automation**: Every ingested scene is automatically classified into DEFCON 1–5 based on a composite Military Activity Score — no manual threshold tuning required.

### SAR Module Entry Points

```bash
# Run the full SAR ingestion pipeline (preprocess → detect → embed → index)
python -m SARIntelligence.pipeline.ingestion_runner

# Run the autonomous agent with a tactical query
python -c "
from SARIntelligence.agent.langgraph_orchestrator import SARLangGraphAgent
agent = SARLangGraphAgent(llm_provider='gemini')
result = agent.run('Identify all DEFCON-2 or higher maritime scenes in the South China Sea')
print(result)
"
```

👉 **[Full SAR Implementation Deep-Dive (`sar_implementation.md`)](sar_implementation.md)**

---

## 🛠️ Quickstart & Execution Guide (Docker)

The entire application stack (FastAPI Backend, Leaflet Web Applications, PostgreSQL/PostGIS, Qdrant Vector DB, and MinIO S3) is containerized and launches with a single command.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or [Docker Engine](https://docs.docker.com/engine/install/) + Docker Compose v2 (Linux)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/varun-ai69/SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-.git
cd SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-
```

### 2. Launch the Stack (Single Command)
Run Docker Compose from the root directory:
```bash
docker compose up -d --build
```
> **Zero-Friction Auto-Provisioning**:
> - **Databases**: PostgreSQL 16 + PostGIS 3.4 automatically initializes all schemas and migrations (`schema.sql`, `change_*` detection tables, `analyst_decisions`, `search_feedback`).
> - **Vector Search**: Qdrant vector engine initializes and mounts high-dimensional collections.
> - **AI Models**: Foundation models (`RemoteCLIP ViT-B-32` and `Prithvi-EO-2.0-300M`) automatically download from Hugging Face on first execution if not already cached locally.
> - **Web UI & API**: Open **[http://localhost:8000](http://localhost:8000)** in your browser!

### 4. Run Automated Test Suite
To verify the complete test suite inside the running backend container:
```bash
docker exec -it eo_backend pytest tests/ -v
```

---

## 🌐 Service Access Endpoints

Once the stack is running, all services and interactive user interfaces are accessible at the following endpoints:

| Service / Interface | Access URL | Port | Purpose |
|---|---|---|---|
| **Map & Ingestion Workspace** | [http://localhost:8000](http://localhost:8000) | `8000` | Global Satellite Map, AOI Polygon Drawing & Ingestion |
| **Semantic Retrieval & Chat Interface** | [http://localhost:8000/retrieval.html](http://localhost:8000/retrieval.html) | `8000` | Multimodal Semantic Search & ChatGPT/Gemini Chat Sessions |
| **Multi-Temporal Change Detection** | [http://localhost:8000/change.html](http://localhost:8000/change.html) | `8000` | Bi-temporal $T_1$ vs $T_2$ Change Analysis & Spectral Deltas |
| **HDBSCAN Clustering Workbench** | [http://localhost:8000/clustering.html](http://localhost:8000/clustering.html) | `8000` | Unsupervised Terrain Discovery, Medoids & Cluster Drill-Down |
| **Analyst Review & Verification Tool** | [http://localhost:8000/review.html](http://localhost:8000/review.html) | `8000` | Analyst Verification Queue & Quality Audit Feeds |
| **Interactive API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | `8000` | Swagger / OpenAPI Interactive REST API Documentation |
| **Qdrant Vector DB Dashboard** | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | `6333` | Vector collection browser, points visualizer & distance metrics |
| **MinIO S3 Storage Console** | [http://localhost:9001](http://localhost:9001) | `9001` | Object Storage Console (User: `eo_admin`, Password: `eo_password`) |
| **PostgreSQL / PostGIS Database** | `localhost:5434` | `5434` (mapped from `5432`) | Spatial Database (DB: `eo_archive`, User: `eo_admin`, Pass: `eo_password`) |

---

## 📁 Repository Directory Structure

For a full breakdown of all subdirectories, modules, and component roles, please refer to the dedicated architecture document:

👉 **[Detailed Repository Directory Structure (`docs/repository_structure.md`)](docs/repository_structure.md)**

```text
├── SARIntelligence/    # 🛡️ Defence SAR Intelligence — autonomous ATR, agent, LLM, SARCLIP
│   ├── agent/          #    LangGraph orchestrator, LLM providers, 4 tactical tools
│   ├── atr_models/     #    YOLOv8-OBB maritime, MSTAR 10-class ground ATR, CFAR
│   ├── embeddings/     #    SARCLIP ViT-L-14 768-D encoder + Qdrant client
│   ├── pipeline/       #    Ingestion runner, 6-layer template builder, MAS scoring
│   └── preprocessing/  #    DN norm, Sigma-0 cal, phase co-registration, RGB composite
├── backend/            # FastAPI REST API, ingestion engine, HDBSCAN jobs & RemoteCLIP encoder
├── frontend/           # Glassmorphic aerospace web interfaces (Map, Retrieval, Change, Clustering, Review)
├── infra/              # Dockerfile, entrypoint.sh, and infrastructure Compose configurations
├── models/             # Pre-trained deep learning checkpoints (RemoteCLIP ViT-B-32)
├── data/               # Persistent tile storage, GeoTIFFs, visual previews, and GeoJSON AOIs
├── docs/               # Architecture records, API contracts, and directory guides
└── tests/              # End-to-end and unit test suites
```

---

## 📖 API Specification

The REST API exposes modular endpoints under `/api/v1` for ingestion, semantic vector retrieval, persistent conversation sessions, HDBSCAN clustering, and spatial coverage.

For complete route definitions, request/response JSON schemas, query parameters, and status codes, see:

👉 **[Complete REST API Contract (`docs/api_contract.md`)](docs/api_contract.md)**

---

## 🎯 Problem Statement & Project Context

This platform addresses **SIH-2026 PS26227** — Semantic Retrieval & Multi-Temporal Analysis of Satellite Imagery (Ministry of Defence / Indian Army).

The system solves two core operational bottlenecks in modern Earth Observation:
1. **Semantic Discovery** — Natural language and visual search over massive satellite archives
2. **Multi-Temporal Change** — Automated, false-alarm-suppressed change detection across multi-year epochs
3. **SAR Defence Reconnaissance** — Autonomous military ATR, threat scoring, and SITREP generation from SAR imagery

For full SAR implementation details including ingestion pipeline, ATR models, agent architecture, and LLM layer:

👉 **[SAR Intelligence Implementation Deep-Dive (`sar_implementation.md`)](sar_implementation.md)**

For complete API contracts and route definitions:

👉 **[REST API Contract (`docs/api_contract.md`)](docs/api_contract.md)**
