# SETUP & DEPLOYMENT GUIDE
## CANOPUS - AI-Powered Earth Observation & SAR Defence Intelligence Platform
### SIH-2026 PS26227 - Ministry of Defence / Indian Army

> **This is the complete guide to get the entire platform running from zero.**
> Everything runs via Docker - no local Python environment needed except for optional GPU SAR inference.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Directory Structure](#2-clone--directory-structure)
3. [Environment Configuration](#3-environment-configuration)
4. [Model Weights - Download Links](#4-model-weights--download-links)
5. [Docker Setup (Full Stack)](#5-docker-setup-full-stack)
6. [Service Access Endpoints](#6-service-access-endpoints)
7. [First Run - Ingest Satellite Data](#7-first-run--ingest-satellite-data)
8. [SAR Intelligence Module Setup](#8-sar-intelligence-module-setup)
9. [GPU / CUDA Setup (Optional but Recommended)](#9-gpu--cuda-setup-optional-but-recommended)
10. [Common Issues & Fixes](#10-common-issues--fixes)
11. [Architecture Quick Reference](#11-architecture-quick-reference)

---

## 1. Prerequisites

### Required

| Tool | Minimum Version | Install Link |
|---|---|---|
| **Docker Desktop** | 4.25+ | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| **Docker Compose** | v2.20+ | Bundled with Docker Desktop |
| **Git** | 2.40+ | [git-scm.com](https://git-scm.com/) |
| **8 GB RAM** | Minimum | --- |
| **20 GB Disk** | For data + images | --- |

### Optional (for GPU-accelerated SAR inference)

| Tool | Version | Notes |
|---|---|---|
| **NVIDIA GPU** | RTX 3060+ recommended | For Prithvi-EO CUDA FP16 inference |
| **CUDA Toolkit** | 12.1+ | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| **NVIDIA Container Toolkit** | Latest | [docs.nvidia.com/datacenter/cloud-native](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |

### API Keys Needed

| Service | Purpose | Get Key |
|---|---|---|
| **Copernicus Data Space** | Sentinel-2 satellite imagery download | [dataspace.copernicus.eu](https://dataspace.copernicus.eu/register) - Free |
| **Google Gemini API** | SAR LangGraph Agent LLM brain | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) - Free tier |
| **Groq API** *(optional)* | Alternative fast LLM (LLaMA-3.3-70B) | [console.groq.com](https://console.groq.com/) - Free tier |

---

## 2. Clone & Directory Structure

```bash
git clone https://github.com/varun-ai69/PDS-Project.git
cd PDS-Project
```

### Repository Layout

```
PDS-Project/
|-- README.md                    <- Project overview
|-- SETUP.md                     <- This file
|-- sar_implementation.md        <- SAR deep-dive doc
|-- docker-compose.yml           <- Main stack orchestration
|-- requirements.txt             <- Python dependencies
|-- .env.example                 <- Environment template (copy to .env)
|
|-- backend/                     <- FastAPI application
|   |-- api/                     #   REST routes (main.py + routers/)
|   |-- ingestion/               #   Tile download, STAC client, DB writer
|   `-- services/                #   Change detection, retrieval, encoder, analyst
|
|-- frontend/                    <- Glassmorphic web UIs
|   |-- index.html               #   Interactive satellite map
|   |-- retrieval.html           #   Semantic text/image search
|   |-- change.html              #   Multi-temporal change workbench
|   |-- clustering.html          #   HDBSCAN cluster explorer
|   `-- review.html              #   Analyst review queue
|
|-- SARIntelligence/             <- Defence SAR module
|   |-- agent/                   #   LangGraph ReAct agent + LLM providers
|   |-- atr_models/              #   YOLOv8-OBB maritime + MSTAR ground ATR
|   |-- embeddings/              #   SARCLIP ViT-L-14 encoder + Qdrant client
|   |-- pipeline/                #   Ingestion runner, template builder, MAS scoring
|   `-- preprocessing/           #   DN norm, Sigma-0 cal, phase co-registration
|
|-- infra/                       <- Docker & infrastructure
|   `-- docker/
|       `-- entrypoint.sh        #   Container startup + health-check waits
|
|-- models/                      <- Model weights (place downloads here)
|   `-- retrieval/               #   RemoteCLIP ViT-B-32 checkpoint
|
|-- data/                        <- Runtime data (auto-created)
|   |-- tiles/                   #   Downloaded GeoTIFFs
|   |-- change_staging/          #   Staged temporal pairs
|   `-- *.geojson                #   Pre-defined AOI regions
|
`-- tests/                       <- Test suites
```

---

## 3. Environment Configuration

```bash
# Copy the template
cp .env.example .env
```

Open `.env` and fill in the required values:

```bash
# ============================================================
# REQUIRED - Satellite Data
# ============================================================
COPERNICUS_CLIENT_ID=your_copernicus_client_id
COPERNICUS_CLIENT_SECRET=your_copernicus_client_secret

# ============================================================
# REQUIRED - Database (keep defaults for Docker)
# ============================================================
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=eo_archive
POSTGRES_USER=eo_admin
POSTGRES_PASSWORD=eo_password

# ============================================================
# REQUIRED - Vector Store (keep defaults for Docker)
# ============================================================
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# ============================================================
# REQUIRED - Object Storage (keep defaults for Docker)
# ============================================================
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# ============================================================
# SAR Intelligence - LLM Provider (choose one)
# ============================================================
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# SAR LLM provider: gemini | groq | openai | sovereign
SAR_LLM_PROVIDER=gemini

# Air-gapped classified mode (no external API calls)
SOVEREIGN_MODE=false

# ============================================================
# OPTIONAL - System Tuning
# ============================================================
LOG_LEVEL=INFO
MAX_WORKERS=4
TILE_SIZE=512
```

**Getting Copernicus credentials**: Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/register), then go to profile -> OAuth Clients -> Create Client.

---

## 4. Model Weights - Download Links

Place all weights in the correct directories **before** running Docker.

### 4a. RemoteCLIP (Optical Semantic Retrieval) - REQUIRED

```bash
mkdir -p models/retrieval
```

| File | Size | Download URL |
|---|---|---|
| `ViT-B-32.pt` | ~354 MB | https://huggingface.co/chendelong/RemoteCLIP/resolve/main/models/ViT-B-32.pt |
| `ViT-L-14.pt` | ~890 MB | https://huggingface.co/chendelong/RemoteCLIP/resolve/main/models/ViT-L-14.pt |
| `RN50.pt` | ~102 MB | https://huggingface.co/chendelong/RemoteCLIP/resolve/main/models/RN50.pt |

**Place at**: `models/retrieval/ViT-B-32.pt`

```bash
# Linux/Mac/WSL quick download
wget -P models/retrieval/ \
  https://huggingface.co/chendelong/RemoteCLIP/resolve/main/models/ViT-B-32.pt

# Windows PowerShell
Invoke-WebRequest -Uri "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/models/ViT-B-32.pt" `
  -OutFile "models\retrieval\ViT-B-32.pt"
```

---

### 4b. Prithvi-EO-2.0-300M (Change Detection) - REQUIRED for Change Detection

```bash
mkdir -p models/prithvi
```

| File | Size | Download URL |
|---|---|---|
| `Prithvi_EO_V2_300M.pt` | ~1.2 GB | https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt |
| `config.json` | ~3 KB | https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/config.json |

**Place at**: `models/prithvi/Prithvi_EO_V2_300M.pt`

> Change Detection falls back to ResNet-50 automatically if Prithvi weights are missing.

---

### 4c. YOLOv8x-OBB (SAR Maritime ATR)

```bash
mkdir -p SARIntelligence/weights/maritime
```

| File | Size | Download URL |
|---|---|---|
| `yolov8x-obb.pt` | ~130 MB | https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-obb.pt |

**Place at**: `SARIntelligence/weights/maritime/yolov8x-obb.pt`

```bash
wget -P SARIntelligence/weights/maritime/ \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-obb.pt
```

---

### 4d. SARCLIP ViT-L-14 (SAR Semantic Search)

```bash
mkdir -p SARIntelligence/weights/sarclip
```

| File | Size | Download URL |
|---|---|---|
| `ViT-L-14.pt` | ~890 MB | https://huggingface.co/chendelong/RemoteCLIP/resolve/main/models/ViT-L-14.pt |

**Place at**: `SARIntelligence/weights/sarclip/ViT-L-14.pt`

---

### Weight Directory Summary (after all downloads)

```
models/
|-- retrieval/
|   `-- ViT-B-32.pt              REQUIRED  - 354 MB
`-- prithvi/
    |-- Prithvi_EO_V2_300M.pt    RECOMMENDED - 1.2 GB
    `-- config.json

SARIntelligence/
`-- weights/
    |-- maritime/
    |   `-- yolov8x-obb.pt       SAR Defence - 130 MB
    `-- sarclip/
        `-- ViT-L-14.pt          SAR Defence - 890 MB
```

---

## 5. Docker Setup (Full Stack)

### Single Command Launch

```bash
docker compose up --build
```

| Service | Image | Port | Purpose |
|---|---|---|---|
| `backend` | Custom Python | 8000 | FastAPI REST API + ML inference |
| `postgres` | postgis/postgis:16-3.4 | 5432 | Tile registry + spatial queries |
| `qdrant` | qdrant/qdrant:v1.9.4 | 6333 | Vector embeddings store |
| `minio` | minio/minio:latest | 9000/9001 | S3-compatible GeoTIFF storage |

### Startup Sequence (automatic)

The `entrypoint.sh` waits in order:
1. Waits for PostgreSQL to accept connections
2. Waits for Qdrant to accept connections
3. Applies PostgreSQL schema migrations
4. Ensures Qdrant `tile_embeddings` collection exists
5. Starts FastAPI via Uvicorn

### GPU Support

```bash
# Verify NVIDIA Container Toolkit is installed, then:
docker compose up --build

# Verify GPU is visible inside container
docker exec -it canopus-backend python -c "import torch; print(torch.cuda.is_available())"
```

---

## 6. Service Access Endpoints

| Interface | URL | Description |
|---|---|---|
| Satellite Map | http://localhost:8000 | Interactive Leaflet map |
| Semantic Search | http://localhost:8000/retrieval.html | Natural language + image search |
| Change Detection | http://localhost:8000/change.html | Multi-temporal change workbench |
| Cluster Explorer | http://localhost:8000/clustering.html | HDBSCAN landscape clustering |
| Analyst Review | http://localhost:8000/review.html | Human-in-the-loop review queue |
| Swagger API Docs | http://localhost:8000/docs | Interactive REST API explorer |
| MinIO Console | http://localhost:9001 | S3 bucket browser (admin/minioadmin) |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector collections browser |

---

## 7. First Run - Ingest Satellite Data

### Via REST API

```bash
curl -X POST http://localhost:8000/api/v1/ingest/ \
  -H "Content-Type: application/json" \
  -d '{
    "region_name": "Delhi NCR",
    "bbox": [76.8, 28.4, 77.4, 28.8],
    "start_date": "2024-01-01",
    "end_date": "2024-10-01",
    "cloud_cover_max": 15,
    "max_tiles": 20
  }'
```

### Pre-defined AOIs (ready to use)

| File | Location | Use Case |
|---|---|---|
| `data/bhadla_solar_aoi.geojson` | Rajasthan, India | Solar farm change detection |
| `data/dholera_sir_aoi.geojson` | Gujarat, India | Smart city construction |
| `data/jewar_airport_aoi.geojson` | Uttar Pradesh, India | Airport infrastructure growth |
| `data/berlin_tesla_aoi.geojson` | Brandenburg, Germany | Industrial facility monitoring |

---

## 8. SAR Intelligence Module Setup

### Initialize SAR Database

```bash
docker exec -it canopus-backend python -c "
from SARIntelligence.pipeline.db_setup import init_sar_database
init_sar_database()
print('SAR database initialized')
"
```

### Run SAR Ingestion Pipeline

```bash
# With synthetic demo data
docker exec -it canopus-backend python -m SARIntelligence.pipeline.ingestion_runner \
  --input-dir /app/data/sar_input \
  --scene-id DEMO_SCENE_001 \
  --domain maritime \
  --lat 22.5 --lon 72.8

# With a real Sentinel-1 GRD .SAFE directory
docker exec -it canopus-backend python -m SARIntelligence.pipeline.ingestion_runner \
  --input-dir /app/data/sar_input/S1A_IW_GRDH_2024.SAFE \
  --scene-id S1A_2024_001 \
  --domain ground
```

### Run the Autonomous SAR Agent

```bash
docker exec -it canopus-backend python -c "
from SARIntelligence.agent.langgraph_orchestrator import SARLangGraphAgent
agent = SARLangGraphAgent(llm_provider='gemini')
result = agent.run('Identify all DEFCON-2 or higher maritime scenes and summarize threat status')
print(result)
"
```

### SAR LLM Provider Options

| Provider | .env Setting | Notes |
|---|---|---|
| Gemini 2.5 Flash | `SAR_LLM_PROVIDER=gemini` | Best reasoning, free tier available |
| Groq LLaMA-3.3-70B | `SAR_LLM_PROVIDER=groq` | Fastest inference |
| OpenAI-compatible | `SAR_LLM_PROVIDER=openai` | Works with Qwen, vLLM, Ollama |
| Sovereign Air-Gapped | `SOVEREIGN_MODE=true` | Zero external API calls, deterministic |

---

## 9. GPU / CUDA Setup (Windows + Docker Desktop)

```powershell
# Step 1: Enable WSL2 backend in Docker Desktop Settings -> General

# Step 2: Install NVIDIA drivers
# Download from: https://www.nvidia.com/Download/index.aspx
# Use Game Ready or Studio drivers (NOT CUDA/Data Center drivers)

# Step 3: Install NVIDIA Container Toolkit in WSL2
wsl
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
exit

# Step 4: Verify GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 10. Common Issues & Fixes

### PostgreSQL connection error
```bash
# Wait 30s for DB to start, then:
docker compose restart backend
```

### Qdrant collection missing
```bash
docker exec -it canopus-backend python -c "
from backend.services.vector_store import get_qdrant_client, ensure_collection_exists
ensure_collection_exists(get_qdrant_client())
"
```

### RemoteCLIP weights not found
Download `ViT-B-32.pt` from HuggingFace and place at `models/retrieval/ViT-B-32.pt`

### Missing Python packages (einops, ultralytics)
```bash
docker exec -it canopus-backend pip install einops ultralytics
```

### Out of GPU memory
Reduce batch size in `backend/services/patch_change_detector.py`:
```python
all_vecs = self.encoder.encode_multiband_patches(all_crops, batch_size=8)  # from 32
```

---

## 11. Architecture Quick Reference

```
USER INTERFACES (port 8000)
  Map | Retrieval | Change | Clustering | Review
            |
            v
FastAPI Backend (port 8000)
  /ingest  /retrieve  /change  /cluster  /review
  /sar-agent  /coverage  /analyst  /export  /health
     |           |           |          |
     v           v           v          v
PostgreSQL   Qdrant       MinIO    ML Inference
PostGIS     Vectors       S3       RemoteCLIP
port 5432   port 6333  port 9000  Prithvi-EO
                                  YOLOv8-OBB
                                  SARCLIP
```

| Layer | Technology | Purpose |
|---|---|---|
| Satellite Retrieval | RemoteCLIP ViT-B-32 + QB-Norm | Zero-shot semantic image search |
| Change Detection | Prithvi-EO-2.0-300M + Left-MAD | Foundation model + robust anomaly scoring |
| SAR ATR | YOLOv8x-OBB | Oriented bounding box maritime/ground detection |
| SAR Embedding | SARCLIP ViT-L-14 768-D | SAR semantic vector search via Qdrant |
| Agent | LangGraph ReAct | Autonomous multi-step tactical intelligence |
| Vector Store | Qdrant v1.9.4 | ANN search with payload filters |
| Spatial DB | PostgreSQL 16 + PostGIS 3.4 | GIS spatial queries and temporal indexing |
| Object Storage | MinIO | Self-hosted S3-compatible GeoTIFF archive |

---

*SIH-2026 PS26227 - Ministry of Defence / Indian Army*
*Semantic Retrieval | Multi-Temporal Change Detection | SAR Defence Reconnaissance*
