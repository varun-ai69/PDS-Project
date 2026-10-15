"""
backend/api/routers/sar_agent.py
================================
FastAPI Router for CANOPUS Sovereign SAR Autonomous Tactical Query Engine.
Powered by:
- Decoupled LangGraph Multi-Node StateGraph Architecture
- Pluggable Groq LPU (openai/gpt-oss-120b) / Gemini / Sovereign Air-Gapped Brain
- 4 Pristine Defense Tools (query_database, semantic_similarity_search, analyze_temporal_surge, get_scene_telemetry)
- Multimodal Image-to-Image and Text-to-Image Semantic Retrieval
"""

import os
import re
import json
import shutil
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from SARIntelligence.agent import (
    SARTacticalLangGraphAgent,
    TacticalTools,
    get_llm_provider,
)

logger = logging.getLogger("SARAgentRouter")

router = APIRouter(prefix="/api/sar", tags=["SAR Autonomous Defense Intelligence"])

# Shared singleton tools suite
_tools_instance: Optional[TacticalTools] = None


def get_tools() -> TacticalTools:
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = TacticalTools()
    return _tools_instance


def get_agent_for_request(provider_name: str = "auto", prefer_offline: bool = False) -> SARTacticalLangGraphAgent:
    """Instantiates or configures the agent with requested LLM brain."""
    tools = get_tools()
    if prefer_offline or provider_name == "sovereign_offline":
        provider = get_llm_provider("sovereign_offline")
    else:
        provider = get_llm_provider(provider_name)
    return SARTacticalLangGraphAgent(llm_provider=provider, tools=tools)


# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -----------------------------------------------------------------------------

class TacticalQueryRequest(BaseModel):
    query: str = Field(..., description="Analyst operational directive or strict coordinate query")
    provider: Optional[str] = Field("auto", description="LLM Brain Provider ('groq', 'gemini', 'sovereign_offline', 'auto')")
    prefer_offline: bool = Field(False, description="Force Sovereign Air-Gapped ReAct engine")


class TacticalQueryResponse(BaseModel):
    query: str
    final_response: str
    total_steps: int
    reasoning_trajectory: list
    direct_data: Dict[str, Any]
    provider_info: Dict[str, str]


# -----------------------------------------------------------------------------
# HELPER: ENRICH DIRECT DATA FOR TACTICAL HUD RENDERING
# -----------------------------------------------------------------------------

def enrich_response_data(res: Dict[str, Any], tools: TacticalTools) -> Dict[str, Any]:
    """Ensures every single item in direct_data has verified coordinates and web image URLs."""
    raw_results = res.get("retrieved_data", [])
    if isinstance(raw_results, dict):
        raw_results = [raw_results]
    elif not isinstance(raw_results, list):
        raw_results = []

    enriched = []
    seen = set()

    for item in raw_results:
        sid = item.get("scene_id") or item.get("pair_id")
        if not sid or sid in seen:
            continue
        seen.add(sid)

        rec = dict(item)

        # Standardize coordinates
        coords = rec.get("coordinates") or {}
        if not coords and ("lat" in rec and "lon" in rec):
            coords = {"lat": rec["lat"], "lon": rec["lon"]}
        rec["coordinates"] = coords

        # Check if temporal pair
        is_temporal = "pair" in sid.lower() or "temporal" in sid.lower() or "net_delta" in rec
        if is_temporal:
            import re
            m = re.search(r'\d+', sid)
            p_num = int(m.group(0)) if m else 1
            p_str = f"{p_num:02d}"

            rec["is_temporal"] = True
            rec["t1_image_url"] = f"/sar-detections/multi_temporal/temporal_pair_{p_str}_T1_baseline_detected.png"
            rec["t2_image_url"] = f"/sar-detections/multi_temporal/temporal_pair_{p_str}_T2_current_detected.png"
            rec["diff_heatmap_url"] = f"/sar-diffs/diff_pair_{p_str}.png"
            rec["overlay_url"] = rec["t2_image_url"]
            rec["preprocessed_url"] = rec["t1_image_url"]
        else:
            rec["is_temporal"] = False
            dom = rec.get("domain", "ground")
            urls = tools._format_urls(sid, dom)
            if not rec.get("overlay_url"):
                rec["overlay_url"] = urls["overlay_url"]
            if not rec.get("preprocessed_url"):
                rec["preprocessed_url"] = urls["preprocessed_url"]

        enriched.append(rec)

    # Also search mentioned scene IDs in final response if direct_data was empty
    if not enriched:
        final_text = res.get("final_response", "")
        mentioned = re.findall(r"\b(ground_[a-zA-Z0-9_]+|ocean_[a-zA-Z0-9_]+|temporal_pair_\d+[a-zA-Z0-9_]*)\b", final_text)
        for msid in mentioned:
            clean_sid = re.sub(r'(_detected)?(\.png)?$', '', msid)
            if clean_sid not in seen:
                seen.add(clean_sid)
                telemetry = tools.get_scene_telemetry(clean_sid)
                if telemetry:
                    enriched.append(telemetry)

    return {
        "results": enriched,
        "total_scenes": len(enriched)
    }


# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@router.post("/query", response_model=TacticalQueryResponse)
async def process_tactical_query(request: TacticalQueryRequest):
    """
    Executes autonomous multi-step LangGraph tactical query.
    Connects dynamically to Groq, Gemini, or Sovereign Air-Gapped Engine.
    """
    try:
        agent = get_agent_for_request(
            provider_name=request.provider or "auto",
            prefer_offline=request.prefer_offline
        )

        res = agent.run_query(request.query)
        enriched_direct_data = enrich_response_data(res, agent.tools)

        provider_name = agent.llm_provider.__class__.__name__
        model_name = getattr(agent.llm_provider, "model_name", "Sovereign-ReAct-Local")

        return TacticalQueryResponse(
            query=res["query"],
            final_response=res["final_response"],
            total_steps=res.get("total_steps", len(res.get("reasoning_trajectory", []))),
            reasoning_trajectory=res.get("reasoning_trajectory", []),
            direct_data=enriched_direct_data,
            provider_info={"provider": provider_name, "model": model_name}
        )
    except Exception as e:
        logger.error("Error executing tactical query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/image", response_model=TacticalQueryResponse)
async def process_image_semantic_query(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    provider: Optional[str] = Form("auto"),
    domain: Optional[str] = Form(None)
):
    """
    Multimodal Image-to-Image (and Image+Text) SAR Semantic Retrieval.
    Uploads an input SAR image chip, extracts 768-D SARCLIP embeddings,
    and returns visually and tactically similar radar targets with comprehensive SITREP.
    """
    upload_dir = os.path.join(agent_upload_dir(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    temp_filename = f"upload_{os.urandom(4).hex()}_{file.filename}"
    temp_path = os.path.join(upload_dir, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        tools = get_tools()
        # 1. Execute Multimodal Image Retrieval
        prompt_text = query or "Locate visually and tactically identical SAR radar passes matching this uploaded target chip."
        matches = tools.semantic_similarity_search(
            image_path=temp_path,
            query_text=query,
            domain_filter=domain,
            top_k=5
        )

        # 2. Invoke Agent to synthesize complete Military SITREP with proof
        agent = get_agent_for_request(provider_name=provider or "auto")
        synthesis = agent.llm_provider.synthesize_report(
            query=f"Uploaded SAR Target Image ({file.filename}): {prompt_text}",
            reasoning_history=[{
                "step": 1,
                "node": "MULTIMODAL_IMAGE_SEARCH",
                "thought": f"Encoded uploaded radar chip '{file.filename}' via 768-D SARCLIP Vision Transformer and retrieved top {len(matches)} matching scenes.",
                "tool": "semantic_similarity_search",
                "args": {"image_path": file.filename, "query_text": query},
                "observation": matches
            }],
            direct_data=matches,
            is_strict_output=False
        )

        enriched_direct_data = {
            "results": matches,
            "total_scenes": len(matches),
            "uploaded_image": f"/sar-uploads/{temp_filename}" if os.path.exists(temp_path) else None
        }

        provider_name = agent.llm_provider.__class__.__name__
        model_name = getattr(agent.llm_provider, "model_name", "Sovereign-ReAct-Local")

        return TacticalQueryResponse(
            query=f"[IMAGE RETRIEVAL] {file.filename} - {prompt_text}",
            final_response=synthesis,
            total_steps=1,
            reasoning_trajectory=[{
                "step": 1,
                "node": "MULTIMODAL_IMAGE_SEARCH",
                "thought": f"Extracted 768-D SARCLIP feature embedding from uploaded image '{file.filename}'. Found {len(matches)} matching radar targets.",
                "tool": "semantic_similarity_search",
                "args": {"filename": file.filename, "prompt": query},
                "observation_summary": f"Top similarity: {matches[0].get('scene_id') if matches else 'None'} (Score: {matches[0].get('similarity_score') if matches else 0})"
            }],
            direct_data=enriched_direct_data,
            provider_info={"provider": provider_name, "model": model_name}
        )

    except Exception as e:
        logger.error("Error executing image semantic query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def agent_upload_dir() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(root, "data")


@router.get("/scene/{scene_id}")
async def get_scene_master_template(scene_id: str):
    """
    Retrieve complete 6-layer Master Template telemetry for a specific radar scene.
    """
    tools = get_tools()
    telemetry = tools.get_scene_telemetry(scene_id)
    if not telemetry:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found in Master Catalog")
    return telemetry


@router.get("/stats")
async def get_pipeline_stats():
    """
    Return operational inventory and database statistics across all SAR sectors.
    """
    tools = get_tools()
    conn = tools._get_db()
    cur = conn.cursor()
    cur.execute("SELECT domain, count(*) FROM scenes GROUP BY domain")
    counts = dict(cur.fetchall())
    conn.close()

    total_catalog_scenes = len(tools.master_catalog)
    return {
        "status": "OPERATIONAL",
        "total_catalog_scenes": total_catalog_scenes,
        "database_counts": counts,
        "cognitive_brains": ["Groq (openai/gpt-oss-120b)", "Google Gemini (gemini-3.5-flash-lite)", "Sovereign Air-Gapped"],
        "tools_deployed": ["query_database", "semantic_similarity_search", "analyze_temporal_surge", "get_scene_telemetry"]
    }
