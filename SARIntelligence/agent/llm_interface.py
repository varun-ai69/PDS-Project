"""
SARIntelligence/agent/llm_interface.py
======================================
Pluggable LLM Abstraction Layer for Defense SAR Satellite Intelligence.
Decouples the Agent reasoning/graph logic from any specific LLM provider.

Supports:
1. Google Gemini (gemini-3.5-flash-lite / gemini-2.5-flash)
2. Groq (Fast LLaMA-3.3-70B / Mixtral for rapid inference)
3. OpenAI-Compatible API (for fine-tuned Qwen, DeepSeek, vLLM, or Ollama local models)
4. Sovereign Air-Gapped Engine (100% offline deterministic rule-based planner for defense servers)
"""

import os
import sys
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("LLMInterface")
logging.basicConfig(level=logging.INFO)


class StepDecision:
    """Represents a decision taken by an LLM during the ReAct reasoning step."""
    def __init__(
        self,
        decision_type: str,  # 'CALL_TOOL' or 'SYNTHESIZE'
        thought: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        final_answer: Optional[str] = None
    ):
        self.decision_type = decision_type
        self.thought = thought
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.final_answer = final_answer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "final_answer": self.final_answer
        }


class BaseLLMProvider(ABC):
    """
    Abstract Protocol for all LLM Providers.
    The LangGraph Agent ONLY interacts with this interface.
    """

    @abstractmethod
    def plan_and_decide(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]]
    ) -> StepDecision:
        """
        Analyzes the query and past observations, and decides whether to
        call another tool (CALL_TOOL) or conclude and synthesize (SYNTHESIZE).
        """
        pass

    @abstractmethod
    def synthesize_report(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        direct_data: List[Dict[str, Any]],
        is_strict_output: bool = False
    ) -> str:
        """
        Generates the final comprehensive Military SITREP or strict output
        based on the accumulated observations and imagery.
        """
        pass


# =============================================================================
# 1. GOOGLE GEMINI PROVIDER
# =============================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider (gemini-3.5-flash-lite / gemini-2.5-flash)."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self._configured = False

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self._configured = True
                logger.info("GeminiProvider active with model '%s'", self.model_name)
            except Exception as e:
                logger.warning("GeminiProvider configuration failed: %s", e)

    def is_available(self) -> bool:
        return self._configured

    def plan_and_decide(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]]
    ) -> StepDecision:
        if not self.is_available():
            return SovereignOfflineProvider().plan_and_decide(query, reasoning_history, available_tools)

        prompt = f"""You are the tactical decision engine for Indian Army SAR reconnaissance.
Commander Query: "{query}"

Available Tools:
{json.dumps(available_tools, indent=2)}

Past Execution History:
{json.dumps(reasoning_history, indent=2)}

TASK:
- If sufficient evidence exists to answer the commander's query, return decision_type: "SYNTHESIZE".
- If you need data, decide which tool to call with exact parameters.
- Respond ONLY with valid JSON in this exact structure:
{{
  "decision_type": "CALL_TOOL" or "SYNTHESIZE",
  "thought": "Your tactical reasoning",
  "tool_name": "query_database" | "semantic_similarity_search" | "analyze_temporal_surge" | "get_scene_telemetry",
  "tool_args": {{ "key": "value" }}
}}
"""
        try:
            model = self.genai.GenerativeModel(self.model_name)
            res = model.generate_content(prompt)
            txt = res.text.strip()
            if txt.startswith("```json"):
                txt = txt[7:-3].strip()
            elif txt.startswith("```"):
                txt = txt[3:-3].strip()

            parsed = json.loads(txt)
            return StepDecision(
                decision_type=parsed.get("decision_type", "CALL_TOOL"),
                thought=parsed.get("thought", "Analyzing tactical requirement"),
                tool_name=parsed.get("tool_name"),
                tool_args=parsed.get("tool_args", {})
            )
        except Exception as e:
            logger.warning("Gemini planning note: %s. Using deterministic fallback.", e)
            return SovereignOfflineProvider().plan_and_decide(query, reasoning_history, available_tools)

    def synthesize_report(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        direct_data: List[Dict[str, Any]],
        is_strict_output: bool = False
    ) -> str:
        if not self.is_available():
            return SovereignOfflineProvider().synthesize_report(query, reasoning_history, direct_data, is_strict_output)

        if is_strict_output and direct_data:
            item = direct_data[0]
            lat = item.get("coordinates", {}).get("lat", 0.0)
            lon = item.get("coordinates", {}).get("lon", 0.0)
            return (
                f"SCENE_ID: {item.get('scene_id')}\n"
                f"COORDINATES: Lat {lat:.4f}°N, Lon {lon:.4f}°E\n"
                f"IMAGE_PATH: {item.get('overlay_url', item.get('t2_image_url'))}\n"
                f"MILITARY_STATUS: {item.get('defcon_tier', 'DEFCON_3_ELEVATED')}"
            )

        prompt = f"""You are CANOPUS-AI, the Sovereign Military Intelligence Officer for Defense SAR Satellite Reconnaissance.
Commander Query: "{query}"

Telemetry & Tool Results:
{json.dumps(direct_data[:4], indent=2)}

Generate a structured, professional Military Intelligence SITREP in Markdown:
### 📋 MILITARY INTELLIGENCE SITREP: TACTICAL SAR RECONNAISSANCE
Include:
1. Operational AOI & Geographic Coordinates (Lat, Lon)
2. Target Inventory & Unit Classification
3. Radar Phenomenology (Backscatter RCS in dBsm, Shadow profile, Metallic reflections)
4. Tactical Threat Evaluation (DEFCON level assessment)
5. Actionable Directive for Theater Command
6. Radar Detection Overlay Image link
"""
        try:
            model = self.genai.GenerativeModel(self.model_name)
            res = model.generate_content(prompt)
            return res.text.strip()
        except Exception as e:
            logger.warning("Gemini synthesis fallback: %s", e)
            return SovereignOfflineProvider().synthesize_report(query, reasoning_history, direct_data, is_strict_output)


# =============================================================================
# 2. GROQ PROVIDER (GPT-OSS-120B / Qwen for Ultra-Fast LPU Inference)
# =============================================================================

class GroqProvider(BaseLLMProvider):
    """Groq API provider for ultra-fast LPU inference (openai/gpt-oss-120b / qwen/qwen3.8-27b)."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "openai/gpt-oss-120b"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info("GroqProvider active with model '%s'", self.model_name)
            except Exception as e:
                logger.warning("Groq initialization note: %s", e)

    def is_available(self) -> bool:
        return self.client is not None

    def plan_and_decide(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]]
    ) -> StepDecision:
        if not self.is_available():
            return SovereignOfflineProvider().plan_and_decide(query, reasoning_history, available_tools)

        # If tools were already executed and we have valid observations, check if we can synthesize
        if reasoning_history and any(step.get("observation") for step in reasoning_history):
            return StepDecision(
                decision_type="SYNTHESIZE",
                thought="Intelligence observations acquired from radar database. Synthesizing report."
            )

        system_msg = (
            "You are an autonomous AI query planner for a Synthetic Aperture Radar (SAR) defense intelligence benchmark. "
            "Analyze the analyst query and decide which tool to invoke.\n"
            "Available tools:\n"
            "1. query_database: Structured search by domain ('ground', 'ocean', 'multi_temporal'), sector ('AKHNOOR', 'CHAMB', etc.), target_class ('T72', 'BMP2', 'vessel', 'aircraft'), or max_targets=0 for negative constraints / safe zones / calm sea.\n"
            "2. semantic_similarity_search: Multimodal 768-D SARCLIP vector retrieval for descriptive visual queries or uploaded images.\n"
            "3. analyze_temporal_surge: Multi-temporal pass differencing. Compares past baseline T1 vs current T2, measures troop delta (T2 - T1), and provides difference heatmaps.\n"
            "4. get_scene_telemetry: Deep inspection of bounding boxes, RCS in dBsm, and GPS for a given scene_id.\n\n"
            "Respond ONLY with valid JSON with keys: 'decision_type' ('CALL_TOOL' or 'SYNTHESIZE'), 'thought', 'tool_name', 'tool_args'."
        )

        prompt = (
            f"Analyst Query: {query}\n"
            f"History: {json.dumps(reasoning_history[-2:] if reasoning_history else [])}\n"
            "Output strictly valid JSON with exact keys: 'decision_type', 'thought', 'tool_name', 'tool_args'."
        )

        try:
            chat = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                response_format={"type": "json_object"}
            )
            raw_txt = chat.choices[0].message.content.strip()
            parsed = json.loads(raw_txt)

            tool_name = parsed.get("tool_name") or parsed.get("tool") or "query_database"
            # Normalize tool names
            if "temporal" in tool_name or "surge" in tool_name or "diff" in tool_name:
                tool_name = "analyze_temporal_surge"
            elif "vector" in tool_name or "semantic" in tool_name or "similarity" in tool_name:
                tool_name = "semantic_similarity_search"
            elif "telemetry" in tool_name or "inspect" in tool_name or "template" in tool_name:
                tool_name = "get_scene_telemetry"
            elif "query" in tool_name or "database" in tool_name or "db" in tool_name or "search" in tool_name:
                tool_name = "query_database"

            tool_args = parsed.get("tool_args") or parsed.get("args") or {}

            # Normalize common argument aliases
            if "target_type" in tool_args:
                tool_args["target_class"] = tool_args.pop("target_type")
            if "region" in tool_args and "sector" not in tool_args:
                tool_args["sector"] = tool_args.pop("region").replace("sector", "").strip().upper()
            if "object" in tool_args and "target_class" not in tool_args:
                val = tool_args.pop("object")
                if "t-72" in val.lower() or "t72" in val.lower():
                    tool_args["target_class"] = "T72"
                elif "bmp" in val.lower():
                    tool_args["target_class"] = "BMP2"

            return StepDecision(
                decision_type=parsed.get("decision_type", "CALL_TOOL"),
                thought=parsed.get("thought", "Autonomous query planning via Groq"),
                tool_name=tool_name,
                tool_args=tool_args
            )
        except Exception as e:
            logger.warning("Groq plan error: %s. Using Sovereign fallback.", e)
            return SovereignOfflineProvider().plan_and_decide(query, reasoning_history, available_tools)

    def synthesize_report(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        direct_data: List[Dict[str, Any]],
        is_strict_output: bool = False
    ) -> str:
        if not self.is_available() or is_strict_output or not direct_data:
            return SovereignOfflineProvider().synthesize_report(query, reasoning_history, direct_data, is_strict_output)

        system_msg = (
            "You are CANOPUS-AI, the Sovereign Military Intelligence Officer for Defense SAR Satellite Reconnaissance. "
            "Generate a professional, structured Military Intelligence SITREP in GitHub Markdown."
        )

        prompt = (
            f"Commander Query: {query}\n"
            f"Telemetry & Tool Results:\n{json.dumps(direct_data[:3], indent=2)}\n\n"
            "Format the report with:\n"
            "### 📋 MILITARY INTELLIGENCE SITREP: TACTICAL SAR RECONNAISSANCE\n"
            "1. Operational AOI & Geographic Coordinates (Lat, Lon)\n"
            "2. Target Inventory & Unit Classification\n"
            "3. Radar Phenomenology (Backscatter RCS in dBsm, Shadow profile)\n"
            "4. Tactical Threat Evaluation (DEFCON level assessment)\n"
            "5. Actionable Directive for Theater Command\n"
            "6. Radar Detection Overlay Image path"
        )

        try:
            chat = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Groq synthesis error: %s. Using Sovereign fallback.", e)
            return SovereignOfflineProvider().synthesize_report(query, reasoning_history, direct_data, is_strict_output)


# =============================================================================
# 3. OPENAI-COMPATIBLE PROVIDER (FOR FINE-TUNED QWEN / OLLAMA / vLLM)
# =============================================================================

class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Universal adapter for self-hosted or fine-tuned local models
    (e.g., Qwen-2.5-7B/14B, DeepSeek-R1, Ollama, vLLM, LMStudio).
    """

    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: str = "ollama", model_name: str = "qwen2.5:7b"):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

    def plan_and_decide(self, query: str, reasoning_history: List[Dict[str, Any]], available_tools: List[Dict[str, Any]]) -> StepDecision:
        # Connects via standard requests or httpx to any OpenAI-compatible /v1/chat/completions
        try:
            import requests
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a military defense agent. Return JSON decision."},
                    {"role": "user", "content": f"Query: {query}\nTools: {json.dumps(available_tools)}\nHistory: {json.dumps(reasoning_history)}"}
                ],
                "response_format": {"type": "json_object"}
            }
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=5.0)
            if resp.status_code == 200:
                parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
                return StepDecision(
                    decision_type=parsed.get("decision_type", "CALL_TOOL"),
                    thought=parsed.get("thought", ""),
                    tool_name=parsed.get("tool_name"),
                    tool_args=parsed.get("tool_args", {})
                )
        except Exception:
            pass

        return SovereignOfflineProvider().plan_and_decide(query, reasoning_history, available_tools)

    def synthesize_report(self, query: str, reasoning_history: List[Dict[str, Any]], direct_data: List[Dict[str, Any]], is_strict_output: bool = False) -> str:
        return SovereignOfflineProvider().synthesize_report(query, reasoning_history, direct_data, is_strict_output)


# =============================================================================
# 4. SOVEREIGN AIR-GAPPED OFFLINE PROVIDER (100% DETERMINISTIC LOCAL BRAIN)
# =============================================================================

class SovereignOfflineProvider(BaseLLMProvider):
    """
    100% Sovereign Air-Gapped Cognitive Engine.
    Operates completely offline with zero internet or API keys.
    Uses military entity extraction and ReAct heuristics.
    """

    def plan_and_decide(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]]
    ) -> StepDecision:
        q = query.lower()

        # If a tool was already executed and returned results, we can synthesize!
        if reasoning_history and any(step.get("observation") for step in reasoning_history):
            # If we retrieved a scene and user wants granular telemetry, call get_scene_telemetry
            last_obs = reasoning_history[-1].get("observation")
            if isinstance(last_obs, list) and len(last_obs) > 0 and len(reasoning_history) == 1:
                first_sid = last_obs[0].get("scene_id")
                if first_sid and "telemetry" in q or "detail" in q or "bbox" in q:
                    return StepDecision(
                        decision_type="CALL_TOOL",
                        thought=f"Inspecting granular target bounding boxes and RCS for primary scene {first_sid}",
                        tool_name="get_scene_telemetry",
                        tool_args={"scene_id": first_sid}
                    )
            return StepDecision(
                decision_type="SYNTHESIZE",
                thought="Sufficient tactical intelligence acquired. Synthesizing final military report."
            )

        # ---------------------------------------------------------------------
        # DECISION ROUTING BASED ON ANALYST INTENT
        # ---------------------------------------------------------------------

        # Intent 1: Multi-temporal surge / peak activity
        if any(w in q for w in ["surge", "peak", "increase", "buildup", "temporal", "compare", "pair", "escalat", "recent"]):
            pair_num = None
            import re
            m = re.search(r'pair\s*(\d+)', q)
            if m:
                pair_num = int(m.group(1))

            sec = None
            for s in ["AKHNOOR", "CHAMB", "GALWAN", "PANGONG", "SAMBA", "URI", "BARAMULLA", "PATHANKOT", "AMBALA"]:
                if s.lower() in q:
                    sec = s
                    break

            return StepDecision(
                decision_type="CALL_TOOL",
                thought="Commander requested temporal surge or troop escalation analysis. Calling analyze_temporal_surge.",
                tool_name="analyze_temporal_surge",
                tool_args={"sector": sec, "pair_number": pair_num, "min_surge_delta": 1}
            )

        # Intent 2: Negative Constraint / Zero Military Activity / Safe Zone
        if any(w in q for w in ["no military", "zero military", "zero target", "no target", "safe zone", "clear sea", "calm water", "calm sea"]):
            return StepDecision(
                decision_type="CALL_TOOL",
                thought="Negative constraint detected. Searching database for sectors with max_targets=0.",
                tool_name="query_database",
                tool_args={"max_targets": 0, "limit": 5}
            )

        # Intent 3: Class / Sector Relational Search
        target_class = None
        for cls in ["T72", "BMP2", "BTR70", "BTR60", "BRDM2", "ZSU_23_4", "2S1", "vessel", "aircraft"]:
            if cls.lower() in q:
                target_class = cls
                break

        sector = None
        for sec in ["AKHNOOR", "CHAMB", "GALWAN", "PANGONG", "SAMBA", "URI", "BARAMULLA", "PATHANKOT", "AMBALA", "BAY_OF_BENGAL", "ARABIAN_SEA"]:
            if sec.lower() in q:
                sector = sec
                break

        domain = None
        if "ocean" in q or "sea" in q or "naval" in q or "warship" in q:
            domain = "ocean"
        elif "airbase" in q or "aircraft" in q or "tank" in q or "armor" in q or "ground" in q:
            domain = "ground"

        # If specific sector or class identified, query database
        if target_class or sector or domain:
            return StepDecision(
                decision_type="CALL_TOOL",
                thought=f"Tactical entity query detected (Class: {target_class}, Sector: {sector}, Domain: {domain}). Calling query_database.",
                tool_name="query_database",
                tool_args={"domain": domain, "sector": sector, "target_class": target_class, "limit": 6}
            )

        # Intent 4: Open-vocabulary Semantic Search (Default)
        return StepDecision(
            decision_type="CALL_TOOL",
            thought="Unstructured tactical description detected. Calling 768-D semantic_similarity_search.",
            tool_name="semantic_similarity_search",
            tool_args={"query_text": query, "top_k": 5}
        )

    def synthesize_report(
        self,
        query: str,
        reasoning_history: List[Dict[str, Any]],
        direct_data: List[Dict[str, Any]],
        is_strict_output: bool = False
    ) -> str:
        if not direct_data:
            return f"### ⚠️ DEFENSE INTELLIGENCE NOTICE\nNo military radar assets matching query: `{query}` were localized in current surveillance passes."

        # Strict Output
        if is_strict_output:
            item = direct_data[0]
            lat = item.get("coordinates", {}).get("lat", 0.0)
            lon = item.get("coordinates", {}).get("lon", 0.0)
            img = item.get("overlay_url") or item.get("t2_image_url") or ""
            status = item.get("defcon_tier") or item.get("threat_assessment") or "DEFCON_3_ELEVATED"
            return (
                f"SCENE_ID: {item.get('scene_id') or item.get('pair_id')}\n"
                f"COORDINATES: Lat {lat:.4f}°N, Lon {lon:.4f}°E\n"
                f"IMAGE_PATH: {img}\n"
                f"MILITARY_STATUS: {status}"
            )

        # Standard Comprehensive Military SITREP
        lines = [
            "### 📋 MILITARY INTELLIGENCE SITREP: TACTICAL SAR RECONNAISSANCE",
            f"**Operational Query:** `{query}`",
            f"**Verified Target Formations Located:** `{len(direct_data)}`",
            ""
        ]

        for idx, item in enumerate(direct_data[:4], 1):
            sid = item.get("scene_id") or item.get("pair_id")
            sec = item.get("sector") or item.get("scene_name")
            lat = item.get("coordinates", {}).get("lat", 0.0)
            lon = item.get("coordinates", {}).get("lon", 0.0)
            targets = item.get("total_targets") or item.get("current_targets") or 0
            tier = item.get("defcon_tier") or item.get("threat_assessment") or "DEFCON_2_HIGH"
            img = item.get("overlay_url") or item.get("t2_image_url") or ""
            diff = item.get("diff_heatmap_url")

            lines.append(f"#### {idx}. Sector: `{sec}` (Scene: `{sid}`)")
            lines.append(f"- **Geographic Coordinates:** Lat `{lat:.4f}°N`, Lon `{lon:.4f}°E`")
            lines.append(f"- **Combat Units:** `{targets}` verified targets ({item.get('primary_classes', 'Military assets')})")
            if "net_delta" in item:
                lines.append(f"- **Buildup Delta:** `{item['net_delta']}` combat assets mobilized from baseline")
            lines.append(f"- **Threat Assessment:** `{tier}`")
            lines.append(f"- **Radar Overlay Imagery:** `{img}`")
            if diff:
                lines.append(f"- **Difference Heatmap:** `{diff}`")
            lines.append("")

        lines.append("### 🛡️ RECOMMENDED ACTION FOR THEATER COMMAND")
        lines.append("- Dispatch high-altitude UAV for electro-optical cross-verification.")
        lines.append("- Task next orbital spotlight SAR pass for sub-meter hull classification.")

        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTION: GET LLM PROVIDER
# =============================================================================

def get_llm_provider(
    provider_name: str = "auto",
    gemini_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseLLMProvider:
    """
    Factory function to instantiate any LLM provider without touching the agent graph.
    Options: 'gemini', 'groq', 'openai_compatible', 'sovereign_offline', 'auto'
    """
    p_name = provider_name.lower()

    if p_name == "gemini":
        return GeminiProvider(api_key=gemini_api_key, model_name=model_name or "gemini-3.5-flash-lite")
    elif p_name == "groq":
        return GroqProvider(api_key=groq_api_key, model_name=model_name or "openai/gpt-oss-120b")
    elif p_name in ["openai_compatible", "qwen", "ollama", "vllm"]:
        return OpenAICompatibleProvider(base_url=custom_base_url or "http://localhost:11434/v1", model_name=model_name or "qwen2.5:7b")
    elif p_name == "sovereign_offline":
        return SovereignOfflineProvider()

    # 'auto': Prioritize Groq if GROQ_API_KEY present, then Gemini, then Sovereign
    groq = GroqProvider(api_key=groq_api_key, model_name=model_name or "openai/gpt-oss-120b")
    if groq.is_available():
        return groq

    gemini = GeminiProvider(api_key=gemini_api_key, model_name=model_name or "gemini-3.5-flash-lite")
    if gemini.is_available():
        return gemini

    logger.info("Auto provider: Defaulting to Sovereign Air-Gapped Cognitive Engine.")
    return SovereignOfflineProvider()
