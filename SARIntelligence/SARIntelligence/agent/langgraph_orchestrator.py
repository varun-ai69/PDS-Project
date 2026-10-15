"""
SARIntelligence/agent/langgraph_orchestrator.py
===============================================
Autonomous Tactical LangGraph Agent for Defense SAR Satellite Reconnaissance.
Implements a state graph with dynamic nodes:
  [START]
     │
     ▼
  1. [Planner Node] ────── (Calls pluggable LLM Brain to decide next action)
     │
     ├──► [CALL_TOOL] ────► 2. [Tool Executor Node] ──► 3. [Evaluator Node] ──┐
     │                                                                       │
     ▼                                                                       ▼
  4. [Synthesizer Node] ◄────────────────────────────────────────────────────┘
     │
     ▼
   [END]

Completely decoupled from any specific LLM:
You can plug in Gemini, Groq, fine-tuned Qwen (via OpenAICompatibleProvider),
or Sovereign Air-Gapped Engine with zero changes to this agent graph!
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from SARIntelligence.agent.tactical_tools import TacticalTools
from SARIntelligence.agent.llm_interface import (
    BaseLLMProvider,
    StepDecision,
    get_llm_provider,
)

logger = logging.getLogger("SARLangGraphAgent")
logging.basicConfig(level=logging.INFO)

# Check LangGraph availability
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    START, END, StateGraph = "START", "END", None


# =============================================================================
# 1. AGENT STATE DEFINITION
# =============================================================================

class TacticalAgentState(TypedDict):
    query: str
    is_strict_output: bool
    step_count: int
    max_steps: int
    reasoning_trajectory: List[Dict[str, Any]]
    current_decision: Optional[Dict[str, Any]]
    retrieved_data: List[Dict[str, Any]]
    evaluation_status: str  # 'CONTINUE', 'FINISH'
    final_response: str


# =============================================================================
# 2. LANGGRAPH AUTONOMOUS AGENT CONTROLLER
# =============================================================================

class SARTacticalLangGraphAgent:
    """
    Sovereign Multi-Node Defense LangGraph Agent.
    Orchestrates autonomous ReAct loops, tool executions, and evaluation.
    Plug-and-play with ANY LLM provider!
    """

    # Schema descriptions provided to the LLM
    TOOLS_SCHEMA = [
        {
            "tool_name": "query_database",
            "description": "Structured lookup by domain ('ground', 'ocean', 'multi_temporal'), sector (e.g. 'AKHNOOR', 'GALWAN'), target_class ('T72', 'BMP2', 'vessel', 'aircraft'), and max_targets=0 for negative constraints / safe zones / calm sea.",
            "parameters": ["domain", "sector", "target_class", "min_targets", "max_targets", "defcon_level", "min_lat", "max_lat", "min_lon", "max_lon", "limit"]
        },
        {
            "tool_name": "semantic_similarity_search",
            "description": "768-D SARCLIP vector retrieval in Qdrant. Supports natural language queries ('tanks along river') OR image file inputs for image-to-image similarity matching.",
            "parameters": ["query_text", "image_path", "domain_filter", "top_k"]
        },
        {
            "tool_name": "analyze_temporal_surge",
            "description": "Multi-temporal surveillance pass differencing. Compares past baseline T1 against current T2, measures troop delta (T2 - T1), and provides difference heatmaps.",
            "parameters": ["sector", "pair_number", "min_surge_delta", "limit"]
        },
        {
            "tool_name": "get_scene_telemetry",
            "description": "Deep inspection of Master Template JSON for a specific scene_id. Returns individual target bounding boxes, RCS in dBsm, and GPS coordinates.",
            "parameters": ["scene_id"]
        }
    ]

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        tools: Optional[TacticalTools] = None,
        max_steps: int = 3
    ):
        self.tools = tools or TacticalTools()
        self.llm_provider = llm_provider or get_llm_provider(provider_name="auto")
        self.max_steps = max_steps
        self.graph = self._compile_state_graph() if LANGGRAPH_AVAILABLE else None

    # -------------------------------------------------------------------------
    # GRAPH COMPILATION
    # -------------------------------------------------------------------------
    def _compile_state_graph(self):
        """Compiles the LangGraph StateGraph with dynamic nodes and conditional edges."""
        workflow = StateGraph(TacticalAgentState)

        # Add Nodes
        workflow.add_node("planner_node", self.node_planner)
        workflow.add_node("tool_executor_node", self.node_tool_executor)
        workflow.add_node("evaluator_node", self.node_evaluator)
        workflow.add_node("synthesizer_node", self.node_synthesizer)

        # Set Entry Point
        workflow.add_edge(START, "planner_node")

        # Conditional Edge after Planner
        def route_after_planner(state: TacticalAgentState) -> str:
            dec = state.get("current_decision") or {}
            if dec.get("decision_type") == "CALL_TOOL":
                return "tool_executor_node"
            return "synthesizer_node"

        workflow.add_conditional_edges(
            "planner_node",
            route_after_planner,
            {
                "tool_executor_node": "tool_executor_node",
                "synthesizer_node": "synthesizer_node"
            }
        )

        # Tool execution always flows to Evaluator
        workflow.add_edge("tool_executor_node", "evaluator_node")

        # Conditional Edge after Evaluator
        def route_after_evaluator(state: TacticalAgentState) -> str:
            if state.get("evaluation_status") == "CONTINUE" and state.get("step_count", 0) < state.get("max_steps", 3):
                return "planner_node"
            return "synthesizer_node"

        workflow.add_conditional_edges(
            "evaluator_node",
            route_after_evaluator,
            {
                "planner_node": "planner_node",
                "synthesizer_node": "synthesizer_node"
            }
        )

        # Synthesizer flows to END
        workflow.add_edge("synthesizer_node", END)

        return workflow.compile()

    # -------------------------------------------------------------------------
    # NODE 1: PLANNER NODE (Calls LLM Brain)
    # -------------------------------------------------------------------------
    def node_planner(self, state: TacticalAgentState) -> Dict[str, Any]:
        query = state["query"]
        trajectory = list(state.get("reasoning_trajectory", []))
        step_idx = state.get("step_count", 0) + 1

        # Check for strict output trigger words
        q_low = query.lower()
        is_strict = any(w in q_low for w in ["sirf", "only coordinate", "only image", "just give", "just coordinate", "direct coordinate"])

        # Call pluggable LLM Brain
        decision: StepDecision = self.llm_provider.plan_and_decide(
            query=query,
            reasoning_history=trajectory,
            available_tools=self.TOOLS_SCHEMA
        )

        logger.info("[STEP %d - PLANNER] Decision: %s | Thought: %s", step_idx, decision.decision_type, decision.thought)

        return {
            "step_count": step_idx,
            "is_strict_output": is_strict,
            "current_decision": decision.to_dict()
        }

    # -------------------------------------------------------------------------
    # NODE 2: TOOL EXECUTOR NODE (Executes One of the 4 Defense Tools)
    # -------------------------------------------------------------------------
    def node_tool_executor(self, state: TacticalAgentState) -> Dict[str, Any]:
        decision = state.get("current_decision") or {}
        tool_name = decision.get("tool_name")
        args = decision.get("tool_args", {})
        thought = decision.get("thought", "")
        step_idx = state.get("step_count", 1)

        observation = None
        retrieved_items = list(state.get("retrieved_data", []))

        logger.info("[STEP %d - TOOL EXECUTION] Invoking '%s' with args: %s", step_idx, tool_name, args)

        try:
            if tool_name == "query_database":
                observation = self.tools.query_database(**args)
                if isinstance(observation, list):
                    retrieved_items.extend(observation)

            elif tool_name == "semantic_similarity_search":
                observation = self.tools.semantic_similarity_search(**args)
                if isinstance(observation, list):
                    retrieved_items.extend(observation)

            elif tool_name == "analyze_temporal_surge":
                observation = self.tools.analyze_temporal_surge(**args)
                if isinstance(observation, list):
                    retrieved_items.extend(observation)

            elif tool_name == "get_scene_telemetry":
                observation = self.tools.get_scene_telemetry(**args)
                if isinstance(observation, dict):
                    retrieved_items.append(observation)

            else:
                logger.warning("Unknown tool '%s'. Defaulting to query_database.", tool_name)
                observation = self.tools.query_database(limit=5)
                retrieved_items.extend(observation)

        except Exception as e:
            logger.error("Tool execution failed: %s", e)
            observation = {"error": str(e)}

        # Format observation summary for trajectory
        obs_summary = ""
        if isinstance(observation, list):
            obs_summary = f"Retrieved {len(observation)} target records. Primary: {[item.get('scene_id') or item.get('pair_id') for item in observation[:3]]}"
        elif isinstance(observation, dict):
            obs_summary = f"Retrieved telemetry for {observation.get('scene_id')}: {observation.get('total_targets')} targets."

        trajectory_step = {
            "step": step_idx,
            "node": "TOOL_EXECUTOR",
            "thought": thought,
            "tool": tool_name,
            "args": args,
            "observation_summary": obs_summary,
            "observation": observation
        }

        new_trajectory = list(state.get("reasoning_trajectory", []))
        new_trajectory.append(trajectory_step)

        # Deduplicate retrieved items
        deduped = []
        seen = set()
        for item in retrieved_items:
            key = item.get("scene_id") or item.get("pair_id")
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)

        return {
            "reasoning_trajectory": new_trajectory,
            "retrieved_data": deduped
        }

    # -------------------------------------------------------------------------
    # NODE 3: EVALUATOR NODE (Reflects on Intelligence Sufficiency)
    # -------------------------------------------------------------------------
    def node_evaluator(self, state: TacticalAgentState) -> Dict[str, Any]:
        retrieved = state.get("retrieved_data", [])
        step_count = state.get("step_count", 1)
        max_steps = state.get("max_steps", 3)

        # If we have retrieved valid data or reached max steps, conclude
        if len(retrieved) > 0 or step_count >= max_steps:
            logger.info("[EVALUATOR] Intelligence requirement fulfilled with %d records. Proceeding to synthesis.", len(retrieved))
            return {"evaluation_status": "FINISH"}

        logger.info("[EVALUATOR] Insufficient data retrieved. Continuing ReAct loop.")
        return {"evaluation_status": "CONTINUE"}

    # -------------------------------------------------------------------------
    # NODE 4: SYNTHESIZER NODE (Synthesizes Final SITREP via LLM)
    # -------------------------------------------------------------------------
    def node_synthesizer(self, state: TacticalAgentState) -> Dict[str, Any]:
        query = state["query"]
        trajectory = state.get("reasoning_trajectory", [])
        retrieved = state.get("retrieved_data", [])
        is_strict = state.get("is_strict_output", False)

        logger.info("[SYNTHESIZER] Generating military intelligence response (Strict: %s)...", is_strict)

        final_text = self.llm_provider.synthesize_report(
            query=query,
            reasoning_history=trajectory,
            direct_data=retrieved,
            is_strict_output=is_strict
        )

        return {"final_response": final_text}

    # -------------------------------------------------------------------------
    # MAIN QUERY ENTRYPOINT (Runs StateGraph or Pure-Python Fallback)
    # -------------------------------------------------------------------------
    def run_query(self, query: str) -> Dict[str, Any]:
        """
        Executes autonomous investigation on commander's operational query.
        Returns complete reasoning trajectory, direct target data, and synthesized response.
        """
        initial_state: TacticalAgentState = {
            "query": query,
            "is_strict_output": False,
            "step_count": 0,
            "max_steps": self.max_steps,
            "reasoning_trajectory": [],
            "current_decision": None,
            "retrieved_data": [],
            "evaluation_status": "CONTINUE",
            "final_response": ""
        }

        # 1. Execute via compiled LangGraph if available
        if self.graph is not None:
            try:
                logger.info("Executing query via LangGraph StateGraph engine...")
                final_state = self.graph.invoke(initial_state)
                return {
                    "query": query,
                    "final_response": final_state.get("final_response", ""),
                    "reasoning_trajectory": final_state.get("reasoning_trajectory", []),
                    "retrieved_data": final_state.get("retrieved_data", []),
                    "total_steps": final_state.get("step_count", 1)
                }
            except Exception as e:
                logger.warning("LangGraph invocation note: %s. Executing via resilient state loop.", e)

        # 2. Resilient Pure-Python ReAct State Machine (Identical behavior, zero dependency issues)
        logger.info("Executing query via Resilient Sovereign ReAct State Machine...")
        state = dict(initial_state)
        while state["step_count"] < state["max_steps"]:
            # Planner
            planner_res = self.node_planner(state)
            state.update(planner_res)

            dec = state.get("current_decision") or {}
            if dec.get("decision_type") == "SYNTHESIZE":
                break

            # Tool Executor
            exec_res = self.node_tool_executor(state)
            state.update(exec_res)

            # Evaluator
            eval_res = self.node_evaluator(state)
            state.update(eval_res)
            if state.get("evaluation_status") == "FINISH":
                break

        # Synthesizer
        synth_res = self.node_synthesizer(state)
        state.update(synth_res)

        return {
            "query": query,
            "final_response": state.get("final_response", ""),
            "reasoning_trajectory": state.get("reasoning_trajectory", []),
            "retrieved_data": state.get("retrieved_data", []),
            "total_steps": state.get("step_count", 1)
        }

    def print_interactive_report(self, query: str) -> Dict[str, Any]:
        """Convenience method for terminal testing."""
        res = self.run_query(query)
        print("\n" + "="*80)
        print(f"COMMANDER QUERY: {query}")
        print("="*80)
        print("\n--- REASONING TRAJECTORY & TOOL TRACE ---")
        for s in res["reasoning_trajectory"]:
            print(f"[STEP {s.get('step')}] Tool: {s.get('tool')}")
            print(f"  Thought: {s.get('thought')}")
            print(f"  Args: {s.get('args')}")
            print(f"  Observation: {s.get('observation_summary')}\n")
        print("--- FINAL MILITARY SITREP ---")
        print(res["final_response"])
        print("="*80 + "\n")
        return res
