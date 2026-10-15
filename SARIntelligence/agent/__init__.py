"""
SARIntelligence/agent/__init__.py
=================================
Core Autonomous Tactical Agent and Pluggable LLM Framework for SAR Satellite Defense.
"""

from .tactical_tools import TacticalTools
from .llm_interface import (
    BaseLLMProvider,
    StepDecision,
    GeminiProvider,
    GroqProvider,
    OpenAICompatibleProvider,
    SovereignOfflineProvider,
    get_llm_provider,
)
from .langgraph_orchestrator import (
    SARTacticalLangGraphAgent,
    TacticalAgentState,
)

# Unified alias for standard usage
SARTacticalAgent = SARTacticalLangGraphAgent

__all__ = [
    "TacticalTools",
    "SARTacticalLangGraphAgent",
    "SARTacticalAgent",
    "TacticalAgentState",
    "BaseLLMProvider",
    "StepDecision",
    "GeminiProvider",
    "GroqProvider",
    "OpenAICompatibleProvider",
    "SovereignOfflineProvider",
    "get_llm_provider",
]
