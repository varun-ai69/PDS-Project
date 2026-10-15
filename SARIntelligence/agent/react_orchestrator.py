"""
SARIntelligence/agent/react_orchestrator.py
===========================================
Backwards-compatible bridge pointing to the decoupled LangGraph Agent.
"""

from .langgraph_orchestrator import SARTacticalLangGraphAgent, TacticalAgentState
from .tactical_tools import TacticalTools
from .llm_interface import get_llm_provider, BaseLLMProvider

SARTacticalAgent = SARTacticalLangGraphAgent

__all__ = [
    "SARTacticalAgent",
    "SARTacticalLangGraphAgent",
    "TacticalAgentState",
    "TacticalTools",
    "get_llm_provider",
    "BaseLLMProvider"
]
