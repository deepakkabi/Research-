"""
Cognitive Swarm: Multi-Agent Coordination Framework for RL Research

This package provides modules for studying scalable coordination,
robustness to communication failures, and safety constraint enforcement.

Example usage:
    from cognitive_swarm import CognitiveAgent
    
    agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7)
"""

from . import environment
from . import modules
from . import governance
from . import agents

# Top-level exports for convenience
from .agents.cognitive_agent import CognitiveAgent

__all__ = [
    "environment", 
    "modules", 
    "governance", 
    "agents",
    "CognitiveAgent",
]
