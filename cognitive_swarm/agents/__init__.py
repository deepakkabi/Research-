"""
Agents module - Contains the core CognitiveAgent integration class.

The CognitiveAgent implements the Secure Decision Pipeline that integrates:
- Trust Gate (Byzantine fault tolerance)
- Mean Field Encoder (Scalability)
- Safety Shield (Hard constraint verification)
- Policy Network (Decision making)
"""

from .cognitive_agent import CognitiveAgent

__all__ = ["CognitiveAgent"]
