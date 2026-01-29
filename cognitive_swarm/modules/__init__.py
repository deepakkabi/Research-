"""
Cognitive Swarm Modules

Neural network modules for multi-agent reinforcement learning:
- HMFEncoder: Heterogeneous Mean Field encoder for agent communication
- LearnedHMFEncoder: Learnable version of HMF encoder
- TrustGate: Byzantine fault tolerance and message reliability verification
- BayesianBeliefModule: Opponent strategy inference (V2)
"""

from .hmf_encoder import HMFEncoder, LearnedHMFEncoder
from .trust_gate import TrustGate, SimpleTrustGate
from .bayesian_beliefs import BayesianBeliefModule

__all__ = [
    "HMFEncoder",
    "LearnedHMFEncoder", 
    "TrustGate",
    "SimpleTrustGate",
    "BayesianBeliefModule"
]
