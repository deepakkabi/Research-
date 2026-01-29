"""
Trust Gate: Message Reliability Verification for Multi-Agent Systems

Research Application: Byzantine fault tolerance in distributed multi-agent systems.
Handles communication failures, faulty agents, and noisy channels.

Integration:
- Input: batched messages from CoordinationEnv.collate_observations()
- Output: reliability_weights for HMFEncoder.forward(trust_weights=...)
- Calibrated for environment's σ=2.0 Gaussian noise (20% corruption rate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, Optional, List


class TrustGate(nn.Module):
    """
    Message reliability verification for multi-agent systems.
    
    Zero Trust Principle: "Never trust, always verify" - every message must be 
    validated against local observations and graph structure.
    
    Architecture:
        1. Consistency Checking: Does message match expected content?
        2. Graph Attention: Is sender in a reliable network position?
        3. Fusion: Combine content-based and structure-based reliability
    
    Integrates with:
        - CoordinationEnv: Handles batched messages with shape (batch, max_neighbors, 8)
        - HMFEncoder: Outputs reliability_weights with shape (batch, max_neighbors)
    
    Args:
        message_dim: Dimension of message vectors (default=8, from environment)
        hidden_dim: Dimension for GAT hidden layers
        num_heads: Number of attention heads in GAT
        consistency_threshold: Minimum consistency score to accept message
                              (default=0.5, calibrated for environment's σ=2.0 noise)
    """
    
    def __init__(
        self, 
        message_dim: int = 8, 
        hidden_dim: int = 64, 
        num_heads: int = 4, 
        consistency_threshold: float = 0.5
    ):
        super().__init__()
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.threshold = consistency_threshold
        self.num_heads = num_heads
        
        # Component 1: Consistency predictor
        # Predicts what each neighbor SHOULD say based on local observations
        # Input: message (8) + local context (10) = 18 dimensions
        self.message_predictor = nn.Sequential(
            nn.Linear(message_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Component 2: Graph Attention Network (structural reliability)
        # Uses graph topology to assess sender reliability
        self.gat = GATConv(
            in_channels=message_dim,
            out_channels=message_dim,
            heads=num_heads,
            concat=False,
            dropout=0.1,
            add_self_loops=True
        )
        
        # Component 3: Reliability fusion
        # Combines consistency and attention signals
        self.fusion = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Component 4: Adaptive thresholding (optional)
        # Allows learning threshold adjustments
        self.threshold_adjust = nn.Parameter(torch.zeros(1))
    
    def forward(
        self, 
        messages: torch.Tensor,           # (batch, num_neighbors, msg_dim=8)
        local_obs: torch.Tensor,          # (batch, obs_dim=10)
        edge_index: torch.Tensor,         # (2, num_edges) - graph structure
        neighbor_ids: torch.Tensor        # (batch, num_neighbors)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter messages based on reliability verification.
        
        Args:
            messages: Neighbor messages from environment
            local_obs: Local observations for prediction
            edge_index: Graph connectivity (COO format)
            neighbor_ids: IDs of neighbors for each agent
        
        Returns:
            filtered_messages: (batch, msg_dim) - weighted sum of reliable messages
            reliability_weights: (batch, num_neighbors) - reliability scores
                                These feed into HMFEncoder as trust_weights
        """
        batch_size = messages.size(0)
        num_neighbors = messages.size(1)
        
        # === STEP 1: Consistency Checking ===
        # Predict what each neighbor SHOULD say based on local observations
        consistency_weights = self._compute_consistency(messages, local_obs)
        # Shape: (batch, num_neighbors)
        
        # === STEP 2: Graph Attention (Structural Reliability) ===
        attention_weights = self._compute_graph_attention(messages, edge_index, batch_size, num_neighbors)
        # Shape: (batch, num_neighbors)
        
        # === STEP 3: Combine Reliability Signals ===
        reliability_weights = self._fuse_reliability_signals(consistency_weights, attention_weights)
        # Shape: (batch, num_neighbors)
        
        # === STEP 4: Filter Messages ===
        filtered_messages = self._filter_messages(messages, reliability_weights)
        # Shape: (batch, msg_dim)
        
        return filtered_messages, reliability_weights
    
    def _compute_consistency(
        self, 
        messages: torch.Tensor, 
        local_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute content-based consistency scores.
        
        Measures how well each message matches expected content based on
        local observations. Uses cosine similarity with learned predictor.
        
        Calibration: threshold=0.5 is tuned for environment's σ=2.0 noise
        """
        batch_size = messages.size(0)
        num_neighbors = messages.size(1)
        
        # Expand local_obs to match each neighbor
        local_obs_expanded = local_obs.unsqueeze(1).expand(-1, num_neighbors, -1)
        # Shape: (batch, num_neighbors, 10)
        
        # Concatenate messages with local context for prediction
        predictor_input = torch.cat([messages, local_obs_expanded], dim=-1)
        # Shape: (batch, num_neighbors, 18)
        
        # Predict expected messages
        expected_messages = self.message_predictor(predictor_input)
        # Shape: (batch, num_neighbors, msg_dim)
        
        # Compute consistency using cosine similarity
        # Cosine similarity is robust to magnitude variations
        consistency_scores = F.cosine_similarity(
            messages, 
            expected_messages, 
            dim=-1
        )  # Shape: (batch, num_neighbors)
        
        # Apply adaptive threshold
        effective_threshold = self.threshold + torch.sigmoid(self.threshold_adjust) * 0.2 - 0.1
        
        # Binary gating with soft edges
        consistency_weights = torch.sigmoid(
            (consistency_scores - effective_threshold) * 10.0
        )
        
        return consistency_weights
    
    def _compute_graph_attention(
        self, 
        messages: torch.Tensor, 
        edge_index: torch.Tensor,
        batch_size: int,
        num_neighbors: int
    ) -> torch.Tensor:
        """
        Compute structure-based reliability using Graph Attention Network.
        
        Assesses sender reliability based on network position and connectivity.
        Agents in central positions may be more reliable than isolated ones.
        """
        # Reshape for GAT: (batch * num_neighbors, msg_dim)
        messages_flat = messages.reshape(-1, self.message_dim)
        
        # Apply Graph Attention Network
        gat_output = self.gat(messages_flat, edge_index)
        # Shape: (batch * num_neighbors, msg_dim)
        
        # Reshape back to batch format
        gat_output = gat_output.view(batch_size, num_neighbors, self.message_dim)
        
        # Compute attention scores from GAT output magnitude
        # Higher magnitude indicates more important/reliable messages
        attention_scores = torch.norm(gat_output, dim=-1)
        # Shape: (batch, num_neighbors)
        
        # Normalize to [0, 1] range
        attention_weights = torch.sigmoid(attention_scores)
        
        return attention_weights
    
    def _fuse_reliability_signals(
        self, 
        consistency_weights: torch.Tensor, 
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine content-based and structure-based reliability scores.
        
        Uses learned fusion to balance:
        - Consistency: Does message content make sense?
        - Attention: Is sender in a reliable position?
        
        Returns normalized weights that sum to 1 for interpretability.
        """
        # Stack both signals
        reliability_signals = torch.stack([consistency_weights, attention_weights], dim=-1)
        # Shape: (batch, num_neighbors, 2)
        
        # Learned fusion
        combined_reliability = self.fusion(reliability_signals).squeeze(-1)
        # Shape: (batch, num_neighbors)
        
        # Apply softmax normalization
        # This ensures weights sum to 1 (required by HMFEncoder)
        reliability_weights = F.softmax(combined_reliability, dim=-1)
        
        return reliability_weights
    
    def _filter_messages(
        self, 
        messages: torch.Tensor, 
        reliability_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply reliability weights to filter messages.
        
        Computes weighted sum where unreliable messages contribute less.
        """
        # Weighted sum: reliable messages have higher contribution
        filtered_messages = (messages * reliability_weights.unsqueeze(-1)).sum(dim=1)
        # Shape: (batch, msg_dim)
        
        return filtered_messages
    
    def detect_faulty_agents(
        self, 
        reliability_weights: torch.Tensor, 
        threshold: float = 0.1
    ) -> List[int]:
        """
        Identify agents with consistently low reliability scores.
        
        This can be used for network diagnosis and recovery:
        - Isolate faulty agents from critical paths
        - Request retransmission from unreliable sources
        - Trigger repair mechanisms
        
        Args:
            reliability_weights: (batch, num_neighbors) - history of reliability scores
            threshold: If average reliability < threshold, flag as faulty
        
        Returns:
            faulty_ids: List of neighbor indices to isolate
        """
        # Average reliability over batch (time dimension)
        avg_reliability = reliability_weights.mean(dim=0)
        
        # Find agents below threshold
        faulty = (avg_reliability < threshold).nonzero(as_tuple=True)[0]
        
        return faulty.tolist()
    
    def get_reliability_stats(
        self, 
        reliability_weights: torch.Tensor
    ) -> dict:
        """
        Compute statistics about network reliability.
        
        Useful for monitoring and debugging.
        """
        return {
            'mean_reliability': reliability_weights.mean().item(),
            'std_reliability': reliability_weights.std().item(),
            'min_reliability': reliability_weights.min().item(),
            'max_reliability': reliability_weights.max().item(),
            'num_reliable': (reliability_weights > 0.5).sum().item(),
            'num_unreliable': (reliability_weights < 0.1).sum().item()
        }


class SimpleTrustGate(nn.Module):
    """
    Simplified Trust Gate for baseline comparisons.
    
    Uses only consistency checking without graph attention.
    Useful for ablation studies.
    """
    
    def __init__(
        self, 
        message_dim: int = 8, 
        hidden_dim: int = 64, 
        consistency_threshold: float = 0.5
    ):
        super().__init__()
        self.message_dim = message_dim
        self.threshold = consistency_threshold
        
        self.message_predictor = nn.Sequential(
            nn.Linear(message_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
    
    def forward(
        self, 
        messages: torch.Tensor,
        local_obs: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        neighbor_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple consistency-only filtering."""
        batch_size = messages.size(0)
        num_neighbors = messages.size(1)
        
        # Expand local observations
        local_obs_expanded = local_obs.unsqueeze(1).expand(-1, num_neighbors, -1)
        
        # Predict expected messages
        predictor_input = torch.cat([messages, local_obs_expanded], dim=-1)
        expected_messages = self.message_predictor(predictor_input)
        
        # Compute consistency
        consistency_scores = F.cosine_similarity(messages, expected_messages, dim=-1)
        
        # Apply threshold
        reliability_weights = torch.where(
            consistency_scores > self.threshold,
            consistency_scores,
            torch.zeros_like(consistency_scores)
        )
        
        # Normalize
        reliability_weights = F.softmax(reliability_weights, dim=-1)
        
        # Filter messages
        filtered_messages = (messages * reliability_weights.unsqueeze(-1)).sum(dim=1)
        
        return filtered_messages, reliability_weights


class TrustGateWithEMA(TrustGate):
    """
    Enhanced Trust Gate with Exponential Moving Average and Adversarial Detection.
    
    Extends base Trust Gate with temporal tracking and explicit adversarial detection
    to fully satisfy PROMPT 4 requirements and research framework Component 4.
    
    New Features:
    - Temporal trust tracking via EMA (Exponential Moving Average)
    - Explicit adversarial agent detection with flagging
    - Decay factor (α/alpha) for trust score updates
    
    Research Context:
    Implements Zero-Trust Architecture (Component 4) with:
    1. Content-based verification (inherited from base)
    2. Structure-based verification (inherited from base)
    3. Temporal verification (NEW: EMA tracking)
    4. Adversarial detection (NEW: explicit flagging)
    
    Args:
        message_dim: Dimension of messages (default=8)
        hidden_dim: Hidden dimension (default=64)
        num_heads: Attention heads (default=4)
        consistency_threshold: Consistency cutoff (default=0.5)
        ema_decay: Decay factor α for EMA (default=0.9) - PROMPT 4 requirement
        adversarial_threshold: Threshold for adversarial detection (default=0.3)
    """
    
    def __init__(
        self,
        message_dim: int = 8,
        hidden_dim: int = 64,
        num_heads: int = 4,
        consistency_threshold: float = 0.5,
        ema_decay: float = 0.9,  # α (alpha) - decay factor
        adversarial_threshold: float = 0.3
    ):
        super().__init__(message_dim, hidden_dim, num_heads, consistency_threshold)
        
        # EMA parameters (PROMPT 4: exponential moving average)
        self.ema_decay = ema_decay  # α in literature (alpha parameter)
        self.adversarial_threshold = adversarial_threshold
        
        # Trust history storage (maps agent_id -> trust_score)
        # In production, this would be part of agent state
        self.trust_history = {}
        
        # Adversarial detection counters (PROMPT 4: adversarial detection)
        self.adversarial_detections = {}
        self.detection_window = 10  # Number of steps to track
    
    def forward_with_ema(
        self,
        messages: torch.Tensor,
        local_obs: torch.Tensor,
        edge_index: torch.Tensor,
        neighbor_ids: torch.Tensor,
        agent_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with EMA trust tracking and adversarial detection.
        
        Returns:
            filtered_messages: Weighted message aggregation
            reliability_weights: Current reliability scores (EMA-smoothed)
            adversarial_flags: Boolean tensor indicating detected adversaries
        """
        # Get current reliability from base Trust Gate
        filtered_messages, reliability_weights = super().forward(
            messages, local_obs, edge_index, neighbor_ids
        )
        
        # Apply EMA temporal smoothing (PROMPT 4 requirement)
        reliability_weights = self._apply_ema_smoothing(
            reliability_weights, neighbor_ids, agent_id
        )
        
        # Detect adversarial agents (PROMPT 4 requirement)
        adversarial_flags = self._detect_adversaries(
            reliability_weights, neighbor_ids, agent_id
        )
        
        return filtered_messages, reliability_weights, adversarial_flags
    
    def _apply_ema_smoothing(
        self,
        current_reliability: torch.Tensor,
        neighbor_ids: torch.Tensor,
        agent_id: int
    ) -> torch.Tensor:
        """
        Apply Exponential Moving Average to trust scores.
        
        Formula: trust_t = α * trust_{t-1} + (1 - α) * reliability_t
        
        Where:
        - α (alpha) = ema_decay = 0.9 (decay factor) - PROMPT 4 requirement
        - trust_{t-1} = previous trust score
        - reliability_t = current measurement
        
        This provides temporal smoothing to avoid trust score oscillation
        from noisy observations.
        """
        batch_size = current_reliability.size(0)
        smoothed = torch.zeros_like(current_reliability)
        
        for batch_idx in range(batch_size):
            for neighbor_idx in range(neighbor_ids.size(1)):
                neighbor_id = int(neighbor_ids[batch_idx, neighbor_idx])
                
                # Get key for history lookup
                key = (agent_id, neighbor_id)
                
                if key in self.trust_history:
                    # Apply EMA update
                    prev_trust = self.trust_history[key]
                    current = current_reliability[batch_idx, neighbor_idx].item()
                    
                    # EMA formula with decay factor α
                    new_trust = (
                        self.ema_decay * prev_trust + 
                        (1 - self.ema_decay) * current
                    )
                    
                    self.trust_history[key] = new_trust
                    smoothed[batch_idx, neighbor_idx] = new_trust
                else:
                    # First observation - initialize
                    trust = current_reliability[batch_idx, neighbor_idx].item()
                    self.trust_history[key] = trust
                    smoothed[batch_idx, neighbor_idx] = trust
        
        return smoothed
    
    def _detect_adversaries(
        self,
        reliability_weights: torch.Tensor,
        neighbor_ids: torch.Tensor,
        agent_id: int
    ) -> torch.Tensor:
        """
        Detect adversarial agents based on persistent low trust.
        
        Detection Criteria:
        1. Trust score < adversarial_threshold for sustained period
        2. Multiple consecutive low-trust observations (70% of window)
        
        This implements explicit adversarial detection as required by PROMPT 4
        for Zero-Trust architecture (Component 4 in research framework).
        
        Returns:
            adversarial_flags: (batch, max_neighbors) bool tensor
                              True = agent flagged as adversarial
        """
        batch_size = reliability_weights.size(0)
        adversarial_flags = torch.zeros_like(reliability_weights, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            for neighbor_idx in range(neighbor_ids.size(1)):
                neighbor_id = int(neighbor_ids[batch_idx, neighbor_idx])
                key = (agent_id, neighbor_id)
                
                # Check if trust is below threshold
                current_trust = reliability_weights[batch_idx, neighbor_idx].item()
                
                if current_trust < self.adversarial_threshold:
                    # Increment detection counter
                    if key not in self.adversarial_detections:
                        self.adversarial_detections[key] = []
                    
                    self.adversarial_detections[key].append(1)
                    
                    # Keep only recent window
                    if len(self.adversarial_detections[key]) > self.detection_window:
                        self.adversarial_detections[key].pop(0)
                    
                    # Flag if consistently low trust
                    recent_detections = sum(self.adversarial_detections[key])
                    if recent_detections >= self.detection_window * 0.7:  # 70% threshold
                        adversarial_flags[batch_idx, neighbor_idx] = True
                else:
                    # Reset counter if trust recovered
                    if key in self.adversarial_detections:
                        self.adversarial_detections[key].append(0)
                        if len(self.adversarial_detections[key]) > self.detection_window:
                            self.adversarial_detections[key].pop(0)
        
        return adversarial_flags
    
    def get_adversarial_stats(self) -> dict:
        """
        Return statistics on detected adversaries for research analysis.
        
        Useful for ablation studies and security validation in research paper.
        """
        total_tracked = len(self.trust_history)
        total_adversarial = sum(
            1 for key in self.adversarial_detections 
            if len(self.adversarial_detections.get(key, [])) > 0 and
            sum(self.adversarial_detections[key]) >= self.detection_window * 0.7
        )
        
        return {
            'total_agents_tracked': total_tracked,
            'adversarial_agents_detected': total_adversarial,
            'adversarial_rate': total_adversarial / max(total_tracked, 1),
            'ema_decay_factor': self.ema_decay,
            'detection_threshold': self.adversarial_threshold
        }
