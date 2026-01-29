"""
Hierarchical Mean Field (HMF) Encoder for Multi-Agent Reinforcement Learning

Academic Research Module for Cognitive Swarm Framework
Based on "Mean Field Multi-Agent RL" (Yang et al., 2018) with extensions
to heterogeneous agent teams.

Research Goal: Enable scalable coordination with 100+ agents by compressing
neighborhood information from O(N) to O(1) through role-based aggregation.
"""

import torch
import torch.nn as nn
from typing import Optional


class HMFEncoder(nn.Module):
    """
    Hierarchical Mean Field Encoder for scalable multi-agent coordination.
    
    Compresses neighbor observations from O(N) individual states to O(1) mean
    field representations grouped by agent role, enabling efficient coordination
    in large-scale heterogeneous multi-agent systems.
    
    Mathematical Foundation:
    Instead of processing all neighbors individually:
        Standard: obs_i = [state_1, state_2, ..., state_K]  # Variable O(N)
        Mean Field: obs_i = [μ_scout, μ_coordinator, μ_support]  # Fixed O(1)
    
    Where μ_role is the weighted mean of all neighbors of that role:
        μ_role = (1/|N_i^role|) Σ_{j ∈ N_i^role} w_j * state_j
    
    Args:
        state_dim (int): Dimension of neighbor state vectors. Default=6, matching
            the environment's self_state = [x, y, health, resource, role_id, team_id]
        num_roles (int): Number of agent role types. Default=3 for Scout(0),
            Coordinator(1), Support(2)
        use_hierarchical (bool): If True, also group by distance (near/far field).
            Creates 6 mean fields (3 roles × 2 distances) instead of 3.
    
    Input:
        neighbor_states: (batch, max_neighbors, state_dim=6) - Neighbor state vectors
        neighbor_roles: (batch, max_neighbors) - Role ID for each neighbor
        neighbor_mask: (batch, max_neighbors) - 1 if valid neighbor, 0 if padding
        trust_weights: (batch, max_neighbors) - Optional trust scores from Trust Gate
    
    Output:
        mean_field_encoding: (batch, num_roles * state_dim) - Fixed-size encoding
            Default: (batch, 18) for 3 roles × 6 state dimensions
    
    Example:
        >>> encoder = HMFEncoder(state_dim=6, num_roles=3)
        >>> neighbor_states = torch.randn(32, 50, 6)  # 32 agents, up to 50 neighbors
        >>> neighbor_roles = torch.randint(0, 3, (32, 50))
        >>> neighbor_mask = torch.ones(32, 50)
        >>> mean_field = encoder(neighbor_states, neighbor_roles, neighbor_mask)
        >>> mean_field.shape  # torch.Size([32, 18])
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        num_roles: int = 3,
        use_hierarchical: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_roles = num_roles
        self.use_hierarchical = use_hierarchical
        
        # Output dimension calculation
        if use_hierarchical:
            # 3 roles × 2 distance groups (near/far) × 6 state_dim = 36
            self.output_dim = num_roles * 2 * state_dim
            self.near_threshold = 3.0  # Distance threshold for near/far grouping
        else:
            # 3 roles × 6 state_dim = 18
            self.output_dim = num_roles * state_dim
        
        # Optional: Learnable aggregation weights (commented out by default)
        # Uncomment to use learned aggregation instead of simple averaging
        # self.use_learned_aggregation = False
        # if self.use_learned_aggregation:
        #     self.role_encoders = nn.ModuleList([
        #         nn.Linear(state_dim, state_dim) for _ in range(num_roles)
        #     ])
    
    def forward(
        self,
        neighbor_states: torch.Tensor,
        neighbor_roles: torch.Tensor,
        neighbor_mask: torch.Tensor,
        trust_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean field encoding from neighbor observations.
        
        Args:
            neighbor_states: (batch, max_neighbors, state_dim) - Neighbor state vectors
            neighbor_roles: (batch, max_neighbors) - Role ID (0=Scout, 1=Coordinator, 2=Support)
            neighbor_mask: (batch, max_neighbors) - 1 if valid, 0 if padding
            trust_weights: (batch, max_neighbors) - Optional trust scores [0, 1]
        
        Returns:
            mean_field: (batch, num_roles * state_dim) - Role-aggregated encoding
        
        Edge Cases Handled:
            1. No neighbors at all → returns zero vector
            2. No neighbors of specific role → zero vector for that role's mean field
            3. All neighbors have zero trust → falls back to zero (can be detected downstream)
        """
        batch_size = neighbor_states.size(0)
        max_neighbors = neighbor_states.size(1)
        
        # Validate input dimensions
        assert neighbor_states.size(2) == self.state_dim, \
            f"Expected state_dim={self.state_dim}, got {neighbor_states.size(2)}"
        
        # If no trust weights provided, use uniform weights (all neighbors equally trusted)
        if trust_weights is None:
            trust_weights = neighbor_mask.float()
        else:
            # Ensure trust weights respect the neighbor mask
            trust_weights = trust_weights * neighbor_mask.float()
        
        # Initialize output storage
        mean_fields = []
        
        # Process each role type
        for role_id in range(self.num_roles):
            # Create mask for neighbors of this specific role
            # role_mask: (batch, max_neighbors) - 1 if neighbor has this role, 0 otherwise
            role_mask = (neighbor_roles == role_id).float() * neighbor_mask.float()
            
            # Combine role mask with trust weights
            # This excludes both: (a) neighbors of other roles, (b) untrusted neighbors
            weights = role_mask * trust_weights  # (batch, max_neighbors)
            
            # Compute weighted average of neighbor states
            # weights: (batch, max_neighbors)
            # neighbor_states: (batch, max_neighbors, state_dim)
            
            # Sum of weights for normalization
            weight_sum = weights.sum(dim=1, keepdim=True)  # (batch, 1)
            
            # Avoid division by zero (happens when no neighbors of this role exist)
            # Use small epsilon to maintain numerical stability
            weight_sum = torch.clamp(weight_sum, min=1e-8)
            
            # Compute weighted mean: Σ(w_j * state_j) / Σ(w_j)
            # Step 1: Broadcast weights to match state dimensions
            weights_expanded = weights.unsqueeze(-1)  # (batch, max_neighbors, 1)
            
            # Step 2: Element-wise multiply and sum over neighbors
            weighted_sum = (weights_expanded * neighbor_states).sum(dim=1)  # (batch, state_dim)
            
            # Step 3: Normalize by weight sum
            mean_field = weighted_sum / weight_sum  # (batch, state_dim)
            
            # If no neighbors of this role exist, mean_field will be near-zero
            # (weighted_sum ≈ 0, divided by epsilon ≈ 0)
            # This is the desired behavior: empty role groups contribute zero to encoding
            
            mean_fields.append(mean_field)
        
        # Concatenate all role mean fields into single vector
        output = torch.cat(mean_fields, dim=-1)  # (batch, num_roles * state_dim)
        
        return output
    
    def forward_hierarchical(
        self,
        neighbor_states: torch.Tensor,
        neighbor_roles: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_distances: torch.Tensor,
        trust_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extended version with distance-based grouping (near/far field).
        
        Creates mean fields for each (role, distance_group) combination:
            - Near field: neighbors within near_threshold (default 3 cells)
            - Far field: neighbors beyond near_threshold
        
        Args:
            neighbor_distances: (batch, max_neighbors) - Euclidean distance to each neighbor
        
        Returns:
            mean_field: (batch, num_roles * 2 * state_dim) - 6 mean fields for 3 roles × 2 distances
        
        Note: Only use this if use_hierarchical=True was set in __init__
        """
        if not self.use_hierarchical:
            raise ValueError("forward_hierarchical() requires use_hierarchical=True in __init__")
        
        batch_size = neighbor_states.size(0)
        
        # If no trust weights, use uniform
        if trust_weights is None:
            trust_weights = neighbor_mask.float()
        else:
            trust_weights = trust_weights * neighbor_mask.float()
        
        mean_fields = []
        
        # Distance groups: near (< threshold) and far (>= threshold)
        near_mask = (neighbor_distances < self.near_threshold).float()
        far_mask = (neighbor_distances >= self.near_threshold).float()
        
        # For each role
        for role_id in range(self.num_roles):
            role_mask = (neighbor_roles == role_id).float() * neighbor_mask.float()
            
            # Near field for this role
            near_weights = role_mask * near_mask * trust_weights
            near_sum = near_weights.sum(dim=1, keepdim=True)
            near_sum = torch.clamp(near_sum, min=1e-8)
            near_field = (near_weights.unsqueeze(-1) * neighbor_states).sum(dim=1) / near_sum
            mean_fields.append(near_field)
            
            # Far field for this role
            far_weights = role_mask * far_mask * trust_weights
            far_sum = far_weights.sum(dim=1, keepdim=True)
            far_sum = torch.clamp(far_sum, min=1e-8)
            far_field = (far_weights.unsqueeze(-1) * neighbor_states).sum(dim=1) / far_sum
            mean_fields.append(far_field)
        
        output = torch.cat(mean_fields, dim=-1)  # (batch, num_roles * 2 * state_dim)
        return output


class LearnedHMFEncoder(nn.Module):
    """
    Variant of HMF Encoder with learnable aggregation weights.
    
    Instead of simple averaging, uses a small MLP to compute importance weights
    for each neighbor based on its state. This adds expressiveness at the cost
    of additional parameters.
    
    Aggregation: μ_role = Σ(α_j * state_j) where α_j = softmax(MLP(state_j))
    
    Trade-offs vs. Standard HMFEncoder:
        Pros: More expressive, can learn which state features matter most
        Cons: More parameters (risk of overfitting), slower computation
    
    Recommended use: When you have abundant training data and need to capture
    complex neighbor importance patterns.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        num_roles: int = 3,
        hidden_dim: int = 32
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_roles = num_roles
        self.output_dim = num_roles * state_dim
        
        # Learnable attention mechanism for each role
        self.role_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_roles)
        ])
    
    def forward(
        self,
        neighbor_states: torch.Tensor,
        neighbor_roles: torch.Tensor,
        neighbor_mask: torch.Tensor,
        trust_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute mean field with learned attention weights."""
        batch_size = neighbor_states.size(0)
        
        if trust_weights is None:
            trust_weights = neighbor_mask.float()
        else:
            trust_weights = trust_weights * neighbor_mask.float()
        
        mean_fields = []
        
        for role_id in range(self.num_roles):
            role_mask = (neighbor_roles == role_id).float() * neighbor_mask.float()
            
            # Compute attention scores using learned MLP
            attention_scores = self.role_attention[role_id](neighbor_states).squeeze(-1)
            # (batch, max_neighbors)
            
            # Mask out non-role and padding neighbors
            attention_scores = attention_scores * role_mask
            
            # Apply softmax to get attention weights (only over valid neighbors)
            # Need to handle case where all scores are masked (no neighbors of this role)
            exp_scores = torch.exp(attention_scores - attention_scores.max(dim=1, keepdim=True)[0])
            exp_scores = exp_scores * role_mask
            attention_weights = exp_scores / (exp_scores.sum(dim=1, keepdim=True) + 1e-8)
            
            # Combine with trust weights
            weights = attention_weights * trust_weights
            weight_sum = weights.sum(dim=1, keepdim=True)
            weight_sum = torch.clamp(weight_sum, min=1e-8)
            
            # Weighted average
            mean_field = (weights.unsqueeze(-1) * neighbor_states).sum(dim=1) / weight_sum
            mean_fields.append(mean_field)
        
        output = torch.cat(mean_fields, dim=-1)
        return output


class HMFEncoderWithAttention(nn.Module):
    """
    Enhanced HMF Encoder with Self-Attention mechanism (PROMPT 2 compliance).
    
    This variant adds Transformer-style attention for scenarios where
    learned aggregation outperforms simple averaging. Configurable to
    maintain backward compatibility with mean-field baseline.
    
    Architecture adds:
    - Multi-head self-attention for message processing
    - Layer normalization for stable training  
    - MLP with GELU activation
    - Positional encoding for spatial awareness
    
    Research Context:
    Supports ablation studies comparing simple mean-field (O(1)) vs 
    attention-based aggregation (O(N²)). Maintains both approaches for
    framework integration paper requirements.
    
    Args:
        state_dim: Dimension of neighbor state vectors (default=6)
        num_roles: Number of agent types (default=3)
        hidden_dim: Hidden dimension for attention (default=64)
        num_heads: Number of attention heads (default=4)
        use_attention: If False, falls back to mean field averaging
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        num_roles: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 4,
        use_attention: bool = True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_roles = num_roles
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        if use_attention:
            # Positional encoding for spatial information
            self.pos_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim),  # (x, y) -> embedding
                nn.ReLU()
            )
            
            # Multi-head self-attention (PROMPT 2 requirement)
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            
            # Layer normalization (PROMPT 2 requirement)
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            
            # MLP with GELU activation (PROMPT 2 requirement)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            
            # Project state to hidden dim
            self.state_proj = nn.Linear(state_dim, hidden_dim)
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dim * num_roles, num_roles * state_dim)
            self.output_dim = num_roles * state_dim
        else:
            # Fall back to simple mean field (original implementation)
            self.output_dim = num_roles * state_dim
            self.mean_field_encoder = HMFEncoder(state_dim, num_roles, use_hierarchical=False)
    
    def forward(
        self,
        neighbor_states: torch.Tensor,
        neighbor_roles: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_positions: Optional[torch.Tensor] = None,
        trust_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional attention mechanism.
        
        Args:
            neighbor_states: (batch, max_neighbors, state_dim)
            neighbor_roles: (batch, max_neighbors)
            neighbor_mask: (batch, max_neighbors)
            neighbor_positions: (batch, max_neighbors, 2) - for positional encoding
            trust_weights: (batch, max_neighbors) - from Trust Gate
        
        Returns:
            encoding: (batch, num_roles * state_dim)
        """
        if not self.use_attention:
            # Use original mean field implementation
            return self.mean_field_encoder(
                neighbor_states, neighbor_roles, neighbor_mask, trust_weights
            )
        
        batch_size = neighbor_states.size(0)
        
        # Project states to hidden dimension
        x = self.state_proj(neighbor_states)  # (batch, neighbors, hidden)
        
        # Add positional encoding if available (PROMPT 2 requirement)
        if neighbor_positions is not None:
            pos_enc = self.pos_encoder(neighbor_positions)
            x = x + pos_enc
        
        # Apply self-attention with masking
        attn_mask = ~neighbor_mask.bool()  # Invert mask (True = ignore)
        x_attn, _ = self.self_attention(x, x, x, key_padding_mask=attn_mask)
        
        # Residual connection + layer norm
        x = self.layer_norm1(x + x_attn)
        
        # MLP with residual
        x_mlp = self.mlp(x)
        x = self.layer_norm2(x + x_mlp)
        
        # Apply trust weights if available
        if trust_weights is not None:
            x = x * trust_weights.unsqueeze(-1)
        
        # Aggregate by role (mean pooling within each role)
        role_features = []
        for role_id in range(self.num_roles):
            role_mask = (neighbor_roles == role_id) & neighbor_mask
            role_x = x * role_mask.unsqueeze(-1).float()
            role_count = role_mask.sum(dim=1, keepdim=True).clamp(min=1)
            role_mean = role_x.sum(dim=1) / role_count.unsqueeze(-1)
            role_features.append(role_mean)
        
        aggregated = torch.cat(role_features, dim=-1)
        
        # Project to output dimension
        output = self.output_proj(aggregated)
        
        return output
