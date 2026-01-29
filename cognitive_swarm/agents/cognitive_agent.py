# Prompt 6 (INTEGRATION)
"""
Cognitive Agent - Core Integration Module

This module implements the main agent class that integrates all components
into a unified "Secure Decision Pipeline" for multi-agent coordination.

The pipeline processes information through 6 stages:
1. OBSERVE: Receive raw observations + messages
2. ORIENT (Security): Filter unreliable messages via Trust Gate
3. ORIENT (Context): Update beliefs via Bayesian module [OPTIONAL - disabled for V1]
4. DECIDE (Scale): Aggregate via Mean Field encoder
5. ACT (Policy): Neural network proposes action
6. ACT (Safety): Safety module validates action

This is the core architectural contribution of the Cognitive Swarm framework.

Academic Research Disclaimer:
This implementation is for research on integrated multi-agent decision-making
architectures, addressing scalability, robustness, uncertainty, and safety.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np

# Import BayesianBeliefModule for V2 functionality
try:
    from cognitive_swarm.modules.bayesian_beliefs import BayesianBeliefModule
    BELIEFS_AVAILABLE = True
except ImportError:
    BELIEFS_AVAILABLE = False


class TrustGate(nn.Module):
    """Placeholder for Trust Gate - to be imported from modules.trust_gate"""
    def __init__(self, message_dim: int, hidden_dim: int = 64, 
                 num_heads: int = 4, consistency_threshold: float = 0.5):
        super().__init__()
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.threshold = consistency_threshold
        
        # Simplified implementation for standalone testing
        self.attention = nn.MultiheadAttention(message_dim, num_heads, batch_first=True)
        self.reliability_net = nn.Sequential(
            nn.Linear(message_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, messages, local_obs, edge_index=None, neighbor_ids=None):
        batch_size, num_neighbors, msg_dim = messages.shape
        
        # Simple attention-based filtering
        attn_out, _ = self.attention(messages, messages, messages)
        
        # Compute reliability weights based on consistency
        local_expanded = local_obs.unsqueeze(1).expand(-1, num_neighbors, -1)
        if local_expanded.size(-1) != msg_dim:
            # Adjust dimensions
            local_expanded = local_expanded[..., :msg_dim] if local_expanded.size(-1) > msg_dim else \
                            torch.cat([local_expanded, torch.zeros(batch_size, num_neighbors, 
                                     msg_dim - local_expanded.size(-1), device=messages.device)], dim=-1)
        
        combined = torch.cat([attn_out, local_expanded], dim=-1)
        reliability_weights = self.reliability_net(combined).squeeze(-1)
        
        # Apply threshold
        reliability_weights = reliability_weights * (reliability_weights > self.threshold).float()
        
        # Filtered messages (weighted average)
        filtered = (messages * reliability_weights.unsqueeze(-1)).sum(dim=1) / \
                   (reliability_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return filtered, reliability_weights


class HMFEncoder(nn.Module):
    """Placeholder for Hierarchical Mean Field Encoder - to be imported from modules.hmf_encoder"""
    def __init__(self, state_dim: int = 6, num_roles: int = 3, use_hierarchical: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.num_roles = num_roles
        self.output_dim = num_roles * state_dim
        self.use_hierarchical = use_hierarchical
    
    def forward(self, neighbor_states, neighbor_roles, neighbor_mask, trust_weights=None):
        batch_size = neighbor_states.size(0)
        device = neighbor_states.device
        
        # Initialize role-based aggregation
        role_aggregates = torch.zeros(batch_size, self.num_roles, self.state_dim, device=device)
        role_counts = torch.zeros(batch_size, self.num_roles, device=device)
        
        # Apply trust weights if provided
        if trust_weights is not None:
            weights = trust_weights.unsqueeze(-1) * neighbor_mask.unsqueeze(-1)
        else:
            weights = neighbor_mask.unsqueeze(-1)
        
        # Aggregate by role
        for role_id in range(self.num_roles):
            role_mask = (neighbor_roles == role_id).float() * weights.squeeze(-1)
            if role_mask.sum() > 0:
                weighted_states = neighbor_states * role_mask.unsqueeze(-1)
                role_aggregates[:, role_id] = weighted_states.sum(dim=1) / (role_mask.sum(dim=1, keepdim=True) + 1e-8)
                role_counts[:, role_id] = role_mask.sum(dim=1)
        
        # Flatten to (batch, num_roles * state_dim)
        mean_field = role_aggregates.reshape(batch_size, -1)
        
        return mean_field


class SafetyConstraintModule:
    """Placeholder for Safety Shield - to be imported from governance.shield"""
    def __init__(self, safe_distance: float = 5.0, proportionality_lambda: float = 1.0):
        self.safe_distance = safe_distance
        self.proportionality_lambda = proportionality_lambda
    
    def verify_action(self, proposed_action: int, state_dict: Dict, agent_id: int) -> Tuple[int, bool, str]:
        """
        Verify if proposed action satisfies safety constraints.
        
        Returns:
            final_action: int - action to execute (fallback to HOLD=6 if unsafe)
            violated: bool - whether constraints were violated
            reason: str - reason for blocking (if blocked)
        """
        INTERVENTION = 4
        HOLD = 6
        
        # Extract state information
        agent_pos = state_dict.get('agent_positions', np.array([[0, 0]]))[agent_id]
        protected_pos = state_dict.get('protected_positions', np.array([]))
        target_pos = state_dict.get('target_positions', np.array([]))
        agent_resource = state_dict.get('agent_resources', np.array([100]))[agent_id]
        target_values = state_dict.get('target_values', np.array([]))
        
        # Only check INTERVENTION actions
        if proposed_action != INTERVENTION:
            return proposed_action, False, ""
        
        # Constraint 1: Protected Entity Distance
        if len(protected_pos) > 0:
            distances = np.linalg.norm(protected_pos - agent_pos, axis=1)
            if np.any(distances < self.safe_distance):
                return HOLD, True, "Protected Entity Constraint"
        
        # Constraint 2: Proportionality (resource vs target value)
        if len(target_pos) > 0 and len(target_values) > 0:
            target_distances = np.linalg.norm(target_pos - agent_pos, axis=1)
            closest_target_idx = np.argmin(target_distances)
            target_value = target_values[closest_target_idx]
            
            if agent_resource > self.proportionality_lambda * target_value:
                return HOLD, True, "Proportionality Constraint"
        
        # Constraint 3: Resource Sufficiency
        if agent_resource < 1:
            return HOLD, True, "Resource Constraint"
        
        return proposed_action, False, ""


class CognitiveAgent(nn.Module):
    """
    Main agent class integrating all modules into the Secure Decision Pipeline.
    
    The agent processes information through these stages IN ORDER:
    1. OBSERVE: Receive raw observations + messages
    2. ORIENT (Security): Filter unreliable messages via Trust Gate
    3. ORIENT (Context): Update beliefs via Bayesian module [OPTIONAL]
    4. DECIDE (Scale): Aggregate via Mean Field encoder
    5. ACT (Policy): Neural network proposes action
    6. ACT (Safety): Safety module validates action
    
    This ordering is critical:
    - Trust Gate FIRST: Corrupted data poisons all downstream processing
    - Beliefs SECOND: Context for aggregation (know opponent strategy before planning)
    - Mean Field THIRD: Uses reliable messages for scalable aggregation
    - Safety LAST: Hard guarantee before execution
    
    Args:
        obs_dim: Local observation dimension (default: 10)
        message_dim: Message vector dimension (default: 8)
        state_dim: Neighbor state dimension (default: 6) - [x, y, health, resource, role_id, team_id]
        action_dim: Number of discrete actions (default: 7)
        num_roles: Number of agent roles (default: 3) - Scout, Coordinator, Support
        num_strategy_types: Number of opponent strategies (default: 4)
        hidden_dim: Hidden layer dimension for policy/value networks (default: 128)
        use_beliefs: Enable Bayesian belief module (default: False for V1)
        use_world_model: Enable world model for planning (default: False for V1)
    
    Dimensional Flow:
        obs_dim=10 + mean_field_dim=18 + belief_dim=0 (V1) = policy_input_dim=28
        where mean_field_dim = num_roles * state_dim = 3 * 6 = 18
    """
    
    def __init__(self, 
                 obs_dim: int = 10,
                 message_dim: int = 8,
                 state_dim: int = 6,
                 action_dim: int = 7,
                 num_roles: int = 3,
                 num_strategy_types: int = 4,
                 hidden_dim: int = 128,
                 use_beliefs: bool = False,
                 use_world_model: bool = False):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_roles = num_roles
        self.use_beliefs = use_beliefs
        self.use_world_model = use_world_model
        self.hidden_dim = hidden_dim
        
        # ============================================================
        # MODULE 1: Trust Gate (Message Reliability)
        # Filters corrupted/Byzantine messages before processing
        # ============================================================
        self.trust_gate = TrustGate(
            message_dim=message_dim,
            hidden_dim=64,
            num_heads=4,
            consistency_threshold=0.5
        )
        
        # ============================================================
        # MODULE 2: Bayesian Beliefs (Strategy Inference) - OPTIONAL
        # Infers opponent strategy from observation sequences
        # V2: Now fully implemented with neural likelihood networks
        # ============================================================
        self.num_strategy_types = num_strategy_types
        if use_beliefs:
            if BELIEFS_AVAILABLE:
                self.belief_module = BayesianBeliefModule(
                    obs_dim=obs_dim,
                    num_types=num_strategy_types,
                    hidden_dim=64,
                    use_neural=True,  # V2: Neural networks
                    update_frequency=1  # V2: Every step
                )
                belief_contrib = num_strategy_types
                print(f"✓ Bayesian beliefs enabled ({num_strategy_types} types)")
            else:
                print("WARNING: BayesianBeliefModule not available. Disabling beliefs.")
                self.belief_module = None
                belief_contrib = 0
        else:
            self.belief_module = None
            belief_contrib = 0
        
        # ============================================================
        # MODULE 3: Mean Field Encoder (Scalability)
        # Aggregates neighbor information: O(N) → O(1)
        # ============================================================
        self.hmf_encoder = HMFEncoder(
            state_dim=state_dim,  # CRITICAL: Use state_dim=6, NOT obs_dim=10
            num_roles=num_roles,
            use_hierarchical=False  # Set True for journal version
        )
        
        # ============================================================
        # MODULE 4: World Model (Foresight) - SKIPPED FOR V1
        # Would predict future states for planning
        # ============================================================
        if use_world_model:
            raise NotImplementedError("World model skipped for Version 1")
        
        # ============================================================
        # MODULE 5: Policy Network (Decision Making)
        # Maps processed state to action logits
        # ============================================================
        # Compute policy input dimension
        mean_field_dim = num_roles * state_dim  # 3 * 6 = 18
        policy_input_dim = obs_dim + mean_field_dim + belief_contrib
        # V1: 10 + 18 + 0 = 28
        
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # ============================================================
        # MODULE 6: Safety Constraints (Operational Rules)
        # Verifies actions satisfy hard constraints
        # ============================================================
        self.safety_module = SafetyConstraintModule(
            safe_distance=5.0,
            proportionality_lambda=1.0
        )
        
        # ============================================================
        # Value Network (for Actor-Critic RL)
        # Estimates state value for advantage computation
        # ============================================================
        self.value_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        print(f"[CognitiveAgent] Initialized with policy_input_dim={policy_input_dim}")
        print(f"  - obs_dim: {obs_dim}")
        print(f"  - mean_field_dim: {mean_field_dim} ({num_roles} roles × {state_dim} state_dim)")
        print(f"  - belief_dim: {belief_contrib}")
        print(f"  - use_beliefs: {use_beliefs}, use_world_model: {use_world_model}")
    
    def forward(self, 
                local_obs: torch.Tensor,               # (batch, obs_dim=10)
                messages: torch.Tensor,                # (batch, num_neighbors, msg_dim=8)
                neighbor_states: torch.Tensor,         # (batch, num_neighbors, state_dim=6)
                neighbor_roles: torch.Tensor,          # (batch, num_neighbors) - role IDs
                neighbor_mask: torch.Tensor,           # (batch, num_neighbors) - valid neighbors
                edge_index: Optional[torch.Tensor] = None,  # (2, num_edges) - graph structure
                neighbor_ids: Optional[torch.Tensor] = None,  # (batch, num_neighbors) - IDs
                other_team_obs: Optional[torch.Tensor] = None,  # (batch, seq_len, obs_dim) - for beliefs
                state_dict: Optional[Dict] = None      # For safety module (not used in forward)
               ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Execute the Secure Decision Pipeline.
        
        Processing flow:
        1. Trust Gate filters messages → reliability_weights
        2. [Bayesian Beliefs updates strategy inference] - SKIPPED IN V1
        3. Mean Field aggregates neighbors using reliability weights → mean_field
        4. Policy network proposes action → action_logits
        5. [Safety verification happens in select_action(), not here]
        
        Returns:
            action_logits: (batch, action_dim=7) - raw policy output
            value: (batch, 1) - state value estimate for RL
            info: dict - intermediate outputs for logging/analysis
        """
        batch_size = local_obs.size(0)
        
        # ============================================================
        # STAGE 1: TRUST GATE (Message Reliability)
        # Filter erroneous/corrupted/Byzantine messages
        # ============================================================
        filtered_messages, reliability_weights = self.trust_gate(
            messages=messages,
            local_obs=local_obs,
            edge_index=edge_index,
            neighbor_ids=neighbor_ids
        )
        # Output:
        #   filtered_messages: (batch, msg_dim=8) - aggregated reliable messages
        #   reliability_weights: (batch, num_neighbors) - trust score per neighbor
        
        # ============================================================
        # STAGE 2: BAYESIAN BELIEFS (Strategy Uncertainty) - OPTIONAL
        # Infer opponent strategy from observation sequences
        # SKIPPED IN V1
        # ============================================================
        if self.use_beliefs and self.belief_module is not None and other_team_obs is not None:
            # Would infer opponent strategy here
            belief_state = self.belief_module(
                observations=other_team_obs.unsqueeze(0) if other_team_obs.dim() == 2 else other_team_obs
            )
        else:
            # V1: No beliefs, use zeros
            belief_state = None
        
        # ============================================================
        # STAGE 3: MEAN FIELD ENCODER (Scalability)
        # Aggregate neighbor information using only reliable messages
        # Reduces O(N) neighbor processing to O(1) role-based aggregation
        # ============================================================
        mean_field = self.hmf_encoder(
            neighbor_states=neighbor_states,
            neighbor_roles=neighbor_roles,
            neighbor_mask=neighbor_mask,
            trust_weights=reliability_weights  # CRITICAL: Only reliable neighbors contribute
        )
        # Output: mean_field (batch, num_roles * state_dim = 18)
        
        # ============================================================
        # STAGE 4: WORLD MODEL (Foresight) - SKIPPED FOR V1
        # Would predict future states for planning
        # ============================================================
        # Skipped for Version 1
        
        # ============================================================
        # STAGE 5: POLICY NETWORK (Decision Making)
        # Combine all processed information and propose action
        # ============================================================
        if belief_state is not None:
            policy_input = torch.cat([local_obs, mean_field, belief_state], dim=-1)
        else:
            policy_input = torch.cat([local_obs, mean_field], dim=-1)
        
        action_logits = self.policy_net(policy_input)
        value = self.value_net(policy_input)
        
        # ============================================================
        # STAGE 6: SAFETY MODULE
        # NOTE: Safety verification is applied AFTER sampling action
        # This is done in select_action() method (see below)
        # Rationale: Policy learns to propose safe actions through
        # reward shaping, safety module provides hard guarantee
        # ============================================================
        
        # Collect intermediate outputs for logging/debugging
        info = {
            'reliability_weights': reliability_weights,  # (batch, num_neighbors)
            'belief_state': belief_state if belief_state is not None else torch.zeros(batch_size, 4, device=local_obs.device),
            'mean_field': mean_field,  # (batch, 18)
            'filtered_messages': filtered_messages,  # (batch, 8)
            'policy_input': policy_input  # (batch, 28 or 32)
        }
        
        return action_logits, value, info
    
    def select_action(self, 
                      action_logits: torch.Tensor,
                      state_dict: Dict,
                      agent_id: int,
                      deterministic: bool = False) -> Tuple[int, bool, str]:
        """
        Sample action from logits and apply safety verification.
        
        This is where STAGE 6 (Safety Module) is applied. The policy proposes
        an action, and the safety module verifies it satisfies hard constraints.
        
        Design rationale:
        - Policy learns to propose safe actions (through reward shaping)
        - Safety module is backup (zero tolerance for violations)
        - Allows exploration during training (shield prevents actual violations)
        - Provides hard guarantee during deployment
        
        Args:
            action_logits: (action_dim,) - logits for single agent
            state_dict: Environment state for safety check
            agent_id: Which agent is acting
            deterministic: If True, use argmax; else sample from distribution
        
        Returns:
            final_action: int - action to execute (0-6)
            was_blocked: bool - whether safety module blocked the action
            block_reason: str - reason for blocking (if blocked, else "")
        """
        # Sample action from policy
        if deterministic:
            proposed_action = action_logits.argmax().item()
        else:
            action_probs = torch.softmax(action_logits, dim=-1)
            proposed_action = torch.multinomial(action_probs, 1).item()
        
        # Apply safety verification (STAGE 6 of pipeline)
        final_action, violated, reason = self.safety_module.verify_action(
            proposed_action=proposed_action,
            state_dict=state_dict,
            agent_id=agent_id
        )
        
        return final_action, violated, reason
    
    def get_policy_input_dim(self) -> int:
        """
        Get the expected policy input dimension.
        Useful for debugging and architecture verification.
        """
        mean_field_dim = self.num_roles * self.state_dim
        belief_dim = 4 if self.use_beliefs else 0
        return self.obs_dim + mean_field_dim + belief_dim


# ============================================================
# INTEGRATION CHECKLIST
# ============================================================
# [✓] Reliability weights flow from Trust Gate to Mean Field
# [✓] Belief state concatenated with observations (if enabled)
# [✓] Safety module can modify actions (in select_action)
# [✓] All modules use consistent data types (torch.Tensor)
# [✓] Batch dimensions handled correctly
# [✓] state_dim (6) vs obs_dim (10) distinction maintained
# [✓] Policy input dimension computed correctly
# [✓] Value network for Actor-Critic RL
# [✓] Info dict for logging/debugging
