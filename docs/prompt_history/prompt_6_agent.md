# Prompt 6: Cognitive Agent (Integration)
PROMPT 6: COGNITIVE AGENT (INTEGRATION)
ACADEMIC RESEARCH DISCLAIMER:
This is for research on integrated multi-agent decision-making architectures.

---

You are implementing the main agent class that integrates ALL previous modules 
into a unified decision-making pipeline. This is the core of the framework and 
represents the architectural contribution.

### INTEGRATION ARCHITECTURE

The agent processes information through these stages IN ORDER:

1. **OBSERVE**: Receive raw observations + messages
2. **ORIENT** (Security): Filter unreliable messages via Trust Gate
3. **ORIENT** (Context): Update beliefs via Bayesian module [OPTIONAL]
4. **DECIDE** (Scale): Aggregate via Mean Field encoder
5. **DECIDE** (Plan): Imagine future via World Model [SKIP for V1]
6. **ACT** (Policy): Neural network proposes action
7. **ACT** (Safety): Safety module validates action

This is called the **"Secure Decision Pipeline"** - your novel integration pattern.

### DIMENSION SPECIFICATIONS

**CRITICAL**: Ensure dimensional consistency across all modules:
```python
# From environment:
state_dim = 6          # Neighbor broadcast dimension (self_state)
message_dim = 8        # Message vector dimension
obs_dim = 10           # Local observation dimension (for policy input)
action_dim = 7         # Discrete(7) - matches environment

# Computed dimensions:
num_roles = 3          # Scout, Coordinator, Support
num_strategy_types = 4  # Strategy A, B, C, D
mean_field_dim = num_roles * state_dim = 18
belief_dim = num_strategy_types = 4

# Policy input dimension:
policy_input_dim = obs_dim + mean_field_dim + belief_dim
                 = 10 + 18 + 4 = 32
```

### IMPLEMENTATION SPECIFICATION
```python
import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

from modules.trust_gate import TrustGate
from modules.hmf_encoder import HMFEncoder
from modules.bayesian_beliefs import BayesianBeliefModule
from governance.shield import SafetyConstraintModule
# from world_model.transformer_wm import TransformerWorldModel  # Skip for V1

class CognitiveAgent(nn.Module):
    """
    Main agent class integrating:
    - Trust Gate (message reliability)
    - Bayesian Beliefs (strategy uncertainty) [OPTIONAL]
    - Mean Field Encoder (scalability)
    - Safety Constraints (operational rules)
    
    This is the core architectural contribution of the research.
    
    Research application: Scalable, robust, safe multi-agent coordination.
    """
    
    def __init__(self, 
                 obs_dim: int = 10,
                 message_dim: int = 8,
                 state_dim: int = 6,
                 action_dim: int = 7,
                 num_roles: int = 3,
                 num_strategy_types: int = 4,
                 hidden_dim: int = 128,
                 use_beliefs: bool = False,  # Can disable for V1
                 use_world_model: bool = False):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.use_beliefs = use_beliefs
        self.use_world_model = use_world_model
        
        # === MODULE 1: Trust Gate (Message Reliability) ===
        self.trust_gate = TrustGate(
            message_dim=message_dim,
            hidden_dim=64,
            num_heads=4,
            consistency_threshold=0.5
        )
        
        # === MODULE 2: Bayesian Beliefs (Strategy Inference) - OPTIONAL ===
        if use_beliefs:
            self.belief_module = BayesianBeliefModule(
                obs_dim=obs_dim,
                num_types=num_strategy_types,
                hidden_dim=64
            )
            belief_contrib = num_strategy_types
        else:
            self.belief_module = None
            belief_contrib = 0
        
        # === MODULE 3: Mean Field Encoder (Scalability) ===
        self.hmf_encoder = HMFEncoder(
            state_dim=state_dim,  # 6, not obs_dim!
            num_roles=num_roles,
            use_hierarchical=False  # Set True for journal version
        )
        
        # === MODULE 4: World Model (Foresight) - SKIP FOR V1 ===
        if use_world_model:
            raise NotImplementedError("World model skipped for Version 1")
        
        # === MODULE 5: Policy Network (Decision Making) ===
        # Input: local_obs + mean_field + belief_state (if used)
        policy_input_dim = (
            obs_dim +                    # 10: Local observation
            num_roles * state_dim +      # 18: Mean field (3 roles * 6 state_dim)
            belief_contrib               # 4 if beliefs used, else 0
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # === MODULE 6: Safety Constraints ===
        self.safety_module = SafetyConstraintModule(
            safe_distance=5,
            proportionality_lambda=1.0
        )
        
        # === Value Network (for RL) ===
        self.value_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
                local_obs: torch.Tensor,           # (batch, obs_dim=10)
                messages: torch.Tensor,            # (batch, num_neighbors, msg_dim=8)
                neighbor_states: torch.Tensor,     # (batch, num_neighbors, state_dim=6)
                neighbor_roles: torch.Tensor,      # (batch, num_neighbors)
                neighbor_mask: torch.Tensor,       # (batch, num_neighbors)
                edge_index: torch.Tensor,          # (2, num_edges)
                neighbor_ids: torch.Tensor,        # (batch, num_neighbors)
                other_team_obs: torch.Tensor = None,  # (batch, seq_len, obs_dim) - optional
                state_dict: Dict = None            # For safety module
               ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Execute the Secure Decision Pipeline.
        
        Returns:
            action_logits: (batch, action_dim=7) - raw policy output
            value: (batch, 1) - state value estimate
            info: dict with intermediate outputs for logging/analysis
        """
        
        # ============ STAGE 1: TRUST GATE (Message Reliability) ============
        # Filter erroneous/corrupted messages
        filtered_messages, reliability_weights = self.trust_gate(
            messages=messages,
            local_obs=local_obs,
            edge_index=edge_index,
            neighbor_ids=neighbor_ids
        )
        # Output: filtered_messages (batch, msg_dim=8)
        #         reliability_weights (batch, num_neighbors)
        
        # ============ STAGE 2: BAYESIAN BELIEFS (Strategy Uncertainty) ============
        if self.use_beliefs and other_team_obs is not None:
            # Infer other team's strategy
            belief_state = self.belief_module(
                observations=other_team_obs.unsqueeze(0) if other_team_obs.dim() == 2 else other_team_obs
            )
            # Output: belief_state (batch, num_strategy_types=4)
        else:
            belief_state = None
        
        # ============ STAGE 3: MEAN FIELD ENCODER (Scalability) ============
        # Aggregate neighbor information using only reliable messages
        mean_field = self.hmf_encoder(
            neighbor_states=neighbor_states,
            neighbor_roles=neighbor_roles,
            neighbor_mask=neighbor_mask,
            trust_weights=reliability_weights  # Only reliable neighbors contribute
        )
        # Output: mean_field (batch, num_roles * state_dim = 18)
        
        # ============ STAGE 4: WORLD MODEL (Foresight) - SKIPPED ============
        # Skipped for Version 1
        
        # ============ STAGE 5: POLICY NETWORK (Decide) ============
        if belief_state is not None:
            policy_input = torch.cat([local_obs, mean_field, belief_state], dim=-1)
        else:
            policy_input = torch.cat([local_obs, mean_field], dim=-1)
        
        action_logits = self.policy_net(policy_input)
        value = self.value_net(policy_input)
        
        # ============ STAGE 6: SAFETY MODULE ============
        # NOTE: Safety verification is applied AFTER sampling action, not on logits
        # This is done in select_action method (see below)
        
        # Collect info for logging/debugging
        info = {
            'reliability_weights': reliability_weights,
            'belief_state': belief_state if belief_state is not None else torch.zeros(local_obs.size(0), 4),
            'mean_field': mean_field,
            'filtered_messages': filtered_messages
        }
        
        return action_logits, value, info
    
    def select_action(self, 
                      action_logits: torch.Tensor,
                      state_dict: Dict,
                      agent_id: int,
                      deterministic: bool = False) -> Tuple[int, bool, str]:
        """
        Sample action from logits and apply safety verification.
        
        Args:
            action_logits: (action_dim,) - logits for single agent
            state_dict: Environment state for safety check
            agent_id: Which agent is acting
            deterministic: If True, use argmax; else sample
        
        Returns:
            final_action: int - action to execute (0-6)
            was_blocked: bool - whether safety module blocked the action
            block_reason: str - reason for blocking (if blocked)
        """
        # Sample action from policy
        if deterministic:
            proposed_action = action_logits.argmax().item()
        else:
            action_probs = torch.softmax(action_logits, dim=-1)
            proposed_action = torch.multinomial(action_probs, 1).item()
        
        # Apply safety verification
        final_action, violated, reason = self.safety_module.verify_action(
            proposed_action=proposed_action,
            state_dict=state_dict,
            agent_id=agent_id
        )
        
        return final_action, violated, reason
```

### DESIGN DECISIONS TO EXPLAIN

1. **Why this specific ordering of modules?**
   - Trust Gate FIRST (reliability before everything - corrupted data poisons all downstream)
   - Beliefs SECOND (context for aggregation - know opponent strategy before planning)
   - Mean Field THIRD (uses reliable messages - scalable aggregation)
   - Safety LAST (final check before execution - hard guarantee)
   - Any other order would compromise reliability or safety

2. **Why safety verification after policy, not during policy?**
   - Policy learns to propose safe actions (through reward shaping)
   - Safety module is backup (zero tolerance for constraint violations)
   - Allows policy to explore during training (shield prevents actual violations)
   - Safety module provides hard guarantee during deployment

3. **Why separate policy and value networks?**
   - Actor-Critic architecture (standard in RL)
   - Value network for advantage estimation
   - Can share early layers (optional optimization for journal version)

4. **Why make beliefs optional?**
   - Computationally expensive
   - Not strictly necessary for proof-of-concept
   - Can add later for journal version
   - Allows ablation: performance with/without opponent modeling

### TESTING REQUIREMENTS
```python
import numpy as np

def test_full_pipeline():
    """Verify all modules connect correctly"""
    agent = CognitiveAgent(
        obs_dim=10,
        message_dim=8,
        state_dim=6,
        action_dim=7,
        num_roles=3,
        num_strategy_types=4,
        use_beliefs=True  # Test with beliefs enabled
    )
    
    # Create dummy inputs
    batch_size = 4
    num_neighbors = 5
    
    local_obs = torch.randn(batch_size, 10)
    messages = torch.randn(batch_size, num_neighbors, 8)
    neighbor_states = torch.randn(batch_size, num_neighbors, 6)  # state_dim=6!
    neighbor_roles = torch.randint(0, 3, (batch_size, num_neighbors))
    neighbor_mask = torch.ones(batch_size, num_neighbors)
    edge_index = torch.randint(0, num_neighbors, (2, 20))
    neighbor_ids = torch.arange(num_neighbors).unsqueeze(0).expand(batch_size, -1)
    other_team_obs = torch.randn(batch_size, 5, 10)  # 5 time steps
    
    # Forward pass
    action_logits, value, info = agent(
        local_obs, messages, neighbor_states, neighbor_roles,
        neighbor_mask, edge_index, neighbor_ids, other_team_obs
    )
    
    # Check shapes
    assert action_logits.shape == (batch_size, 7)
    assert value.shape == (batch_size, 1)
    assert 'reliability_weights' in info
    assert 'belief_state' in info
    assert 'mean_field' in info
    assert info['mean_field'].shape == (batch_size, 18)  # 3 roles * 6 state_dim
    print("✓ Full pipeline test passed")

def test_safety_integration():
    """Verify safety module blocks unsafe actions"""
    agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, use_beliefs=False)
    
    # Create scenario where INTERVENTION would violate constraints
    state_dict = {
        'agent_positions': np.array([[0, 0]]),
        'protected_positions': np.array([[3, 0]]),  # Too close (< 5)
        'target_positions': np.array([[10, 10]]),
        'agent_resources': np.array([10]),
        'target_values': np.array([5])
    }
    
    # Policy proposes INTERVENTION (action 4)
    action_logits = torch.zeros(1, 7)
    action_logits[0, 4] = 10.0  # Strongly prefer INTERVENTION
    
    final_action, blocked, reason = agent.select_action(
        action_logits[0], state_dict, agent_id=0
    )
    
    assert blocked == True
    assert final_action == 6  # Should be changed to HOLD
    assert reason == "Protected Entity Constraint"
    print("✓ Safety integration test passed")

def test_ablation_components():
    """Test each component can be disabled for ablation studies"""
    
    # Test 1: Disable beliefs
    agent_no_beliefs = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, 
                                      use_beliefs=False)
    
    inputs = {
        'local_obs': torch.randn(1, 10),
        'messages': torch.randn(1, 5, 8),
        'neighbor_states': torch.randn(1, 5, 6),
        'neighbor_roles': torch.randint(0, 3, (1, 5)),
        'neighbor_mask': torch.ones(1, 5),
        'edge_index': torch.randint(0, 5, (2, 10)),
        'neighbor_ids': torch.arange(5).unsqueeze(0),
        'other_team_obs': None  # No beliefs, so this is None
    }
    
    out1 = agent_no_beliefs(**inputs)
    assert out1[0].shape == (1, 7)
    print("✓ Beliefs disabled successfully")
    
    # Test 2: Disable Trust Gate (set threshold very low)
    agent_no_beliefs.trust_gate.threshold = -1.0  # Accept everything
    out2 = agent_no_beliefs(**inputs)
    print("✓ Trust Gate disabled successfully")
    
    # Test 3: Disable Safety Module
    agent_no_beliefs.safety_module.safe_distance = 0  # No restrictions
    print("✓ Safety module disabled successfully")
    
    print("✓ All ablation tests passed")

def test_dimensional_consistency():
    """Verify all dimensions match across modules"""
    agent = CognitiveAgent(
        obs_dim=10,
        message_dim=8,
        state_dim=6,
        action_dim=7,
        num_roles=3,
        use_beliefs=True
    )
    
    # Verify internal dimensions
    assert agent.hmf_encoder.state_dim == 6
    assert agent.hmf_encoder.output_dim == 18  # 3 * 6
    assert agent.trust_gate.message_dim == 8
    
    # Verify policy input dimension
    expected_policy_input = 10 + 18 + 4  # obs + mean_field + beliefs = 32
    actual_policy_input = agent.policy_net[0].in_features
    assert actual_policy_input == expected_policy_input
    
    print("✓ Dimensional consistency test passed")
```

### OUTPUT FORMAT

Provide:
1. Complete `cognitive_agent.py`
2. Test file `test_agent.py`
3. Architecture diagram (ASCII):
Raw Obs + Messages ↓ [Trust Gate] ← Message Reliability (filters 20% noise) ↓ Filtered Messages (reliability_weights) ↓ [Bayesian Beliefs] ← Strategy Inference (OPTIONAL) ↓ Belief State (4-dim) ↓ [Mean Field Encoder] ← Scalability (18-dim output) ↓ Compact Representation ↓ [Policy Network] ← Decision Making (32-dim input → 7 actions) ↓ Proposed Action ↓ [Safety Module] ← Constraint Verification (blocks if unsafe) ↓ Final Safe Action

4. Integration checklist:
   - [✓] Reliability weights flow from Trust Gate to Mean Field
   - [✓] Belief state concatenated with observations (if enabled)
   - [✓] Safety module can modify actions
   - [✓] All modules use consistent data types (torch.Tensor)
   - [✓] Batch dimensions handled correctly
   - [✓] state_dim vs obs_dim distinction maintained

### CRITICAL NOTE

This file is the HEART of your contribution. The specific ordering and 
integration of modules is what makes this work novel. In your paper:

1. **Methods section**: Describe this architecture in detail with the flow diagram
2. **Results section**: Show ablation studies for each component
3. **Discussion**: Explain why this ordering matters (security before aggregation, etc.)

The "Secure Decision Pipeline" is your framework's identity.

### ACADEMIC CONTEXT

This work presents a unified framework for multi-agent coordination that addresses:
- Scalability (Mean Field Theory → O(N) to O(1))
- Robustness (Reliability verification → handles 20% message corruption)
- Uncertainty (Bayesian inference → opponent modeling)
- Safety (Constraint satisfaction → 0% violations)

The key contribution is the INTEGRATION of these typically separate techniques
into a cohesive decision-making architecture with a specific processing order.

Related work: Multi-agent coordination surveys, Safe MARL, Robust MARL,
Opponent Modeling in competitive settings, Mean Field MARL

