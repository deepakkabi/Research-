# Cognitive Swarm Framework - Architecture Documentation

## Overview

The Cognitive Swarm framework implements a novel **Secure Decision Pipeline** for multi-agent coordination that addresses four key challenges:

1. **Scalability**: O(N) → O(1) via Hierarchical Mean Field aggregation
2. **Robustness**: Byzantine fault tolerance via Trust Gate (handles 20% message corruption)
3. **Uncertainty**: Bayesian strategy inference (optional in V1)
4. **Safety**: Hard constraint verification (0% violations guaranteed)

## The Secure Decision Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                     COGNITIVE AGENT                             │
│                  Secure Decision Pipeline                       │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: OBSERVE                                                 │
│ ─────────────────────────────────────────────────────────────── │
│   Raw Observations + Messages from Environment                   │
│   • local_obs: (batch, 10)       - Agent's local state          │
│   • messages: (batch, N, 8)      - Neighbor broadcasts          │
│   • neighbor_states: (batch, N, 6) - Neighbor information       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: ORIENT (Security) - TRUST GATE                         │
│ ─────────────────────────────────────────────────────────────── │
│   Filter Corrupted/Byzantine Messages                            │
│   • Graph Attention Network (GAT)                                │
│   • Consistency checking across neighbors                        │
│   • Outputs: reliability_weights (batch, N)                     │
│                                                                   │
│   HANDLES: 20% Gaussian noise injection (σ=2.0)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: ORIENT (Context) - BAYESIAN BELIEFS [V1: SKIPPED]     │
│ ─────────────────────────────────────────────────────────────── │
│   Infer Opponent Strategy from Observation Sequences             │
│   • Bayesian inference over strategy types                       │
│   • Outputs: belief_state (batch, 4)                            │
│                                                                   │
│   STATUS: Optional - disabled for Version 1                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: DECIDE (Scale) - MEAN FIELD ENCODER                   │
│ ─────────────────────────────────────────────────────────────── │
│   Aggregate Neighbor Information by Role                         │
│   • Hierarchical aggregation: Scout, Coordinator, Support        │
│   • Uses reliability_weights (only reliable neighbors)           │
│   • Outputs: mean_field (batch, 18) = 3 roles × 6 state_dim    │
│                                                                   │
│   COMPLEXITY: O(N) neighbors → O(1) role aggregates             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: ACT (Policy) - NEURAL NETWORK                         │
│ ─────────────────────────────────────────────────────────────── │
│   Propose Action from Processed State                            │
│   • Input: obs (10) + mean_field (18) + beliefs (0) = 28 dim   │
│   • Hidden layers: 128 → 128                                     │
│   • Outputs: action_logits (batch, 7)                           │
│             value (batch, 1)                                     │
│                                                                   │
│   ACTIONS: 0-6 (Discrete), including INTERVENTION=4, HOLD=6     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: ACT (Safety) - SAFETY SHIELD                          │
│ ─────────────────────────────────────────────────────────────── │
│   Verify Action Satisfies Hard Constraints                       │
│   • Protected Entity Distance: d ≥ 5 units                      │
│   • Proportionality: resource ≤ λ × target_value                │
│   • Resource Sufficiency: resource ≥ 1                          │
│                                                                   │
│   IF VIOLATED: fallback to HOLD (action 6)                      │
│   GUARANTEE: 0% constraint violations in deployment              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Final Safe Action (0-6)
```

## Module Integration Flow

```
┌─────────────┐    reliability_weights    ┌─────────────┐
│ Trust Gate  │ ────────────────────────> │ Mean Field  │
└─────────────┘                            │  Encoder    │
                                           └─────────────┘
                                                  │
                                           mean_field (18-dim)
                                                  │
┌─────────────┐                                  ↓
│  Bayesian   │                            ┌─────────────┐
│  Beliefs    │ ────────────────────────> │   Policy    │
│ [Optional]  │    belief_state (4-dim)   │   Network   │
└─────────────┘                            └─────────────┘
                                                  │
                                           action_logits
                                                  │
                                                  ↓
                                           ┌─────────────┐
                                           │   Safety    │
                                           │   Shield    │
                                           └─────────────┘
                                                  │
                                           final_action
```

## Dimensional Flow (Version 1)

```
Component                Input Dimension              Output Dimension
─────────────────────────────────────────────────────────────────────
Environment              -                            obs (10)
                                                      messages (N×8)
                                                      neighbor_states (N×6)

Trust Gate               messages (batch, N, 8)       filtered_msgs (batch, 8)
                         local_obs (batch, 10)        reliability (batch, N)

Bayesian Beliefs         other_team_obs (batch, T, 10)  belief_state (batch, 4)
[SKIPPED in V1]          

Mean Field Encoder       neighbor_states (batch, N, 6)  mean_field (batch, 18)
                         neighbor_roles (batch, N)
                         trust_weights (batch, N)

Policy Network           obs (10) + mean_field (18)   action_logits (batch, 7)
                         = policy_input (28)          value (batch, 1)

Safety Shield            proposed_action (int)        final_action (int)
                         state_dict (dict)            violated (bool)
                                                      reason (str)
```

## Key Design Decisions

### 1. Why This Specific Module Ordering?

**Trust Gate FIRST** (Stage 2):
- Corrupted data poisons all downstream processing
- Must filter unreliable messages before aggregation
- Prevents Byzantine agents from compromising the swarm

**Beliefs SECOND** (Stage 3):
- Context for aggregation - know opponent strategy before planning
- Informs how to weigh different types of information
- Optional in V1 as it's computationally expensive

**Mean Field THIRD** (Stage 4):
- Uses filtered, reliable messages only
- Scalable aggregation reduces O(N) to O(1)
- Enables coordination in large swarms (100+ agents)

**Safety LAST** (Stage 6):
- Final verification before execution
- Hard guarantee regardless of policy output
- Zero tolerance for constraint violations

Any other ordering would compromise either reliability or safety.

### 2. Why Safety Verification After Policy?

**Policy learns to propose safe actions** (through reward shaping):
- Negative rewards for violating constraints
- Positive rewards for safe, effective actions
- Policy gradually learns safe behavior patterns

**Safety module is backup** (hard guarantee):
- Zero tolerance enforcement
- Catches rare edge cases policy missed
- Critical for deployment in real systems

**Allows exploration during training**:
- Policy can explore action space freely
- Shield prevents actual violations from occurring
- Faster learning than constrained optimization

### 3. Why Separate Policy and Value Networks?

**Actor-Critic Architecture**:
- Standard in modern RL (PPO, A2C, etc.)
- Policy network (actor) proposes actions
- Value network (critic) estimates state value

**Advantage Estimation**:
- Value network enables GAE (Generalized Advantage Estimation)
- Reduces variance in policy gradients
- Faster, more stable training

**Optional Optimization** (for journal version):
- Can share early layers between policy and value
- Reduces parameters while maintaining performance
- Trade-off: memory vs. computation

### 4. Why Make Beliefs Optional?

**Computational Cost**:
- Bayesian inference is expensive
- Requires observation history (memory overhead)
- May not be needed for simple opponents

**Ablation Studies**:
- Can measure impact on performance
- Compare: with beliefs vs. without beliefs
- Justifies computational cost in paper

**Incremental Development**:
- V1: Core pipeline without beliefs
- V2/Journal: Add beliefs if needed
- Easier to debug and validate

## Integration Checklist

- [✓] **Reliability weights flow**: Trust Gate → Mean Field Encoder
- [✓] **Belief state concatenated**: Beliefs → Policy input (if enabled)
- [✓] **Safety module modifies actions**: Policy → Safety Shield → Final action
- [✓] **Consistent data types**: All modules use torch.Tensor
- [✓] **Batch dimensions handled**: All operations support batching
- [✓] **state_dim vs obs_dim distinction**: state_dim=6 for neighbors, obs_dim=10 for local
- [✓] **Policy input dimension correct**: 10 + 18 + 0 = 28 (V1 without beliefs)
- [✓] **Value network for RL**: Actor-Critic architecture
- [✓] **Info dict for logging**: Intermediate outputs tracked

## Academic Context

### Novel Contribution

The **Secure Decision Pipeline** represents a unified framework that integrates:

1. **Scalability** (Mean Field Theory): O(N) → O(1) complexity
2. **Robustness** (Byzantine Tolerance): Handles 20% message corruption
3. **Uncertainty** (Bayesian Inference): Opponent modeling (optional)
4. **Safety** (Constraint Satisfaction): 0% violations guaranteed

**Key Insight**: The specific ordering and integration of these typically separate techniques creates a cohesive decision-making architecture with emergent properties beyond any single component.

### Related Work

- **Multi-Agent Coordination**: Yang et al. (2020) - Mean Field MARL
- **Safe MARL**: Sootla et al. (2022) - Safe Multi-Agent RL
- **Robust MARL**: Pinto et al. (2017) - Robust Adversarial RL  
- **Opponent Modeling**: He et al. (2016) - Opponent Modeling in Games
- **Byzantine Tolerance**: Lamport et al. (1982) - Byzantine Generals Problem

### Research Applications

This framework is applicable to:

- **Autonomous Systems**: Robot swarms, drone coordination
- **Cybersecurity**: Distributed intrusion detection
- **Traffic Management**: Connected autonomous vehicles
- **Resource Allocation**: Cloud computing, edge networks
- **Emergency Response**: Search and rescue, disaster response

## Usage Example

```python
from cognitive_swarm import CognitiveAgent

# Initialize agent
agent = CognitiveAgent(
    obs_dim=10,
    message_dim=8,
    state_dim=6,
    action_dim=7,
    num_roles=3,
    hidden_dim=128,
    use_beliefs=False  # V1 configuration
)

# Forward pass through pipeline
action_logits, value, info = agent(
    local_obs=local_obs,        # (batch, 10)
    messages=messages,          # (batch, N, 8)
    neighbor_states=neighbor_states,  # (batch, N, 6)
    neighbor_roles=neighbor_roles,    # (batch, N)
    neighbor_mask=neighbor_mask       # (batch, N)
)

# Select safe action
final_action, violated, reason = agent.select_action(
    action_logits=action_logits[0],
    state_dict=state_dict,
    agent_id=0,
    deterministic=True
)
```

## Testing

Run the comprehensive test suite:

```bash
cd tests
python test_agent.py
```

Run the full system demonstration:

```bash
cd scripts
python full_system_demo.py
```

Run training:

```bash
cd scripts
python train.py
```

## File Structure

```
cognitive_swarm/
├── agents/
│   ├── __init__.py
│   └── cognitive_agent.py          # Main integration (THIS FILE)
├── modules/
│   ├── trust_gate.py               # Byzantine fault tolerance
│   ├── hmf_encoder.py              # Mean Field aggregation
│   └── bayesian_beliefs.py         # Strategy inference (optional)
├── governance/
│   └── shield.py                   # Safety constraints
└── environment/
    └── coordination_env.py         # Multi-agent environment

tests/
└── test_agent.py                   # Comprehensive test suite

scripts/
├── full_system_demo.py             # System demonstration
└── train.py                        # PPO training script
```

## Version Information

**Version 1.0** (Current):
- ✓ Trust Gate (Byzantine tolerance)
- ✓ Mean Field Encoder (Scalability)
- ✓ Safety Shield (Hard constraints)
- ✓ Policy Network (Decision making)
- ⏭ Bayesian Beliefs (Disabled)
- ⏭ World Model (Not implemented)

**Version 2.0** (Journal/Future):
- Enable Bayesian Beliefs
- Add World Model for planning
- Hierarchical Mean Field
- Shared policy/value networks
- Multi-task learning

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{cognitive_swarm2026,
  title={Secure Decision Pipeline: A Unified Framework for Scalable, Robust, and Safe Multi-Agent Coordination},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2026}
}
```

## License

Academic Research Use Only

---

*This is the HEART of the Cognitive Swarm framework. The specific ordering and integration of modules is what makes this work novel.*
