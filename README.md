# Cognitive Swarm Framework

**A Unified Framework for Scalable, Robust, and Safe Multi-Agent Coordination**

## Overview

The Cognitive Swarm framework implements a novel **Secure Decision Pipeline** that integrates four key capabilities into a unified decision-making architecture:

1. **Scalability**: O(N) → O(1) via Hierarchical Mean Field aggregation
2. **Robustness**: Handles 20% message corruption via Byzantine fault-tolerant Trust Gate
3. **Uncertainty**: Bayesian strategy inference for opponent modeling (optional)
4. **Safety**: Hard constraint verification with 0% violations guaranteed

## Key Innovation

The **Secure Decision Pipeline** processes information through 6 stages in a specific order:

```
Raw Observations
     ↓
[Trust Gate] ← Filter corrupted messages (20% noise tolerance)
     ↓
[Bayesian Beliefs] ← Infer opponent strategy [OPTIONAL in V1]
     ↓
[Mean Field Encoder] ← Aggregate neighbors O(N)→O(1)
     ↓
[Policy Network] ← Propose action
     ↓
[Safety Shield] ← Verify constraints (0% violations)
     ↓
Final Safe Action
```

**The ordering matters!** Any other sequence would compromise reliability or safety.

## Quick Start

### Installation

```bash
# Clone or navigate to the project
cd cognitive_swarm

# Install dependencies
pip install torch numpy
```

### Basic Usage

```python
from cognitive_swarm import CognitiveAgent

# Initialize agent
agent = CognitiveAgent(
    obs_dim=10,
    message_dim=8,
    state_dim=6,
    action_dim=7,
    num_roles=3,
    use_beliefs=False  # V1 configuration
)

# Forward pass through Secure Decision Pipeline
action_logits, value, info = agent(
    local_obs=local_obs,          # (batch, 10)
    messages=messages,            # (batch, N, 8)
    neighbor_states=neighbor_states,  # (batch, N, 6)
    neighbor_roles=neighbor_roles,    # (batch, N)
    neighbor_mask=neighbor_mask       # (batch, N)
)

# Select safe action
final_action, violated, reason = agent.select_action(
    action_logits=action_logits[0],
    state_dict=state_dict,
    agent_id=0
)
```

### Running the Demo

```bash
# View architecture (no dependencies required)
python scripts/architecture_demo.py

# Run full system demonstration (requires PyTorch)
python scripts/full_system_demo.py

# Run comprehensive tests
python tests/test_agent.py

# Train the agent
python scripts/train.py
```

## Architecture

### Module Integration

```
┌─────────────┐  reliability_weights  ┌─────────────┐
│ Trust Gate  │ ──────────────────> │ Mean Field  │
└─────────────┘                      │  Encoder    │
                                     └─────────────┘
                                            │
                                      mean_field (18)
                                            │
                                            ↓
                                     ┌─────────────┐
                                     │   Policy    │
                                     │   Network   │
                                     └─────────────┘
                                            │
                                     action_logits (7)
                                            │
                                            ↓
                                     ┌─────────────┐
                                     │   Safety    │
                                     │   Shield    │
                                     └─────────────┘
                                            │
                                     final_action
```

### Dimensional Flow (V1)

| Component | Input | Output |
|-----------|-------|--------|
| Environment | - | obs (10), messages (N×8), neighbor_states (N×6) |
| Trust Gate | messages (batch,N,8) | reliability_weights (batch,N) |
| Mean Field Encoder | neighbor_states (batch,N,6) | mean_field (batch,18) |
| Policy Network | obs (10) + mean_field (18) | action_logits (batch,7) |
| Safety Shield | proposed_action (int) | final_action (int) |

**Key**: `state_dim=6` for neighbors, `obs_dim=10` for local observations

## Project Structure

```
cognitive_swarm/
├── agents/
│   └── cognitive_agent.py      # Main integration (500+ lines)
├── modules/
│   ├── trust_gate.py           # Byzantine fault tolerance
│   ├── hmf_encoder.py          # Mean Field aggregation
│   └── bayesian_beliefs.py     # Strategy inference [OPTIONAL]
├── governance/
│   └── shield.py               # Safety constraints
└── environment/
    └── coordination_env.py     # Multi-agent environment

tests/
└── test_agent.py               # Comprehensive test suite (400+ lines)

scripts/
├── architecture_demo.py        # Architecture visualization
├── full_system_demo.py         # Full system demonstration (500+ lines)
└── train.py                    # PPO training script (400+ lines)
```

## Key Features

### 1. Byzantine Fault Tolerance
- Filters corrupted messages using Graph Attention Network
- Handles 20% Gaussian noise injection (σ=2.0)
- Consistency checking across neighbors
- Output: reliability_weights for each neighbor

### 2. Scalable Aggregation
- Hierarchical Mean Field: O(N) neighbors → O(1) role aggregates
- Groups by role: Scout, Coordinator, Support
- Uses only reliable neighbors (filtered by Trust Gate)
- Enables coordination in swarms of 100+ agents

### 3. Safe Decision Making
- Three hard constraints:
  - Protected Entity Distance: d ≥ 5 units
  - Proportionality: resource ≤ λ × target_value
  - Resource Sufficiency: resource ≥ 1
- Fallback to HOLD action if violated
- Guarantee: 0% constraint violations

### 4. Actor-Critic Architecture
- Separate policy and value networks
- Enables advantage estimation (GAE)
- Compatible with PPO, A2C, SAC algorithms

## Design Rationale

### Why This Module Ordering?

1. **Trust Gate FIRST**: Corrupted data poisons all downstream processing
2. **Beliefs SECOND** [Optional]: Provides context before aggregation
3. **Mean Field THIRD**: Uses filtered messages for scalable aggregation
4. **Safety LAST**: Final verification ensures deployment guarantee

### Why Safety After Policy?

- **Policy learns safe behavior** through reward shaping
- **Safety is backup** for hard guarantee
- **Allows exploration** during training while preventing violations
- **Combines** learning efficiency with deployment safety

## Academic Context

### Research Contribution

This work presents a unified framework that addresses:

- **Scalability**: Mean Field Theory
- **Robustness**: Byzantine Tolerance
- **Uncertainty**: Bayesian Inference
- **Safety**: Constraint Satisfaction

**Novel Contribution**: The specific integration and ordering of these techniques creates emergent properties beyond any single component.

### Applications

- Autonomous robot swarms
- Drone coordination
- Distributed cybersecurity
- Connected autonomous vehicles
- Emergency response coordination

### Related Work

- Mean Field MARL (Yang et al., 2020)
- Safe Multi-Agent RL (Sootla et al., 2022)
- Robust Adversarial RL (Pinto et al., 2017)
- Opponent Modeling (He et al., 2016)
- Byzantine Generals Problem (Lamport et al., 1982)

## Testing

The framework includes comprehensive tests:

```bash
# Run all tests
python tests/test_agent.py
```

Tests cover:
- ✓ Full pipeline integration
- ✓ Safety constraint verification
- ✓ Ablation studies (each module)
- ✓ Dimensional consistency
- ✓ Batch processing
- ✓ Gradient flow
- ✓ Mock environment integration

## Training

Simple PPO training example:

```bash
python scripts/train.py
```

The training script demonstrates:
- PPO algorithm implementation
- Actor-Critic updates
- GAE advantage estimation
- Safety constraint integration
- Statistics tracking

## Version Information

**Version 1.0** (Current):
- ✓ Trust Gate (Byzantine tolerance)
- ✓ Mean Field Encoder (Scalability)
- ✓ Safety Shield (Hard constraints)
- ✓ Policy Network (Decision making)
- ⏭ Bayesian Beliefs (Disabled for V1)
- ⏭ World Model (Not implemented)

**Version 2.0** (Future/Journal):
- Enable Bayesian Beliefs
- Add World Model for planning
- Hierarchical Mean Field enhancement
- Shared policy/value networks
- Multi-task learning

## Documentation

- `README.md` - This file (quick start)
- `ARCHITECTURE.md` - Complete architecture documentation
- `scripts/architecture_demo.py` - Interactive architecture visualization
- Code comments - Extensive inline documentation

## Integration Checklist

- [✓] Reliability weights: Trust Gate → Mean Field Encoder
- [✓] Belief state: Beliefs → Policy (if enabled)
- [✓] Safety module: Can modify actions
- [✓] Consistent data types: All torch.Tensor
- [✓] Batch dimensions: Properly handled
- [✓] state_dim vs obs_dim: Distinction maintained
- [✓] Policy input dimension: Correctly computed (28 for V1)
- [✓] Value network: For Actor-Critic RL
- [✓] Info dict: Intermediate outputs tracked

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{cognitive_swarm2026,
  title={Secure Decision Pipeline: A Unified Framework for Scalable, 
         Robust, and Safe Multi-Agent Coordination},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2026}
}
```

## License

Academic Research Use Only

## Contact

For questions or contributions, please contact [your contact information].

---

**The Cognitive Swarm Framework - Where Security Meets Scalability**

*The specific ordering and integration of modules is what makes this work novel.*
