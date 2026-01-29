# Multi-Agent Coordination Environment

**Academic Research Environment for Studying Coordination Under Adversarial Conditions**

This is a PettingZoo-based simulation environment designed for research on scalable multi-agent reinforcement learning (MARL) with emphasis on:
- **Scalability**: 100-1000 agents
- **Robustness**: Communication failures and noise
- **Safety**: Constraint enforcement around protected entities
- **Adversarial conditions**: Unreliable communication channels

---

## ⚠️ Research Disclaimer

This is an **ABSTRACT research environment** for academic purposes only. It is designed to advance understanding of coordination, safety, and robustness in multi-agent systems. **NOT intended for real-world deployment.**

Inspired by research in cooperative MARL including QMIX, MADDPG, and fault-tolerant distributed systems.

---

## Installation

```bash
# Install dependencies
pip install numpy torch gymnasium pettingzoo

# Verify installation
python test_env.py
```

---

## Quick Start

### Basic Usage

```python
from coordination_env import CoordinationEnv

# Create environment with 100 agents
env = CoordinationEnv(
    num_agents=100,
    num_adversaries=30,
    num_protected=10,
    seed=42
)

# Reset environment
observations = env.reset(seed=42)

# Run episode
for step in range(500):
    # Random actions for demonstration
    actions = {
        agent: env.action_space.sample()
        for agent in env.agents
    }
    
    # Step environment
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Check if episode ended
    if any(terminations.values()):
        break

print(f"Episode completed in {step} steps")
```

### Integration with Neural Networks

```python
from coordination_env import CoordinationEnv
import torch

env = CoordinationEnv(num_agents=50, seed=42)
obs = env.reset(seed=42)

# Convert observations to batched tensors
obs_list = list(obs.values())
batched_obs = env.collate_observations(obs_list, max_neighbors=20)

# Now batched_obs contains PyTorch tensors ready for neural networks:
# - batched_obs['local_obs']: (batch_size, 6)
# - batched_obs['messages']: (batch_size, 20, 8)
# - batched_obs['neighbor_mask']: (batch_size, 20)
# - batched_obs['neighbor_states']: (batch_size, 20, 6)

# Feed to neural network
# output = your_model(batched_obs)
```

---

## Environment Specification

### Action Space

7 discrete actions (integer indices 0-6):

| Index | Action | Description |
|-------|--------|-------------|
| 0 | MOVE_NORTH | Move one cell north |
| 1 | MOVE_SOUTH | Move one cell south |
| 2 | MOVE_EAST | Move one cell east |
| 3 | MOVE_WEST | Move one cell west |
| 4 | INTERVENTION | High-impact action (neutralize adversary within range 2) |
| 5 | COMMUNICATE | Broadcast message to neighbors within comm_radius |
| 6 | HOLD | No action |

**Critical**: These indices must match across all modules (Shield, Agent, Training Loop)

### Observation Space

Each agent receives a dictionary with:

```python
{
    'local_grid': np.array(5, 5, 4),  # Local 5x5 grid, 4 channels
    'self_state': np.array(6,),        # [x, y, health, resource, role_id, team_id]
    'messages': List[np.array(8,)],    # Variable-length list of messages
    'neighbor_ids': np.array,          # IDs of agents within comm_radius
}
```

**Channels in local_grid**:
- Channel 0: Teammates (1.0 if present, 0.0 otherwise)
- Channel 1: Adversaries
- Channel 2: Protected entities
- Channel 3: Obstacles

**Message format** (8-dimensional vector):
```python
[sender_id, role, x, y, status, target_x, target_y, priority]
```

### Agent Roles

Three role types with specific distribution:
- **Scout (0)**: 50% of agents - reconnaissance
- **Coordinator (1)**: 30% of agents - team coordination
- **Support (2)**: 20% of agents - resource management

### Reward Structure

```python
reward = (
    +10 * adversaries_neutralized
    -5  * teammates_lost
    -20 * protected_entities_harmed    # Critical for safety testing
    +1  * mission_progress
)
```

### Episode Termination

Episodes end when:
1. Maximum steps reached (default: 500), OR
2. All adversaries neutralized, OR
3. More than 50% of agents disabled

---

## Communication System

### Communication Model

- **Radius**: Default 7 cells (configurable)
- **Message format**: 8-dimensional vectors
- **Broadcasting**: Agents using COMMUNICATE action broadcast to all neighbors

### Noise Injection (Critical Feature)

**Purpose**: Simulate unreliable communication or adversarial interference

**Mechanism**: The `inject_noise()` method:
- Randomly selects 20% of agents each step
- Replaces their messages with Gaussian noise: N(0, σ=2.0)
- **Critical**: Recipients do NOT know their channel is corrupted

**Design Rationale**:

1. **Why 20% noise rate?**
   - Based on fault tolerance research in distributed systems
   - Represents realistic communication failure rates
   - Challenging enough to require robust filtering, but not overwhelming

2. **Why Gaussian noise with σ=2.0?**
   - Real messages have magnitude in range [0, 10]
   - σ=2.0 produces noise in ~[-4, +4] (95% confidence interval)
   - **Signal-to-noise ratio ≈ 2.5:1**
   - Makes detection challenging but feasible
   - Calibrated with Trust Gate threshold=0.5

3. **Alternative noise models** (for future research):
   - Targeted corruption (specific agents)
   - Burst corruption (temporal patterns)
   - Byzantine failures (adversarial messages)

**Usage**:

```python
# Automatic noise injection during step()
obs, rewards, terms, truncs, infos = env.step(actions)

# Check if agent had noisy channel
if infos[agent]['noisy_channel']:
    print(f"{agent} had corrupted communication this step")
```

---

## Key Methods

### `reset(seed=None)`

Initialize episode with fresh state.

**Returns**: Dictionary of initial observations for all agents

### `step(actions)`

Execute one environment step.

**Args**: `actions` - Dictionary mapping agent_id → action_int

**Returns**: `(observations, rewards, terminations, truncations, infos)`

### `inject_noise(messages_dict, noise_probability=0.2)`

Corrupt communication channels with Gaussian noise.

**Args**:
- `messages_dict`: Dictionary of agent messages
- `noise_probability`: Fraction of channels to corrupt (default: 0.2)

**Returns**: Dictionary with corrupted messages

### `get_protected_distances(agent_id)`

Calculate distances from agent to all protected entities.

**Returns**: Array of distances, shape `(num_protected_entities,)`

**Use case**: Safety constraint testing

### `collate_observations(obs_list, max_neighbors=20)`

Convert variable-length observations to fixed-size batched tensors.

**Args**:
- `obs_list`: List of observation dictionaries
- `max_neighbors`: Maximum neighbors to consider (pads/truncates)

**Returns**: Dictionary of PyTorch tensors ready for neural networks

**Output tensors**:
- `local_obs`: (batch, 6) - Self state vectors
- `messages`: (batch, max_neighbors, 8) - Padded messages
- `neighbor_mask`: (batch, max_neighbors) - Valid message indicators
- `neighbor_ids`: (batch, max_neighbors) - Neighbor IDs
- `neighbor_states`: (batch, max_neighbors, 6) - Extracted neighbor states
- `neighbor_roles`: (batch, max_neighbors) - Extracted neighbor roles

---

## Parameter Tuning Guidelines

### Scalability Testing

```python
# Small-scale testing (fast iteration)
env = CoordinationEnv(num_agents=10, num_adversaries=5)

# Medium-scale (typical research)
env = CoordinationEnv(num_agents=100, num_adversaries=30)

# Large-scale (scalability research)
env = CoordinationEnv(num_agents=1000, num_adversaries=100, grid_size=100)
```

### Communication Parameters

```python
# High communication reliability (5% noise)
env = CoordinationEnv(noise_probability=0.05)

# Moderate noise (default, 20%)
env = CoordinationEnv(noise_probability=0.2)

# High noise / adversarial conditions (40% noise)
env = CoordinationEnv(noise_probability=0.4)

# Extended communication range
env = CoordinationEnv(comm_radius=10)

# Limited communication (local only)
env = CoordinationEnv(comm_radius=3)
```

### Difficulty Adjustment

```python
# Easy: Few adversaries, many agents
env = CoordinationEnv(num_agents=100, num_adversaries=10)

# Hard: Many adversaries, safety critical
env = CoordinationEnv(num_agents=50, num_adversaries=50, num_protected=20)

# Extended episodes
env = CoordinationEnv(max_steps=1000)
```

---

## Integration with Neural Network Modules

### Trust Gate Integration

The environment is designed to work with a Trust Gate module for message filtering:

```python
from coordination_env import CoordinationEnv
# from trust_gate import TrustGate  # Your module

env = CoordinationEnv(num_agents=100, seed=42)
# trust_gate = TrustGate(message_dim=8, hidden_dim=64, threshold=0.5)

obs = env.reset(seed=42)
obs_list = list(obs.values())
batched = env.collate_observations(obs_list, max_neighbors=20)

# Trust gate can now process batched messages
# trust_scores = trust_gate(batched['messages'], batched['neighbor_mask'])
# filtered_messages = batched['messages'] * (trust_scores > 0.5)
```

### Mean Field Integration

For mean field approximations:

```python
batched = env.collate_observations(obs_list, max_neighbors=20)

# Neighbor states and roles are pre-extracted
neighbor_states = batched['neighbor_states']  # (batch, max_neighbors, 6)
neighbor_roles = batched['neighbor_roles']    # (batch, max_neighbors)

# Compute mean field for each role
# mean_field_by_role = compute_mean_field(neighbor_states, neighbor_roles)
```

### Policy Network Integration

```python
from coordination_env import CoordinationEnv
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=128, num_actions=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, state):
        return self.network(state)

env = CoordinationEnv(num_agents=50)
policy = PolicyNetwork()

obs = env.reset()
batched = env.collate_observations(list(obs.values()))

# Forward pass
logits = policy(batched['local_obs'])
actions = torch.argmax(logits, dim=1)
```

---

## Testing

Run comprehensive test suite:

```bash
python test_env.py
```

**Test coverage**:
- ✓ Action space (7 actions, correct indices)
- ✓ Observation consistency (obstacles block view)
- ✓ Noise injection rate (~20% ± 5%)
- ✓ Noise statistics (Gaussian N(0, 2.0))
- ✓ Protected entity distances
- ✓ Observation batching (correct tensor shapes)
- ✓ Communication system
- ✓ Safety constraints
- ✓ Full episode integration

---

## Research Applications

### 1. Communication Robustness

Study how agents learn to filter unreliable messages:

```python
env = CoordinationEnv(noise_probability=0.3)
# Train agents with trust gate to identify corrupted messages
```

### 2. Scalability Studies

Test coordination algorithms at different scales:

```python
for n in [50, 100, 200, 500, 1000]:
    env = CoordinationEnv(num_agents=n)
    # Measure coordination performance vs. scale
```

### 3. Safety Constraint Learning

Research safe policies that avoid harming protected entities:

```python
env = CoordinationEnv(num_protected=20)

# In training loop, penalize violations
for step in range(max_steps):
    actions = policy.select_actions(obs)
    
    # Check safety constraints
    for agent in env.agents:
        distances = env.get_protected_distances(agent)
        if distances.min() < safety_threshold and actions[agent] == env.INTERVENTION:
            # Apply safety constraint
            actions[agent] = env.HOLD
```

### 4. Adversarial Robustness

Test performance under varying adversarial pressure:

```python
for adv_ratio in [0.1, 0.3, 0.5, 0.7]:
    num_adv = int(adv_ratio * num_agents)
    env = CoordinationEnv(num_agents=100, num_adversaries=num_adv)
```

---

## Performance Considerations

### Memory Efficiency

For large-scale experiments (500+ agents):

```python
# Use smaller observation radius to reduce memory
env = CoordinationEnv(num_agents=1000, obs_radius=3)

# Limit max_neighbors in batching
batched = env.collate_observations(obs_list, max_neighbors=10)
```

### Computational Efficiency

- Grid operations use NumPy for vectorization
- Avoid Python loops in main computation paths
- Message processing is O(n × k) where k = avg neighbors

### Parallelization

Environment supports multiple parallel instances:

```python
envs = [CoordinationEnv(num_agents=100, seed=i) for i in range(4)]
# Use with vectorized environments or multi-processing
```

---

## Key Design Decisions

### 1. Discrete vs. Continuous Actions

**Choice**: Discrete (7 actions)

**Rationale**: 
- Simpler for initial research
- Easier to analyze learned policies
- Sufficient for coordination research
- Future work can extend to continuous

### 2. Local vs. Global Observations

**Choice**: Local (5×5 grid, radius=5)

**Rationale**:
- Tests partial observability
- More realistic for large-scale systems
- Forces agents to communicate
- Scalable to 1000+ agents

### 3. Communication Noise Model

**Choice**: Random 20% Gaussian noise

**Alternatives considered**:
- **Targeted corruption**: Specific agents (would require adversarial module)
- **Burst corruption**: Temporal patterns (would add complexity)
- **Byzantine failures**: Malicious messages (future extension)

**Trade-offs**:
- Current model is simple but effective
- 20% rate is challenging but learnable
- Gaussian noise is detectable with proper filtering

### 4. Fixed vs. Dynamic Grid

**Choice**: Fixed 50×50 grid

**Rationale**:
- Simplifies implementation
- Sufficient for research goals
- Can be made dynamic in future versions

---

## Troubleshooting

### Common Issues

**1. "Action space mismatch"**
```python
# Ensure actions are integers 0-6
assert all(0 <= a <= 6 for a in actions.values())
```

**2. "Dimension mismatch in batching"**
```python
# Check that all observations have consistent structure
for obs in obs_list:
    assert obs['self_state'].shape == (6,)
```

**3. "Out of memory with large-scale environments"**
```python
# Reduce max_neighbors or batch size
batched = env.collate_observations(obs_list[:100], max_neighbors=10)
```

---

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{marl_coordination_env,
  title={Multi-Agent Coordination Environment with Communication Noise},
  author={Research Implementation},
  year={2024},
  note={Inspired by QMIX, MADDPG, and distributed systems fault tolerance research}
}
```

---

## Future Extensions

Potential research directions:

1. **Dynamic adversaries**: Learning adversarial behaviors
2. **Hierarchical coordination**: Multi-level team structures
3. **Heterogeneous agents**: Different capabilities per role
4. **Communication protocols**: Learnable message formats
5. **Byzantine agents**: Adversarial teammates
6. **Continuous actions**: Smooth movement and interventions
7. **3D environments**: Extended spatial reasoning

---

## License

Academic Research Only - Not for commercial or real-world deployment

---

## Contact

For questions about this environment or research collaboration:
- Review the test suite: `test_env.py`
- Check the implementation: `coordination_env.py`
- Examine edge cases in the test functions

---

**Last Updated**: 2026
**Version**: 1.0
**Status**: Research Prototype
