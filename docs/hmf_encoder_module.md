# Hierarchical Mean Field (HMF) Encoder

## Overview

The HMF Encoder is a neural network module for the Cognitive Swarm framework that enables scalable multi-agent coordination through Mean Field Theory. It compresses neighborhood observations from O(N) to O(1) by aggregating neighbors into role-based mean fields.

## Research Context

**Problem**: Standard MARL requires each agent to process observations from all N neighbors individually, leading to O(N²) complexity in communication and O(N) in observation space.

**Solution**: Mean Field Theory approximates the collective neighbor distribution as role-based density functions, reducing complexity to O(1) per agent.

**Novel Contribution**: Extension of Mean Field MARL to heterogeneous agent teams with multiple roles (not just homogeneous swarms).

**Academic Foundation**: Based on "Mean Field Multi-Agent RL" (Yang et al., 2018) with extensions for heterogeneous multi-agent systems.

## Installation

```bash
# The module is part of the cognitive_swarm package
# Ensure PyTorch is installed
pip install torch

# Import from the modules package
from cognitive_swarm.modules import HMFEncoder, LearnedHMFEncoder
```

## Quick Start

```python
import torch
from cognitive_swarm.modules import HMFEncoder

# Initialize encoder
encoder = HMFEncoder(state_dim=6, num_roles=3)

# Prepare neighbor data (from environment)
batch_size = 32  # 32 agents
max_neighbors = 50  # Up to 50 neighbors per agent
neighbor_states = torch.randn(batch_size, max_neighbors, 6)  # State vectors
neighbor_roles = torch.randint(0, 3, (batch_size, max_neighbors))  # Role IDs
neighbor_mask = torch.ones(batch_size, max_neighbors)  # Valid neighbor mask

# Optional: Trust weights from Trust Gate module
trust_weights = torch.rand(batch_size, max_neighbors)  # [0, 1]

# Encode neighbors into mean fields
mean_fields = encoder(neighbor_states, neighbor_roles, neighbor_mask, trust_weights)

# Output shape: (32, 18) = 32 agents × 18-dim mean field (3 roles × 6 state_dim)
print(mean_fields.shape)  # torch.Size([32, 18])
```

## Mathematical Specification

### Standard Multi-Agent Observation
```
obs_i = [neighbor_1_state, neighbor_2_state, ..., neighbor_K_state]
Dimension: K × state_dim (variable length, depends on K)
```

### Mean Field Observation
```
obs_i = [μ_scout, μ_coordinator, μ_support]
Dimension: num_roles × state_dim (fixed length = 18)

Where μ_role = (1/|N_i^role|) Σ_{j ∈ N_i^role} w_j * state_j
```

- **μ_scout**: Mean field for Scout agents (role_id=0)
- **μ_coordinator**: Mean field for Coordinator agents (role_id=1)  
- **μ_support**: Mean field for Support agents (role_id=2)

Each mean field is a weighted average of all neighbors of that role, producing a 6-dimensional vector matching the environment's `self_state` format:
```
self_state = [x, y, health, resource, role_id, team_id]
```

## API Reference

### `HMFEncoder`

Main encoder using simple weighted averaging.

**Parameters:**
- `state_dim` (int, default=6): Dimension of neighbor state vectors
- `num_roles` (int, default=3): Number of agent role types
- `use_hierarchical` (bool, default=False): Enable distance-based grouping

**Input (forward method):**
- `neighbor_states`: (batch, max_neighbors, state_dim) - Neighbor state vectors
- `neighbor_roles`: (batch, max_neighbors) - Role ID for each neighbor (0/1/2)
- `neighbor_mask`: (batch, max_neighbors) - 1 if valid, 0 if padding
- `trust_weights`: (batch, max_neighbors) - Optional trust scores [0, 1]

**Output:**
- `mean_field`: (batch, num_roles × state_dim) - Fixed-size encoding (default: 18)

### `LearnedHMFEncoder`

Advanced encoder with learnable attention weights.

**Parameters:**
- `state_dim` (int, default=6): Dimension of neighbor state vectors
- `num_roles` (int, default=3): Number of agent role types
- `hidden_dim` (int, default=32): Hidden dimension for attention MLPs

Uses learned attention mechanism instead of simple averaging:
```
α_j = softmax(MLP(state_j))
μ_role = Σ(α_j * state_j)
```

## Integration with Environment

The encoder is designed to work seamlessly with the `CoordinationEnv` from the environment module:

```python
from cognitive_swarm.environment import CoordinationEnv
from cognitive_swarm.modules import HMFEncoder

# Initialize environment and encoder
env = CoordinationEnv(num_agents=100, max_neighbors=50)
encoder = HMFEncoder(state_dim=6, num_roles=3)

# During training loop
obs = env.reset()
neighbor_data = env.collate_observations(obs)

# Encode mean fields
mean_fields = encoder(
    neighbor_data['neighbor_states'],
    neighbor_data['neighbor_roles'],
    neighbor_data['neighbor_mask']
)

# Use mean_fields as input to policy network
# policy_input = torch.cat([local_obs, mean_fields], dim=-1)
```

## Edge Cases Handled

### 1. No Neighbors At All
```python
# All neighbors masked (agent is isolated)
mask = torch.zeros(1, 10)
out = encoder(states, roles, mask)
# Returns: near-zero vector (small values due to numerical stability)
```

### 2. Missing Role Groups
```python
# No Support agents nearby (only Scouts and Coordinators)
roles = torch.tensor([[0, 0, 1, 1, 0]])  # No role 2
out = encoder(states, roles, mask)
# out[:, 12:18] will be near-zero (Support mean field)
```

### 3. Untrusted Neighbors
```python
# Some neighbors have zero trust (from Trust Gate)
trust = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
out = encoder(states, roles, mask, trust)
# Neighbors with trust=0 are excluded from mean field
```

### 4. Variable Neighborhood Sizes
```python
# Different agents have different numbers of neighbors
mask = torch.rand(32, 50) > 0.5  # Random valid neighbors
out = encoder(states, roles, mask)
# Always returns (32, 18) regardless of how many neighbors each agent has
```

## Aggregation Method Comparison

| Method | Complexity | Parameters | Pros | Cons |
|--------|-----------|-----------|------|------|
| **Simple Averaging** (HMFEncoder) | O(N) | 0 | Fast, interpretable, no overfitting risk, proven in MF theory | Loses individual neighbor importance, uniform weighting |
| **Learned Aggregation** (LearnedHMFEncoder) | O(N) | ~200 (3 MLPs) | Can learn importance weights, more expressive, adapts to data | Risk of overfitting, requires more training data |
| **Attention-based** | O(N²) | Large | Most expressive, models pairwise interactions | Defeats scalability purpose, same complexity as standard MARL |

**Recommendation**: Start with `HMFEncoder` for efficiency and interpretability. Use `LearnedHMFEncoder` if you have abundant training data and need more expressive aggregation.

## Hierarchical Extension

Enable distance-based grouping for near/far field separation:

```python
encoder = HMFEncoder(state_dim=6, num_roles=3, use_hierarchical=True)

# Compute distances to neighbors (Euclidean)
distances = torch.norm(neighbor_positions - ego_position, dim=-1)

# Encode with distance grouping
mean_fields = encoder.forward_hierarchical(
    neighbor_states, neighbor_roles, neighbor_mask, distances
)

# Output: (batch, 36) = 3 roles × 2 distances × 6 state_dim
# [μ_scout_near, μ_scout_far, μ_coord_near, μ_coord_far, μ_support_near, μ_support_far]
```

Distance grouping:
- **Near field**: neighbors within 3 cells (high influence on immediate coordination)
- **Far field**: neighbors 3-7 cells away (strategic positioning awareness)

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python tests/test_hmf.py

# Or use pytest if available
pytest tests/test_hmf.py -v
```

Tests include:
- ✓ Dimensionality: O(N) → O(1) compression
- ✓ Empty role groups: Zero vectors for missing roles
- ✓ Trust weighting: Integration with Trust Gate
- ✓ Batch processing: Multiple agents simultaneously
- ✓ No neighbors: Isolated agent handling
- ✓ Role separation: Correct grouping by role
- ✓ Averaging correctness: Mathematical verification
- ✓ Partial masking: Variable neighborhood sizes
- ✓ Hierarchical mode: Distance-based grouping
- ✓ Learned encoder: Gradient flow verification
- ✓ Environment integration: Real-world format
- ✓ Scalability: O(N) complexity verification

## Ablation Studies

### 1. Test Mean Field vs. Individual Processing

Compare encoding efficiency:

```python
# Baseline: Process all neighbors individually (O(N))
individual_features = neighbor_states.reshape(batch, -1)  # (batch, K*6)

# Mean Field: Fixed-size encoding (O(1))
mean_field_features = encoder(neighbor_states, neighbor_roles, mask)  # (batch, 18)

# Train two policies and compare:
# - Sample efficiency (data required to converge)
# - Scalability (performance as num_neighbors increases)
# - Generalization (transfer to different team sizes)
```

### 2. Test Role-Based vs. Uniform Aggregation

Compare against treating all neighbors identically:

```python
# Uniform aggregation (no role separation)
uniform_mean = neighbor_states.mean(dim=1)  # (batch, 6)

# Role-based aggregation (our method)
role_mean = encoder(neighbor_states, neighbor_roles, mask)  # (batch, 18)

# Hypothesis: Role-based should perform better in heterogeneous teams
# where different roles have distinct behaviors
```

### 3. Test Trust Integration

Evaluate impact of trust weighting:

```python
# Without trust
mean_field_no_trust = encoder(states, roles, mask, trust_weights=None)

# With trust (from Trust Gate)
mean_field_with_trust = encoder(states, roles, mask, trust_weights)

# Hypothesis: Trust weighting should improve robustness to faulty agents
# Test in environments with Byzantine/faulty agents
```

## Research Applications

This module supports research in:

1. **Scalable MARL**: Coordination with 100+ agents
2. **Heterogeneous Teams**: Role-based coordination strategies
3. **Robust MARL**: Integration with trust mechanisms
4. **Transfer Learning**: Fixed encoding enables transfer across team sizes

## Citation

If you use this module in your research, please cite:

```bibtex
@article{yang2018mean,
  title={Mean field multi-agent reinforcement learning},
  author={Yang, Yaodong and Luo, Rui and Li, Minne and Zhou, Ming and Zhang, Weinan and Wang, Jun},
  journal={International Conference on Machine Learning},
  year={2018}
}
```

For heterogeneous extensions, cite relevant survey papers on heterogeneous multi-agent systems.

## Future Extensions

Potential enhancements:
- [ ] Graph neural network-based aggregation
- [ ] Temporal mean fields (incorporate neighbor trajectory history)
- [ ] Adaptive role discovery (learn roles instead of pre-defining)
- [ ] Multi-scale hierarchical grouping (role × distance × resource level)

## License

Part of the Cognitive Swarm research framework.
