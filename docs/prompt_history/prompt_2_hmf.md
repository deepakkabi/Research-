# Prompt 2: HMF Encoder Module

## Overview

This prompt specified the implementation of the **Heterogeneous Mean Field (HMF) Encoder** 
for scalable multi-agent coordination.

## Key Requirements

1. **Role-Based Aggregation**: Aggregate neighbor states by role (Scout, Coordinator, Support)
2. **Input**: `(batch, max_neighbors, state_dim=6)`
3. **Output**: `(batch, num_roles * state_dim)` = `(batch, 18)`
4. **Trust Integration**: Accept optional `trust_weights` from Trust Gate

## Mathematical Specification

For each role r:
```
μ_r = Σ_j (w_j * s_j) / Σ_j w_j
```

Where:
- `w_j` = trust_weight * role_mask * neighbor_mask
- `s_j` = neighbor state vector (dim=6)

## Dimensional Flow

```
neighbor_states: (batch, N, 6)  
neighbor_roles:  (batch, N)     
neighbor_mask:   (batch, N)     
trust_weights:   (batch, N)     
                    ↓
              HMFEncoder
                    ↓
mean_field:      (batch, 18)    # 3 roles × 6 state_dim
```

## Implementation File

- Location: `cognitive_swarm/modules/hmf_encoder.py`
- Classes: `HMFEncoder`, `LearnedHMFEncoder`
- Tests: `tests/test_hmf.py`

## Related Prompts

- **Prompt 4 (Trust Gate)**: Provides `trust_weights` input
- **Prompt 6 (Integration)**: Uses HMF output in policy network
