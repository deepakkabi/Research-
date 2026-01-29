# Safety Constraint Module (Shield)

## Overview

The Safety Constraint Module enforces operational safety rules as **hard constraints** that cannot be violated in multi-agent reinforcement learning systems. This is Module 3 of the Cognitive Swarm framework.

## Key Features

✅ **Hard Constraints** - Actions are blocked before execution, providing mathematical safety guarantees  
✅ **Soft Guidance** - Reward shaping helps policies learn to avoid violations  
✅ **Interpretable Rules** - Symbolic logic makes decisions transparent and auditable  
✅ **Parameter Tuning** - Adjust risk tolerance without changing code  
✅ **Research Ready** - Comprehensive statistics and logging for analysis

## Installation

The module requires NumPy (PyTorch is optional for neural cost estimation):

```bash
pip install numpy
# Optional: pip install torch  # For neural cost estimator
```

## Quick Start

```python
from cognitive_swarm.governance import SafetyConstraintModule

# Create shield with default parameters
shield = SafetyConstraintModule(
    safe_distance=5.0,           # Minimum distance to protected entities
    proportionality_lambda=1.0   # Cost-benefit weight
)

# Check if an action is safe
state_dict = {
    'agent_positions': np.array([[0.0, 0.0]]),
    'protected_positions': np.array([[10.0, 10.0]]),
    'target_positions': np.array([[2.0, 0.0]]),
    'agent_resources': np.array([10]),
    'target_values': np.array([5.0])
}

final_action, violated, reason = shield.verify_action(
    proposed_action=4,  # INTERVENTION
    state_dict=state_dict,
    agent_id=0
)

if violated:
    print(f"Action blocked: {reason}")
    # final_action will be HOLD (6)
else:
    print("Action allowed")
    # final_action will be INTERVENTION (4)
```

## Safety Constraints

### Rule 1: Protected Entity Safety (Hard Constraint)

**Purpose:** Maintain minimum distance from protected entities  
**Formula:** `BLOCK if min_distance < safe_distance`  
**Default:** 5.0 cells  

**Example:**
```python
shield = SafetyConstraintModule(safe_distance=7.0)  # More conservative
```

### Rule 2: Proportionality (Cost-Benefit Constraint)

**Purpose:** Ensure expected benefit exceeds potential cost  
**Formula:** `Proportionality_Score = Benefit - λ × Cost`  
**Block if:** Score < 0  

**Mathematical Details:**
- Benefit = Σ(target_value_i × P(success_i))
- Cost = Σ(collateral_impact_i × P(occurrence_i))

**Example:**
```python
shield = SafetyConstraintModule(
    proportionality_lambda=1.5  # Weigh costs more heavily
)
```

### Rule 3: Resource Conservation

**Purpose:** Preserve resources for high-value opportunities  
**Formula:** `BLOCK if resource < threshold AND target_value < high_value_threshold`  
**Defaults:** threshold=2, high_value=5.0

**Example:**
```python
shield = SafetyConstraintModule(
    reserve_threshold=3,      # Higher reserve
    high_value_threshold=8.0  # Stricter high-value definition
)
```

## Integration with Training

### Blocking Unsafe Actions

```python
# In your agent's action selection
proposed_action = policy.select_action(state)

# Apply safety shield
safe_action, violated, reason = shield.verify_action(
    proposed_action, 
    state_dict,
    agent_id
)

# Execute the safe action
next_state, reward, done, info = env.step(safe_action)
```

### Reward Shaping

```python
# In your training loop
base_reward = env_reward

# Add safety penalty to guide learning
safety_penalty = shield.compute_reward_penalty(
    proposed_action,
    state_dict,
    agent_id
)

total_reward = base_reward + safety_penalty

# Update policy with total reward
policy.update(state, action, total_reward, next_state)
```

## Parameter Tuning Guide

### Safe Distance

| Value | Use Case | Strictness |
|-------|----------|-----------|
| 3.0 | High-precision systems | Permissive |
| 5.0 | **Default - Balanced** | Moderate |
| 7.0 | High-risk scenarios | Conservative |
| 10.0 | Maximum protection | Very restrictive |

### Proportionality Lambda

| Value | Risk Tolerance | Decision Bias |
|-------|---------------|---------------|
| 0.5 | High risk | Benefit-favoring |
| 1.0 | **Default - Balanced** | Symmetric |
| 1.5 | Low risk | Cost-aware |
| 2.0 | Very low risk | Cost-averse |

**Formula Impact:**
```
Score = Benefit - λ × Cost

Example with Benefit=5, Cost=4:
- λ=0.5: Score = 3.0  ✓ ALLOW
- λ=1.0: Score = 1.0  ✓ ALLOW
- λ=1.5: Score = -1.0 ✗ BLOCK
- λ=2.0: Score = -3.0 ✗ BLOCK
```

### Action Range and Splash Radius

```python
shield = SafetyConstraintModule(
    action_range=3.0,      # How far actions reach
    splash_radius=1.5      # Collateral damage zone
)
# Risk Zone = action_range + splash_radius = 4.5 cells
```

## Research Applications

### Statistics Tracking

```python
# Get violation statistics
stats = shield.get_constraint_statistics()
print(stats)
# Output:
# {
#     'Protected Entity Constraint': 0.05,  # 5% of checks
#     'Proportionality Constraint': 0.12,   # 12% of checks
#     'Resource Conservation Constraint': 0.03  # 3% of checks
# }

# Reset for new experiment
shield.reset_statistics()
```

### Ablation Studies

Compare policies trained with different configurations:

```python
configs = {
    'baseline': SafetyConstraintModule(proportionality_lambda=1.0),
    'conservative': SafetyConstraintModule(proportionality_lambda=2.0),
    'permissive': SafetyConstraintModule(proportionality_lambda=0.5)
}

for name, shield in configs.items():
    policy = train_policy(env, shield, episodes=10000)
    results[name] = evaluate_policy(policy, episodes=1000)
```

## Action Space

The module uses the following action indices (must match your environment):

```python
MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
INTERVENTION = 4  # The action we regulate
COMMUNICATE = 5
HOLD = 6  # Safe fallback action
```

Only the INTERVENTION action (4) is subject to safety constraints. All other actions pass through unchanged.

## State Dictionary Format

The module expects states in this format:

```python
state_dict = {
    'agent_positions': np.array([[x1, y1], [x2, y2], ...]),    # (num_agents, 2)
    'protected_positions': np.array([[x1, y1], ...]),          # (num_protected, 2)
    'target_positions': np.array([[x1, y1], ...]),             # (num_targets, 2)
    'agent_resources': np.array([r1, r2, ...]),                # (num_agents,)
    'target_values': np.array([v1, v2, ...])                   # (num_targets,)
}
```

## Advanced Usage

### Custom Cost Estimation

The module includes a placeholder for neural cost estimation (requires PyTorch):

```python
# Train cost estimator on historical data
state_features = extract_features(state_dict)
estimated_cost = shield.forward(state_features)  # Neural network

# Note: Currently uses heuristic-based estimation by default
```

### Multi-Agent Scenarios

```python
# Each agent gets its own safety check
for agent_id in range(num_agents):
    action = policies[agent_id].select_action(state)
    safe_action, _, _ = shield.verify_action(action, state_dict, agent_id)
    actions[agent_id] = safe_action
```

## Testing

Run the comprehensive test suite:

```bash
cd /home/claude
PYTHONPATH=/home/claude python tests/test_shield.py
```

Expected output: All tests passing ✓

## Documentation

See `docs/safety_justification.md` for:
- Ethical justification for constraints
- Parameter sensitivity analysis
- Research contributions
- Ablation study designs

## Citation

If you use this module in your research, please cite:

```bibtex
@software{cognitive_swarm_shield,
  title={Safety Constraint Module for Multi-Agent Reinforcement Learning},
  author={Cognitive Swarm Research Team},
  year={2026},
  version={0.1.0}
}
```

## License

Part of the Cognitive Swarm framework. See main repository for license details.

## Related Work

- Amodei et al., "Concrete Problems in AI Safety" (2016)
- Safe RL, Constrained MDPs, Neuro-symbolic AI
- International Humanitarian Law principles
