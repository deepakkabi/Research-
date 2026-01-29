# Cognitive Swarm - Integration Guide

## Overview

This guide explains how the 6 modules in the Cognitive Swarm framework interconnect to form the **Secure Decision Pipeline**.

## Module Connection Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURE DECISION PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

          ┌──────────────────┐
          │   Environment    │
          │ CoordinationEnv  │
          └────────┬─────────┘
                   │
    ┌──────────────┼──────────────────┐
    │              │                  │
    ▼              ▼                  ▼
local_obs(10)  messages(N,8)  neighbor_states(N,6)
    │              │                  │
    │              ▼                  │
    │     ┌────────────────┐          │
    │     │   Trust Gate   │          │
    │     │ (Module 1)     │          │
    │     └───────┬────────┘          │
    │             │                   │
    │     reliability_weights(N)      │
    │             │                   │
    │             ▼                   │
    │     ┌────────────────┐          │
    │     │  Bayesian      │◄─────────┤ other_team_obs
    │     │  Beliefs (M2)  │          │
    │     └───────┬────────┘          │
    │             │                   │
    │     belief_state(4)             │
    │             │                   │
    │             ▼                   ▼
    │     ┌────────────────────────────┐
    │     │     HMF Encoder (M3)       │
    │     │  + trust_weights filter    │
    │     └────────────┬───────────────┘
    │                  │
    │           mean_field(18)
    │                  │
    ▼                  ▼
┌──────────────────────────────────────┐
│   CONCATENATE: obs + mf + belief     │
│           policy_input(28-32)        │
└──────────────────┬───────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Policy Network │
          │    (Module 5)  │
          └───────┬────────┘
                  │
          action_logits(7)
                  │
                  ▼
          ┌────────────────┐
          │  Safety Shield │
          │    (Module 6)  │
          └───────┬────────┘
                  │
            final_action
```

## Data Flow Summary

| Source | Output | Destination | Shape |
|--------|--------|-------------|-------|
| Environment | `local_obs` | Policy | (batch, 10) |
| Environment | `messages` | Trust Gate | (batch, N, 8) |
| Environment | `neighbor_states` | HMF Encoder | (batch, N, 6) |
| Trust Gate | `reliability_weights` | HMF Encoder | (batch, N) |
| Bayesian Beliefs | `belief_state` | Policy | (batch, 4) |
| HMF Encoder | `mean_field` | Policy | (batch, 18) |
| Policy | `action_logits` | Safety Shield | (batch, 7) |
| Safety Shield | `final_action` | Environment | int |

## Integration Points

### 1. Trust Gate → HMF Encoder

```python
# In CognitiveAgent.forward():
filtered_messages, reliability_weights = self.trust_gate(messages, local_obs)

mean_field = self.hmf_encoder(
    neighbor_states=neighbor_states,
    trust_weights=reliability_weights  # Only reliable neighbors contribute
)
```

### 2. Bayesian Beliefs → Policy (Optional, V2)

```python
if self.use_beliefs and self.belief_module is not None:
    belief_state = self.belief_module(other_team_obs)
    policy_input = torch.cat([local_obs, mean_field, belief_state], dim=-1)
else:
    policy_input = torch.cat([local_obs, mean_field], dim=-1)
```

### 3. Policy → Safety Shield

```python
# Safety applied AFTER action selection (in select_action method)
action_logits, value, info = agent(...)
final_action, violated, reason = agent.select_action(
    action_logits[0], state_dict, agent_id
)
```

## Dimension Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `obs_dim` | 10 | Local observation (x, y, health, ...) |
| `message_dim` | 8 | Inter-agent message size |
| `state_dim` | 6 | Neighbor state (x, y, health, resource, role, team) |
| `action_dim` | 7 | Discrete actions (MOVE_*, HOLD, INTERVENTION) |
| `num_roles` | 3 | Scout, Coordinator, Support |
| `num_types` | 4 | Strategy types (V2) |

## Quick Start

```python
from cognitive_swarm import CognitiveAgent

# V1 Configuration (without beliefs)
agent = CognitiveAgent(
    obs_dim=10,
    message_dim=8,
    state_dim=6,
    action_dim=7,
    use_beliefs=False
)

# V2 Configuration (with beliefs)
agent = CognitiveAgent(
    obs_dim=10,
    use_beliefs=True  # Enables 4-type opponent modeling
)
```

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture
- [DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md) - Project summary
- [integration_notes.md](integration_notes.md) - Technical notes
