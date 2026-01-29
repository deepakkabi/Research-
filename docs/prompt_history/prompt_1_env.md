# Prompt 1: Environment
PROMPT 1: ENVIRONMENT
ACADEMIC RESEARCH DISCLAIMER:
This is for multi-agent reinforcement learning research at [Your Institution/Lab].
The goal is to advance understanding of coordination, safety, and robustness 
in abstract multi-agent systems. This is NOT intended for real-world deployment.

---

You are implementing a multi-agent reinforcement learning environment for 
research on scalable coordination under adversarial conditions. This is a 
PettingZoo-based simulation for academic research.

### RESEARCH CONTEXT

This environment is designed to test:
- Scalability (100-1000 agents)
- Robustness to communication failures
- Coordination under partial observability
- Safety constraint enforcement

This is an ABSTRACT research environment, not a real-world application.

### REQUIREMENTS

1. **Base Framework**
   - Inherit from `pettingzoo.ParallelEnv`
   - Support 100-1000 agents (configurable)
   - Agents have 3 role types: Scout (0), Coordinator (1), Support (2)

2. **Environment Dynamics**
   - Grid world: 50x50 cells
   - Each cell can contain: agent, obstacle, protected_entity, adversarial_agent
   - Agents have local observation radius: 5 cells
   - Actions: 7 discrete actions (see Action Space below)

3. **Action Space Specification**

   **CRITICAL**: Actions are discrete integers with the following mapping:
```python
   MOVE_NORTH = 0
   MOVE_SOUTH = 1
   MOVE_EAST = 2
   MOVE_WEST = 3
   INTERVENTION = 4  # High-impact action
   COMMUNICATE = 5   # Broadcast message
   HOLD = 6          # No action
   
   action_space = Discrete(7)
```
   
   **These indices MUST match across all modules (Shield, Agent, Training Loop)**

4. **Observation Space** (per agent)
   
   Return a dictionary with:
```python
   {
       'local_grid': np.array (5, 5, 4),  # 4 channels: teammates, adversaries, protected, obstacles
       'self_state': np.array (6,),        # [x, y, health, resource, role_id, team_id]
       'messages': List[np.array],         # Messages from neighbors (variable length)
       'neighbor_ids': np.array,           # IDs of agents within comm range
   }
```
   
   **Dimension specifications**:
   - `self_state` dimension = 6 (this becomes `state_dim` in other modules)
   - `messages` dimension = 8 per message
   - `local_grid` is NOT used directly in state_dim calculations

5. **Communication Model**
   
   - Agents can communicate within radius=7 cells
   - Messages are vectors: `np.array (8,)` containing:
```python
     [sender_id, role, x, y, status, target_x, target_y, priority]
```
   
   - **Critical Feature**: Implement `inject_noise()` method:
     * Randomly select 20% of agents each step
     * Noisy channels broadcast Gaussian noise: N(0, σ=2.0) instead of real messages
     * Affected agents do NOT know their channel is corrupted (no flag in observation)
     * This simulates communication failures / adversarial interference

6. **Reward Structure**
```python
   reward = (
       +10 * adversaries_neutralized
       -5 * teammates_lost
       -20 * protected_entities_harmed  # Critical for safety testing
       +1 * mission_progress
   )
```

7. **Episode Termination**
   - Max steps: 500
   - OR all adversaries neutralized
   - OR >50% agents disabled

### IMPLEMENTATION SPECIFICATIONS

**Method 1: `reset()`**
- Place 100 friendly agents (roles distributed 50% Scout, 30% Coordinator, 20% Support)
- Place 30 adversarial agents (stationary or simple scripted behavior)
- Place 10 protected entities randomly
- Return initial observations dict

**Method 2: `step(actions_dict)`**
- Input: `{agent_id: action_int}` for all agents
- Process movement (resolve collisions)
- Process intervention actions (check range, update status)
- Process communications (call `inject_noise()`)
- Return: `observations, rewards, dones, infos`

**Method 3: `inject_noise(messages_dict)`**
```python
def inject_noise(self, messages_dict, noise_probability=0.2):
    """
    Simulate communication channel corruption.
    Replace 20% of messages with Gaussian noise.
    
    Args:
        messages_dict: {agent_id: message_vector}
        noise_probability: Fraction of channels to corrupt
    
    Returns:
        corrupted_messages: Same structure but some vectors replaced with noise
    
    Critical: Do NOT mark which messages are corrupted (recipients must infer)
    
    Research purpose: Test robustness to unreliable communication
    
    Implementation note:
    - Use Gaussian noise N(0, σ=2.0)
    - Real messages have typical magnitude ~[0-10] range
    - This noise level is calibrated with Trust Gate threshold=0.5
    """
    pass
```

**Method 4: `get_protected_distances()`**
```python
def get_protected_distances(self, agent_id):
    """
    Return distances to all protected entities from agent's position.
    Needed for safety constraint testing.
    
    Returns:
        np.array of shape (num_protected_entities,)
    """
    pass
```

**Method 5: `collate_observations()` - CRITICAL FOR INTEGRATION**
```python
def collate_observations(self, obs_list, max_neighbors=20):
    """
    Convert list of observations to batched tensors for neural networks.
    
    This is REQUIRED for integration with Trust Gate and other neural modules.
    
    Args:
        obs_list: List of observation dicts from environment
        max_neighbors: Maximum number of neighbors to consider
    
    Returns:
        batched_obs: Dict with tensors of shape (batch, ...)
    """
    batch_size = len(obs_list)
    
    # Pad messages to fixed length
    all_messages = []
    all_masks = []
    all_neighbor_ids = []
    
    for obs in obs_list:
        msgs = obs['messages']
        ids = obs['neighbor_ids']
        num_msgs = len(msgs)
        
        if num_msgs < max_neighbors:
            # Pad with zeros
            padded_msgs = msgs + [np.zeros(8)] * (max_neighbors - num_msgs)
            padded_ids = np.concatenate([ids, np.zeros(max_neighbors - num_msgs)])
            mask = [1] * num_msgs + [0] * (max_neighbors - num_msgs)
        else:
            # Truncate
            padded_msgs = msgs[:max_neighbors]
            padded_ids = ids[:max_neighbors]
            mask = [1] * max_neighbors
        
        all_messages.append(np.stack(padded_msgs))
        all_masks.append(np.array(mask))
        all_neighbor_ids.append(padded_ids)
    
    return {
        'local_obs': torch.tensor(np.stack([o['self_state'] for o in obs_list]), dtype=torch.float32),
        'messages': torch.tensor(np.stack(all_messages), dtype=torch.float32),
        'neighbor_mask': torch.tensor(np.stack(all_masks), dtype=torch.float32),
        'neighbor_ids': torch.tensor(np.stack(all_neighbor_ids), dtype=torch.long),
        # Also need to extract neighbor states for Mean Field
        'neighbor_states': torch.tensor(np.stack(all_messages)[:, :, :6], dtype=torch.float32),  # First 6 dims
        'neighbor_roles': torch.tensor(np.stack(all_messages)[:, :, 1], dtype=torch.long)  # Role is index 1
    }
```

### DESIGN DECISIONS YOU MUST EXPLAIN

1. **Why 20% noise rate?**
   - Based on communication failure rates in distributed systems research
   - Cite: "Fault Tolerance in Multi-Agent Systems" (academic)

2. **Why Gaussian noise with σ=2.0?**
   - Real messages have magnitude ~[0-10]
   - σ=2.0 produces noise ~[-4, +4] (95% confidence)
   - Signal-to-noise ratio ~2.5:1 makes detection challenging but feasible
   - Calibrated with Trust Gate threshold=0.5

3. **Alternative noise models:**
   - Targeted corruption (specific agents)
   - Burst corruption (temporal patterns)
   - Explain trade-offs

### TESTING REQUIREMENTS

Include test functions:
```python
def test_noise_rate():
    """Verify that approximately 20% of messages are corrupted over 100 steps"""
    pass

def test_observation_consistency():
    """Verify that agents can't observe through obstacles"""
    pass

def test_protected_proximity():
    """Verify that protected entity distances are computed correctly"""
    pass

def test_action_space():
    """Verify action space has exactly 7 actions with correct indices"""
    env = CoordinationEnv(num_agents=10)
    assert env.action_space.n == 7
    # Test each action executes correctly
    pass

def test_batching():
    """Verify collate_observations produces correct tensor shapes"""
    env = CoordinationEnv(num_agents=10)
    obs = env.reset()
    obs_list = list(obs.values())
    
    batched = env.collate_observations(obs_list, max_neighbors=20)
    
    assert batched['messages'].shape == (10, 20, 8)
    assert batched['neighbor_mask'].shape == (10, 20)
    assert batched['neighbor_states'].shape == (10, 20, 6)
    print("✓ Batching test passed")
```

### OUTPUT FORMAT

Provide:
1. Complete `coordination_env.py` file
2. A separate `test_env.py` file with all tests
3. A `README.md` explaining:
   - How to instantiate the environment
   - Example usage code
   - Parameter tuning guidelines
   - Integration with neural network modules

### CODE QUALITY STANDARDS

- Use type hints for all methods
- Add docstrings (Google style)
- Include assert statements for invariants
- Handle edge cases (empty neighbor lists, out-of-bounds actions)
- Use numpy for efficiency (no Python loops for computations)

### ACADEMIC CONTEXT

This environment is for research on:
- Multi-agent coordination at scale
- Robustness to communication failures
- Safety constraints in high-stakes systems
- Adversarial robustness in distributed systems

Citation: This is inspired by research in cooperative MARL (cite: "QMIX", "MADDPG")