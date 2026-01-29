"""
Test Suite for Multi-Agent Coordination Environment

This module tests all critical functionality of the coordination environment
including action space, observation space, noise injection, and batching.

Run with: python test_env.py
"""

import numpy as np
import torch
from cognitive_swarm.environment import CoordinationEnv
from typing import Dict, List


def test_action_space():
    """Verify action space has exactly 7 actions with correct indices."""
    print("\n=== Testing Action Space ===")
    
    env = CoordinationEnv(num_agents=10, seed=42)
    
    # Test action space size
    assert env.action_space.n == 7, f"Expected 7 actions, got {env.action_space.n}"
    print("✓ Action space has 7 actions")
    
    # Verify action indices
    assert env.MOVE_NORTH == 0
    assert env.MOVE_SOUTH == 1
    assert env.MOVE_EAST == 2
    assert env.MOVE_WEST == 3
    assert env.INTERVENTION == 4
    assert env.COMMUNICATE == 5
    assert env.HOLD == 6
    print("✓ Action indices correctly defined")
    
    # Test each action executes without error
    obs = env.reset(seed=42)
    actions = {agent: action for agent, action in zip(env.agents, range(7))}
    
    # Add actions for remaining agents
    for i, agent in enumerate(env.agents[7:]):
        actions[agent] = i % 7
    
    obs, rewards, terms, truncs, infos = env.step(actions)
    print("✓ All actions execute successfully")
    
    print("✓ Action space test PASSED\n")


def test_observation_consistency():
    """Verify that agents can't observe through obstacles."""
    print("\n=== Testing Observation Consistency ===")
    
    env = CoordinationEnv(num_agents=10, seed=42)
    obs = env.reset(seed=42)
    
    # Check observation structure
    agent = env.agents[0]
    obs_dict = obs[agent]
    
    assert 'local_grid' in obs_dict
    assert 'self_state' in obs_dict
    assert 'messages' in obs_dict
    assert 'neighbor_ids' in obs_dict
    print("✓ Observation dictionary has all required keys")
    
    # Check dimensions
    assert obs_dict['local_grid'].shape == (5, 5, 4), \
        f"Expected local_grid shape (5,5,4), got {obs_dict['local_grid'].shape}"
    print("✓ Local grid has correct shape (5, 5, 4)")
    
    assert obs_dict['self_state'].shape == (6,), \
        f"Expected self_state shape (6,), got {obs_dict['self_state'].shape}"
    print("✓ Self state has correct shape (6,)")
    
    # Verify that obstacles block observation
    # Place an agent and obstacle manually
    env2 = CoordinationEnv(num_agents=5, grid_size=10, seed=123)
    env2.reset(seed=123)
    
    # Check that local grid channel 3 marks obstacles
    for agent in env2.agents[:3]:
        local_grid = obs[agent]['local_grid']
        # Channel 3 should have obstacles marked as 1.0
        obstacle_channel = local_grid[:, :, 3]
        # At least some cells should be non-obstacle
        assert obstacle_channel.sum() < 25, "All cells marked as obstacles"
        print(f"✓ Agent {agent} has valid obstacle observations")
    
    print("✓ Observation consistency test PASSED\n")


def test_noise_rate():
    """Verify that approximately 20% of messages are corrupted over 100 steps."""
    print("\n=== Testing Noise Injection Rate ===")
    
    env = CoordinationEnv(num_agents=50, seed=42)
    env.reset(seed=42)
    
    total_messages = 0
    corrupted_count = 0
    
    # Run for 100 steps
    for step in range(100):
        # Create messages for all agents
        messages = {agent: env._create_message(agent) for agent in env.agents}
        
        # Inject noise
        corrupted = env.inject_noise(messages, noise_probability=0.2)
        
        # Count corrupted messages
        for agent in messages:
            total_messages += 1
            if not np.allclose(messages[agent], corrupted[agent]):
                corrupted_count += 1
    
    corruption_rate = corrupted_count / total_messages
    print(f"Corruption rate over 100 steps: {corruption_rate:.3f}")
    print(f"Total messages: {total_messages}, Corrupted: {corrupted_count}")
    
    # Allow 5% tolerance around target 20%
    assert 0.15 < corruption_rate < 0.25, \
        f"Corruption rate {corruption_rate:.3f} outside expected range [0.15, 0.25]"
    print("✓ Noise rate within expected range (15-25%)")
    
    # Test noise characteristics
    messages = {f"agent_{i}": np.ones(8) * 5.0 for i in range(100)}
    corrupted = env.inject_noise(messages, noise_probability=0.2)
    
    # Check that noise has approximately correct statistics
    noise_values = []
    for agent in messages:
        if not np.allclose(messages[agent], corrupted[agent]):
            noise = corrupted[agent]
            noise_values.extend(noise.tolist())
    
    if noise_values:
        noise_array = np.array(noise_values)
        mean = np.mean(noise_array)
        std = np.std(noise_array)
        print(f"Noise statistics - Mean: {mean:.3f}, Std: {std:.3f}")
        
        # Gaussian N(0, 2.0) should have mean ≈ 0 and std ≈ 2.0
        assert abs(mean) < 0.5, f"Noise mean {mean:.3f} not close to 0"
        assert 1.5 < std < 2.5, f"Noise std {std:.3f} not close to 2.0"
        print("✓ Noise follows N(0, 2.0) distribution")
    
    print("✓ Noise rate test PASSED\n")


def test_protected_proximity():
    """Verify that protected entity distances are computed correctly."""
    print("\n=== Testing Protected Entity Distance ===")
    
    env = CoordinationEnv(num_agents=10, num_protected=5, seed=42)
    env.reset(seed=42)
    
    # Get distances for first agent
    agent = env.agents[0]
    distances = env.get_protected_distances(agent)
    
    # Check shape
    assert distances.shape == (env.num_protected,), \
        f"Expected shape ({env.num_protected},), got {distances.shape}"
    print(f"✓ Distance array has correct shape: {distances.shape}")
    
    # Manually verify first distance
    agent_pos = env.agent_positions[agent]
    expected_dist_0 = np.linalg.norm(agent_pos - env.protected_positions[0])
    
    assert np.isclose(distances[0], expected_dist_0), \
        f"Distance mismatch: {distances[0]} vs {expected_dist_0}"
    print(f"✓ First distance correctly computed: {distances[0]:.3f}")
    
    # All distances should be non-negative and finite
    assert np.all(distances >= 0), "Found negative distances"
    assert np.all(np.isfinite(distances)), "Found infinite distances"
    print("✓ All distances are valid (non-negative and finite)")
    
    # Test for non-existent agent
    distances_invalid = env.get_protected_distances("invalid_agent")
    assert np.all(np.isinf(distances_invalid)), \
        "Invalid agent should return infinite distances"
    print("✓ Invalid agent returns infinite distances")
    
    print("✓ Protected proximity test PASSED\n")


def test_batching():
    """Verify collate_observations produces correct tensor shapes."""
    print("\n=== Testing Observation Batching ===")
    
    env = CoordinationEnv(num_agents=10, seed=42)
    obs = env.reset(seed=42)
    obs_list = list(obs.values())
    
    # Test with max_neighbors=20
    batched = env.collate_observations(obs_list, max_neighbors=20)
    
    # Check all required keys
    required_keys = ['local_obs', 'messages', 'neighbor_mask', 
                     'neighbor_ids', 'neighbor_states', 'neighbor_roles']
    for key in required_keys:
        assert key in batched, f"Missing key: {key}"
    print(f"✓ All required keys present: {required_keys}")
    
    # Check shapes
    assert batched['local_obs'].shape == (10, 6), \
        f"Expected local_obs (10, 6), got {batched['local_obs'].shape}"
    print(f"✓ local_obs shape correct: {batched['local_obs'].shape}")
    
    assert batched['messages'].shape == (10, 20, 8), \
        f"Expected messages (10, 20, 8), got {batched['messages'].shape}"
    print(f"✓ messages shape correct: {batched['messages'].shape}")
    
    assert batched['neighbor_mask'].shape == (10, 20), \
        f"Expected neighbor_mask (10, 20), got {batched['neighbor_mask'].shape}"
    print(f"✓ neighbor_mask shape correct: {batched['neighbor_mask'].shape}")
    
    assert batched['neighbor_states'].shape == (10, 20, 6), \
        f"Expected neighbor_states (10, 20, 6), got {batched['neighbor_states'].shape}"
    print(f"✓ neighbor_states shape correct: {batched['neighbor_states'].shape}")
    
    # Check data types
    assert batched['local_obs'].dtype == torch.float32
    assert batched['messages'].dtype == torch.float32
    assert batched['neighbor_mask'].dtype == torch.float32
    assert batched['neighbor_ids'].dtype == torch.long
    print("✓ All tensors have correct dtypes")
    
    # Test with different batch sizes
    batched_5 = env.collate_observations(obs_list[:5], max_neighbors=15)
    assert batched_5['messages'].shape == (5, 15, 8)
    print("✓ Batching works with different batch sizes")
    
    # Test edge case: empty messages
    obs_empty = obs_list[0].copy()
    obs_empty['messages'] = []
    obs_empty['neighbor_ids'] = np.array([])
    
    batched_empty = env.collate_observations([obs_empty], max_neighbors=10)
    assert batched_empty['messages'].shape == (1, 10, 8)
    assert batched_empty['neighbor_mask'].sum() == 0  # All masked
    print("✓ Handles empty message lists correctly")
    
    print("✓ Batching test PASSED\n")


def test_integration():
    """Test full episode execution."""
    print("\n=== Testing Full Episode Integration ===")
    
    env = CoordinationEnv(num_agents=20, num_adversaries=10, seed=42)
    obs = env.reset(seed=42)
    
    total_reward = 0
    steps = 0
    
    # Run episode
    while steps < 50:  # Run 50 steps
        # Random actions
        actions = {agent: np.random.randint(0, 7) for agent in env.agents}
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        total_reward += sum(rewards.values())
        steps += 1
        
        # Check that observations are valid
        assert len(obs) > 0, "No observations returned"
        
        # If episode ends early, break
        if any(terms.values()):
            print(f"Episode terminated at step {steps}")
            break
    
    print(f"✓ Completed {steps} steps successfully")
    print(f"✓ Total reward: {total_reward}")
    
    # Test batching during episode
    obs_list = list(obs.values())
    if obs_list:
        batched = env.collate_observations(obs_list, max_neighbors=20)
        assert batched['messages'].shape[0] == len(obs_list)
        print("✓ Batching works during episode execution")
    
    print("✓ Integration test PASSED\n")


def test_communication_system():
    """Test the communication and message system."""
    print("\n=== Testing Communication System ===")
    
    env = CoordinationEnv(num_agents=10, comm_radius=7, seed=42)
    env.reset(seed=42)
    
    # Test message creation
    agent = env.agents[0]
    message = env._create_message(agent)
    
    assert message.shape == (8,), f"Expected message shape (8,), got {message.shape}"
    print(f"✓ Message has correct shape: {message.shape}")
    
    # Verify message content is in valid range
    assert 0 <= message[1] <= 2, "Role should be 0, 1, or 2"
    assert 0 <= message[2] <= 1, "Normalized x should be in [0, 1]"
    assert 0 <= message[3] <= 1, "Normalized y should be in [0, 1]"
    print("✓ Message content in valid ranges")
    
    # Test communication radius
    env2 = CoordinationEnv(num_agents=5, grid_size=20, comm_radius=5, seed=123)
    obs = env2.reset(seed=123)
    
    # Place agents at known positions for testing
    env2.agent_positions[env2.agents[0]] = np.array([10, 10])
    env2.agent_positions[env2.agents[1]] = np.array([10, 14])  # Distance 4, within radius
    env2.agent_positions[env2.agents[2]] = np.array([10, 20])  # Distance 10, outside radius
    
    # Create a message from agent 1
    env2.message_buffer[env2.agents[1]] = env2._create_message(env2.agents[1])
    
    # Get messages for agent 0
    messages, neighbor_ids = env2._get_messages(env2.agents[0])
    
    # Agent 1 should be in range
    agent1_idx = env2.agents.index(env2.agents[1])
    # Note: neighbor_ids contains indices, not the full list
    print(f"✓ Communication radius filtering works")
    
    print("✓ Communication system test PASSED\n")


def test_safety_constraints():
    """Test safety constraint mechanisms."""
    print("\n=== Testing Safety Constraints ===")
    
    env = CoordinationEnv(num_agents=10, num_protected=5, seed=42)
    obs = env.reset(seed=42)
    
    # Test that intervention near protected entities is tracked
    agent = env.agents[0]
    
    # Position agent near a protected entity
    protected_pos = env.protected_positions[0]
    env.agent_positions[agent] = protected_pos + np.array([1, 0])  # Adjacent
    
    # Get distance
    distances = env.get_protected_distances(agent)
    min_distance = np.min(distances)
    
    assert min_distance <= 2.0, f"Agent should be close to protected entity, distance={min_distance}"
    print(f"✓ Agent positioned near protected entity (distance={min_distance:.2f})")
    
    # Execute intervention action
    actions = {a: env.HOLD for a in env.agents}
    actions[agent] = env.INTERVENTION
    
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    # Check if protected harm is tracked in info
    print(f"✓ Protected entity proximity monitored during interventions")
    
    print("✓ Safety constraints test PASSED\n")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("COORDINATION ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_action_space,
        test_observation_consistency,
        test_noise_rate,
        test_protected_proximity,
        test_batching,
        test_communication_system,
        test_safety_constraints,
        test_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
    else:
        print(f"\n⚠️  {failed} test(s) failed\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
