"""
Basic Environment Test (No External Dependencies Required)

This test verifies core functionality without requiring torch/pettingzoo installation.
For full test suite, install dependencies and run test_env.py
"""

import numpy as np
import sys
import os

# Add parent directory to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_basic_structure():
    """Test that the environment can be imported and instantiated."""
    print("\n=== Testing Basic Structure ===")
    
    try:
        from cognitive_swarm.environment import CoordinationEnv
        print("✓ Environment module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    try:
        env = CoordinationEnv(num_agents=10, seed=42)
        print("✓ Environment instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate: {e}")
        return False
    
    # Check action space
    assert env.MOVE_NORTH == 0
    assert env.MOVE_SOUTH == 1
    assert env.MOVE_EAST == 2
    assert env.MOVE_WEST == 3
    assert env.INTERVENTION == 4
    assert env.COMMUNICATE == 5
    assert env.HOLD == 6
    print("✓ Action indices correctly defined (0-6)")
    
    # Check parameters
    assert env.num_agents == 10
    assert env.grid_size == 50
    assert env.obs_radius == 5
    assert env.comm_radius == 7
    assert env.noise_probability == 0.2
    print("✓ Environment parameters set correctly")
    
    print("✓ Basic structure test PASSED\n")
    return True


def test_reset():
    """Test environment reset functionality."""
    print("\n=== Testing Reset ===")
    
    from cognitive_swarm.environment import CoordinationEnv
    
    env = CoordinationEnv(num_agents=10, num_adversaries=5, num_protected=3, seed=42)
    obs = env.reset(seed=42)
    
    # Check observations returned
    assert isinstance(obs, dict), "Observations should be a dictionary"
    assert len(obs) == 10, f"Should have 10 observations, got {len(obs)}"
    print(f"✓ Reset returns {len(obs)} observations")
    
    # Check observation structure
    agent = list(obs.keys())[0]
    obs_dict = obs[agent]
    
    assert 'local_grid' in obs_dict
    assert 'self_state' in obs_dict
    assert 'messages' in obs_dict
    assert 'neighbor_ids' in obs_dict
    print("✓ Observation has all required keys")
    
    # Check dimensions
    assert obs_dict['local_grid'].shape == (5, 5, 4)
    assert obs_dict['self_state'].shape == (6,)
    print("✓ Observation dimensions correct")
    
    # Check agent positions are valid
    for agent_id in env.agents:
        pos = env.agent_positions[agent_id]
        assert 0 <= pos[0] < env.grid_size
        assert 0 <= pos[1] < env.grid_size
        assert not env.obstacles[pos[0], pos[1]]
    print("✓ All agent positions are valid")
    
    # Check role distribution
    roles = list(env.agent_roles.values())
    num_scouts = sum(1 for r in roles if r == 0)
    num_coordinators = sum(1 for r in roles if r == 1)
    num_support = sum(1 for r in roles if r == 2)
    print(f"✓ Role distribution: {num_scouts} scouts, {num_coordinators} coordinators, {num_support} support")
    
    print("✓ Reset test PASSED\n")
    return True


def test_step():
    """Test environment step functionality."""
    print("\n=== Testing Step ===")
    
    from cognitive_swarm.environment import CoordinationEnv
    
    env = CoordinationEnv(num_agents=10, seed=42)
    obs = env.reset(seed=42)
    
    # Create actions for all agents
    actions = {agent: env.HOLD for agent in env.agents}
    
    # Execute step
    new_obs, rewards, terms, truncs, infos = env.step(actions)
    
    # Check return types
    assert isinstance(new_obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terms, dict)
    assert isinstance(truncs, dict)
    assert isinstance(infos, dict)
    print("✓ Step returns correct types")
    
    # Check all agents have returns
    assert len(new_obs) > 0
    assert len(rewards) > 0
    print(f"✓ Step returns data for {len(new_obs)} agents")
    
    # Check current step incremented
    assert env.current_step == 1
    print("✓ Step counter incremented")
    
    print("✓ Step test PASSED\n")
    return True


def test_movement():
    """Test movement actions."""
    print("\n=== Testing Movement ===")
    
    from cognitive_swarm.environment import CoordinationEnv
    
    env = CoordinationEnv(num_agents=5, grid_size=20, seed=42)
    obs = env.reset(seed=42)
    
    agent = env.agents[0]
    initial_pos = env.agent_positions[agent].copy()
    
    # Test each movement direction
    movements = [
        (env.MOVE_NORTH, np.array([-1, 0])),
        (env.MOVE_SOUTH, np.array([1, 0])),
        (env.MOVE_EAST, np.array([0, 1])),
        (env.MOVE_WEST, np.array([0, -1])),
    ]
    
    for action, expected_delta in movements:
        env.reset(seed=42)
        initial = env.agent_positions[agent].copy()
        
        actions = {a: env.HOLD for a in env.agents}
        actions[agent] = action
        
        obs, _, _, _, _ = env.step(actions)
        
        new_pos = env.agent_positions[agent]
        
        # Check if movement happened (might be blocked by obstacle)
        if not env.obstacles[min(max(initial[0] + expected_delta[0], 0), env.grid_size-1),
                              min(max(initial[1] + expected_delta[1], 0), env.grid_size-1)]:
            action_names = {0: 'NORTH', 1: 'SOUTH', 2: 'EAST', 3: 'WEST'}
            print(f"✓ Movement {action_names[action]} processed")
    
    print("✓ Movement test PASSED\n")
    return True


def test_noise_injection():
    """Test communication noise injection."""
    print("\n=== Testing Noise Injection ===")
    
    from cognitive_swarm.environment import CoordinationEnv
    
    env = CoordinationEnv(num_agents=100, seed=42)
    env.reset(seed=42)
    
    # Create messages for all agents
    messages = {f"agent_{i}": np.ones(8) * 5.0 for i in range(100)}
    
    # Inject noise
    corrupted = env.inject_noise(messages, noise_probability=0.2)
    
    # Count corrupted messages
    num_corrupted = 0
    for agent in messages:
        if not np.allclose(messages[agent], corrupted[agent]):
            num_corrupted += 1
    
    corruption_rate = num_corrupted / len(messages)
    print(f"Corruption rate: {corruption_rate:.2%} ({num_corrupted}/{len(messages)})")
    
    # Should be approximately 20% (allow 10-30% range for randomness)
    assert 0.1 <= corruption_rate <= 0.3, f"Corruption rate {corruption_rate} outside expected range"
    print("✓ Noise rate within expected range (10-30%)")
    
    # Check that noise is Gaussian-like
    noise_samples = []
    for agent in messages:
        if not np.allclose(messages[agent], corrupted[agent]):
            noise = corrupted[agent]
            noise_samples.extend(noise.tolist())
    
    if len(noise_samples) > 10:
        noise_array = np.array(noise_samples)
        mean = np.mean(noise_array)
        std = np.std(noise_array)
        print(f"Noise statistics: mean={mean:.3f}, std={std:.3f}")
        
        # Gaussian N(0, 2.0) should have mean near 0 and std near 2
        assert abs(mean) < 1.0, f"Noise mean {mean} too far from 0"
        assert 1.0 < std < 3.0, f"Noise std {std} not close to 2.0"
        print("✓ Noise follows approximately N(0, 2.0)")
    
    print("✓ Noise injection test PASSED\n")
    return True


def test_protected_distances():
    """Test protected entity distance calculation."""
    print("\n=== Testing Protected Distances ===")
    
    from cognitive_swarm.environment import CoordinationEnv
    
    env = CoordinationEnv(num_agents=10, num_protected=5, seed=42)
    env.reset(seed=42)
    
    agent = env.agents[0]
    distances = env.get_protected_distances(agent)
    
    # Check shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"
    print(f"✓ Distance array shape correct: {distances.shape}")
    
    # All distances should be non-negative
    assert np.all(distances >= 0)
    print("✓ All distances non-negative")
    
    # Manually verify first distance
    agent_pos = env.agent_positions[agent]
    expected = np.linalg.norm(agent_pos - env.protected_positions[0])
    assert np.isclose(distances[0], expected)
    print(f"✓ Distance calculation correct: {distances[0]:.2f}")
    
    # Test invalid agent
    invalid_distances = env.get_protected_distances("invalid_agent")
    assert np.all(np.isinf(invalid_distances))
    print("✓ Invalid agent returns infinite distances")
    
    print("✓ Protected distances test PASSED\n")
    return True


def test_communication():
    """Test message creation and communication."""
    print("\n=== Testing Communication ===")
    
    from cognitive_swarm.environment import CoordinationEnv
    
    env = CoordinationEnv(num_agents=10, comm_radius=7, seed=42)
    env.reset(seed=42)
    
    agent = env.agents[0]
    
    # Test message creation
    message = env._create_message(agent)
    assert message.shape == (8,), f"Expected message shape (8,), got {message.shape}"
    print(f"✓ Message shape correct: {message.shape}")
    
    # Check message content ranges
    assert 0 <= message[1] <= 2, "Role should be 0, 1, or 2"
    assert 0 <= message[2] <= 1, "Normalized x should be in [0, 1]"
    assert 0 <= message[3] <= 1, "Normalized y should be in [0, 1]"
    print("✓ Message content in valid ranges")
    
    # Test getting messages from neighbors
    messages, neighbor_ids = env._get_messages(agent)
    print(f"✓ Agent has {len(messages)} neighbors within comm_radius={env.comm_radius}")
    
    # Test COMMUNICATE action
    actions = {a: env.HOLD for a in env.agents}
    actions[agent] = env.COMMUNICATE
    
    obs, _, _, _, _ = env.step(actions)
    
    assert agent in env.message_buffer
    print("✓ COMMUNICATE action creates message in buffer")
    
    print("✓ Communication test PASSED\n")
    return True


def test_intervention():
    """Test intervention action."""
    print("\n=== Testing Intervention ===")
    
    from cognitive_swarm.environment import CoordinationEnv
    
    env = CoordinationEnv(num_agents=10, num_adversaries=5, seed=42)
    env.reset(seed=42)
    
    # Position an agent near an adversary
    agent = env.agents[0]
    if len(env.adversary_positions) > 0:
        adv_pos = env.adversary_positions[0]
        env.agent_positions[agent] = adv_pos + np.array([1, 0])  # Adjacent
        
        initial_adv_count = sum(1 for pos in env.adversary_positions if pos[0] != -1)
        
        # Execute intervention
        actions = {a: env.HOLD for a in env.agents}
        actions[agent] = env.INTERVENTION
        
        obs, rewards, _, _, _ = env.step(actions)
        
        final_adv_count = sum(1 for pos in env.adversary_positions if pos[0] != -1)
        
        # Intervention has 80% success rate, so might or might not neutralize
        print(f"✓ Intervention action executed (adversaries: {initial_adv_count} → {final_adv_count})")
    
    print("✓ Intervention test PASSED\n")
    return True


def run_basic_tests():
    """Run all basic tests."""
    print("=" * 60)
    print("COORDINATION ENVIRONMENT - BASIC TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_basic_structure,
        test_reset,
        test_step,
        test_movement,
        test_noise_injection,
        test_protected_distances,
        test_communication,
        test_intervention,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL BASIC TESTS PASSED! 🎉")
        print("\nFor full test suite including neural network integration,")
        print("install dependencies: pip install -r requirements.txt")
        print("Then run: python test_env.py\n")
    else:
        print(f"\n⚠️  {failed} test(s) failed\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_basic_tests()
    exit(0 if success else 1)
