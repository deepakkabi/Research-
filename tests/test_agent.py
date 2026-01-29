"""
Test Suite for Cognitive Agent Integration

Tests the complete Secure Decision Pipeline and all module integrations.
"""

import torch
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_swarm.agents.cognitive_agent import CognitiveAgent


class TestCognitiveAgent:
    """Test suite for the integrated CognitiveAgent"""
    
    def test_initialization(self):
        """Test agent initializes with correct dimensions"""
        agent = CognitiveAgent(
            obs_dim=10,
            message_dim=8,
            state_dim=6,
            action_dim=7,
            num_roles=3,
            hidden_dim=128,
            use_beliefs=False
        )
        
        # Verify modules exist
        assert agent.trust_gate is not None
        assert agent.hmf_encoder is not None
        assert agent.safety_module is not None
        assert agent.policy_net is not None
        assert agent.value_net is not None
        
        # Verify dimensions
        assert agent.obs_dim == 10
        assert agent.state_dim == 6
        assert agent.action_dim == 7
        assert agent.num_roles == 3
        
        # Verify policy input dimension (V1: no beliefs)
        expected_input_dim = 10 + (3 * 6) + 0  # obs + mean_field + beliefs = 28
        assert agent.get_policy_input_dim() == expected_input_dim
        assert agent.policy_net[0].in_features == expected_input_dim
        
        print("✓ Initialization test passed")
    
    def test_full_pipeline(self):
        """Verify all modules connect correctly in forward pass"""
        agent = CognitiveAgent(
            obs_dim=10,
            message_dim=8,
            state_dim=6,
            action_dim=7,
            num_roles=3,
            use_beliefs=False  # V1: disabled
        )
        
        # Create dummy inputs matching environment format
        batch_size = 4
        num_neighbors = 5
        
        local_obs = torch.randn(batch_size, 10)
        messages = torch.randn(batch_size, num_neighbors, 8)
        neighbor_states = torch.randn(batch_size, num_neighbors, 6)  # state_dim=6!
        neighbor_roles = torch.randint(0, 3, (batch_size, num_neighbors))
        neighbor_mask = torch.ones(batch_size, num_neighbors)
        edge_index = torch.randint(0, num_neighbors, (2, 20))
        neighbor_ids = torch.arange(num_neighbors).unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass through entire pipeline
        action_logits, value, info = agent(
            local_obs=local_obs,
            messages=messages,
            neighbor_states=neighbor_states,
            neighbor_roles=neighbor_roles,
            neighbor_mask=neighbor_mask,
            edge_index=edge_index,
            neighbor_ids=neighbor_ids,
            other_team_obs=None  # No beliefs in V1
        )
        
        # Check output shapes
        assert action_logits.shape == (batch_size, 7), f"Expected (4, 7), got {action_logits.shape}"
        assert value.shape == (batch_size, 1), f"Expected (4, 1), got {value.shape}"
        
        # Check info dict contains all intermediate outputs
        assert 'reliability_weights' in info
        assert 'belief_state' in info
        assert 'mean_field' in info
        assert 'filtered_messages' in info
        assert 'policy_input' in info
        
        # Check intermediate shapes
        assert info['reliability_weights'].shape == (batch_size, num_neighbors)
        assert info['mean_field'].shape == (batch_size, 18), f"Expected (4, 18), got {info['mean_field'].shape}"
        assert info['filtered_messages'].shape == (batch_size, 8)
        assert info['policy_input'].shape == (batch_size, 28)  # V1: 10 + 18 + 0
        
        print("✓ Full pipeline test passed")
    
    def test_safety_integration(self):
        """Verify safety module blocks unsafe actions"""
        agent = CognitiveAgent(
            obs_dim=10,
            message_dim=8,
            state_dim=6,
            action_dim=7,
            use_beliefs=False
        )
        
        # Create scenario where INTERVENTION would violate protected entity constraint
        state_dict = {
            'agent_positions': np.array([[0, 0]]),
            'protected_positions': np.array([[3, 0]]),  # Distance = 3 < 5 (safe_distance)
            'target_positions': np.array([[10, 10]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([5])
        }
        
        # Policy strongly prefers INTERVENTION (action 4)
        action_logits = torch.zeros(1, 7)
        action_logits[0, 4] = 10.0  # High logit for INTERVENTION
        
        # Select action with safety verification
        final_action, blocked, reason = agent.select_action(
            action_logits=action_logits[0],
            state_dict=state_dict,
            agent_id=0,
            deterministic=True
        )
        
        # Safety module should block INTERVENTION and fallback to HOLD
        assert blocked == True, "Safety module should block unsafe action"
        assert final_action == 6, f"Should fallback to HOLD (6), got {final_action}"
        assert reason == "Protected Entity Constraint", f"Wrong reason: {reason}"
        
        print("✓ Safety integration test passed")
    
    def test_safety_allows_safe_actions(self):
        """Verify safety module allows safe actions"""
        agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, use_beliefs=False)
        
        # Create scenario where INTERVENTION is safe
        state_dict = {
            'agent_positions': np.array([[0, 0]]),
            'protected_positions': np.array([[10, 0]]),  # Distance = 10 > 5 (safe)
            'target_positions': np.array([[5, 5]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([20])  # High value, so proportionality OK
        }
        
        # Policy proposes INTERVENTION
        action_logits = torch.zeros(1, 7)
        action_logits[0, 4] = 10.0
        
        final_action, blocked, reason = agent.select_action(
            action_logits[0], state_dict, agent_id=0, deterministic=True
        )
        
        # Should NOT be blocked
        assert blocked == False, f"Safe action should not be blocked: {reason}"
        assert final_action == 4, f"Action should remain INTERVENTION (4), got {final_action}"
        assert reason == "", f"Should have no block reason, got: {reason}"
        
        print("✓ Safety allows safe actions test passed")
    
    def test_ablation_no_beliefs(self):
        """Test agent works correctly without beliefs module (V1 configuration)"""
        agent = CognitiveAgent(
            obs_dim=10,
            message_dim=8,
            state_dim=6,
            action_dim=7,
            use_beliefs=False  # V1: disabled
        )
        
        assert agent.belief_module is None
        assert agent.use_beliefs == False
        
        # Create minimal inputs
        local_obs = torch.randn(2, 10)
        messages = torch.randn(2, 3, 8)
        neighbor_states = torch.randn(2, 3, 6)
        neighbor_roles = torch.randint(0, 3, (2, 3))
        neighbor_mask = torch.ones(2, 3)
        
        # Forward pass should work without other_team_obs
        action_logits, value, info = agent(
            local_obs=local_obs,
            messages=messages,
            neighbor_states=neighbor_states,
            neighbor_roles=neighbor_roles,
            neighbor_mask=neighbor_mask,
            other_team_obs=None  # Not needed when beliefs disabled
        )
        
        assert action_logits.shape == (2, 7)
        assert info['belief_state'].shape == (2, 4)  # Should be zeros
        assert torch.all(info['belief_state'] == 0), "Belief state should be all zeros when disabled"
        
        print("✓ Ablation (no beliefs) test passed")
    
    def test_ablation_disable_trust_gate(self):
        """Test agent with Trust Gate effectively disabled (low threshold)"""
        agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, use_beliefs=False)
        
        # Lower threshold to accept all messages
        agent.trust_gate.threshold = -1.0  # Accept everything
        
        local_obs = torch.randn(2, 10)
        messages = torch.randn(2, 3, 8)
        neighbor_states = torch.randn(2, 3, 6)
        neighbor_roles = torch.randint(0, 3, (2, 3))
        neighbor_mask = torch.ones(2, 3)
        
        action_logits, value, info = agent(
            local_obs, messages, neighbor_states, neighbor_roles, neighbor_mask
        )
        
        # All neighbors should have high reliability weights
        assert action_logits.shape == (2, 7)
        # With low threshold, most weights should be non-zero
        
        print("✓ Ablation (Trust Gate disabled) test passed")
    
    def test_dimensional_consistency(self):
        """Verify all dimensions match across modules"""
        agent = CognitiveAgent(
            obs_dim=10,
            message_dim=8,
            state_dim=6,
            action_dim=7,
            num_roles=3,
            use_beliefs=False
        )
        
        # Verify HMF encoder dimensions
        assert agent.hmf_encoder.state_dim == 6
        assert agent.hmf_encoder.output_dim == 18  # 3 * 6
        assert agent.hmf_encoder.num_roles == 3
        
        # Verify Trust Gate dimension
        assert agent.trust_gate.message_dim == 8
        
        # Verify policy network dimensions
        expected_policy_input = 10 + 18 + 0  # obs + mean_field + beliefs (V1: no beliefs)
        actual_policy_input = agent.policy_net[0].in_features
        assert actual_policy_input == expected_policy_input, \
            f"Policy input mismatch: expected {expected_policy_input}, got {actual_policy_input}"
        
        # Verify action output
        assert agent.policy_net[-1].out_features == 7
        
        # Verify value network
        assert agent.value_net[0].in_features == expected_policy_input
        assert agent.value_net[-1].out_features == 1
        
        print("✓ Dimensional consistency test passed")
    
    def test_batch_processing(self):
        """Test agent handles different batch sizes correctly"""
        agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, use_beliefs=False)
        
        for batch_size in [1, 4, 16]:
            local_obs = torch.randn(batch_size, 10)
            messages = torch.randn(batch_size, 5, 8)
            neighbor_states = torch.randn(batch_size, 5, 6)
            neighbor_roles = torch.randint(0, 3, (batch_size, 5))
            neighbor_mask = torch.ones(batch_size, 5)
            
            action_logits, value, info = agent(
                local_obs, messages, neighbor_states, neighbor_roles, neighbor_mask
            )
            
            assert action_logits.shape == (batch_size, 7)
            assert value.shape == (batch_size, 1)
            assert info['mean_field'].shape == (batch_size, 18)
        
        print("✓ Batch processing test passed")
    
    def test_gradient_flow(self):
        """Test gradients flow through entire pipeline"""
        agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, use_beliefs=False)
        
        # Create inputs with requires_grad
        local_obs = torch.randn(2, 10, requires_grad=True)
        messages = torch.randn(2, 3, 8)
        neighbor_states = torch.randn(2, 3, 6)
        neighbor_roles = torch.randint(0, 3, (2, 3))
        neighbor_mask = torch.ones(2, 3)
        
        action_logits, value, info = agent(
            local_obs, messages, neighbor_states, neighbor_roles, neighbor_mask
        )
        
        # Compute loss and backprop
        loss = action_logits.sum() + value.sum()
        loss.backward()
        
        # Check gradients exist
        assert local_obs.grad is not None
        assert any(p.grad is not None for p in agent.policy_net.parameters())
        assert any(p.grad is not None for p in agent.value_net.parameters())
        
        print("✓ Gradient flow test passed")
    
    def test_deterministic_vs_stochastic_action_selection(self):
        """Test both action selection modes"""
        agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, use_beliefs=False)
        
        state_dict = {
            'agent_positions': np.array([[0, 0]]),
            'protected_positions': np.array([[20, 0]]),
            'target_positions': np.array([[10, 10]]),
            'agent_resources': np.array([100]),
            'target_values': np.array([50])
        }
        
        # Create action logits favoring action 2
        action_logits = torch.tensor([-1.0, 0.0, 5.0, -1.0, -1.0, -1.0, -1.0])
        
        # Deterministic: should always pick action 2
        action_det, _, _ = agent.select_action(action_logits, state_dict, 0, deterministic=True)
        assert action_det == 2
        
        # Stochastic: sample multiple times, should get action 2 most often
        actions = []
        for _ in range(100):
            action_stoch, _, _ = agent.select_action(action_logits, state_dict, 0, deterministic=False)
            actions.append(action_stoch)
        
        # Action 2 should be most common
        assert actions.count(2) > 80  # Should be ~98% given logits, allow some variance
        
        print("✓ Deterministic vs stochastic action selection test passed")


def test_integration_with_mock_environment():
    """Test agent with environment-like data structures"""
    agent = CognitiveAgent(
        obs_dim=10,
        message_dim=8,
        state_dim=6,
        action_dim=7,
        num_roles=3,
        use_beliefs=False
    )
    
    # Simulate environment observation format
    num_agents = 3
    max_neighbors = 10
    
    # Mock collate_observations output
    batched_obs = {
        'local_obs': torch.randn(num_agents, 10),
        'messages': torch.randn(num_agents, max_neighbors, 8),
        'neighbor_states': torch.randn(num_agents, max_neighbors, 6),
        'neighbor_roles': torch.randint(0, 3, (num_agents, max_neighbors)),
        'neighbor_mask': torch.randint(0, 2, (num_agents, max_neighbors)).float(),
        'neighbor_ids': torch.arange(max_neighbors).unsqueeze(0).expand(num_agents, -1),
    }
    
    # Mock edge index (graph connectivity)
    edge_index = torch.randint(0, max_neighbors, (2, 30))
    
    # Forward pass
    action_logits, value, info = agent(
        local_obs=batched_obs['local_obs'],
        messages=batched_obs['messages'],
        neighbor_states=batched_obs['neighbor_states'],
        neighbor_roles=batched_obs['neighbor_roles'],
        neighbor_mask=batched_obs['neighbor_mask'],
        edge_index=edge_index,
        neighbor_ids=batched_obs['neighbor_ids']
    )
    
    assert action_logits.shape == (num_agents, 7)
    assert value.shape == (num_agents, 1)
    
    # Select actions for all agents
    state_dict = {
        'agent_positions': np.random.rand(num_agents, 2) * 50,
        'protected_positions': np.random.rand(2, 2) * 50,
        'target_positions': np.random.rand(5, 2) * 50,
        'agent_resources': np.random.rand(num_agents) * 100,
        'target_values': np.random.rand(5) * 50
    }
    
    actions = []
    blocked_count = 0
    for i in range(num_agents):
        action, blocked, reason = agent.select_action(
            action_logits[i], state_dict, agent_id=i, deterministic=False
        )
        actions.append(action)
        if blocked:
            blocked_count += 1
    
    assert len(actions) == num_agents
    assert all(0 <= a < 7 for a in actions)
    
    print(f"✓ Integration with mock environment test passed (blocked {blocked_count}/{num_agents} actions)")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("COGNITIVE AGENT TEST SUITE")
    print("="*60 + "\n")
    
    test_suite = TestCognitiveAgent()
    
    tests = [
        ("Initialization", test_suite.test_initialization),
        ("Full Pipeline", test_suite.test_full_pipeline),
        ("Safety Integration (blocking)", test_suite.test_safety_integration),
        ("Safety Integration (allowing)", test_suite.test_safety_allows_safe_actions),
        ("Ablation: No Beliefs", test_suite.test_ablation_no_beliefs),
        ("Ablation: Trust Gate Disabled", test_suite.test_ablation_disable_trust_gate),
        ("Dimensional Consistency", test_suite.test_dimensional_consistency),
        ("Batch Processing", test_suite.test_batch_processing),
        ("Gradient Flow", test_suite.test_gradient_flow),
        ("Action Selection Modes", test_suite.test_deterministic_vs_stochastic_action_selection),
        ("Mock Environment Integration", test_integration_with_mock_environment),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            print("-" * 60)
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
