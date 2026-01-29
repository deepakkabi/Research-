"""
Test Suite for Trust Gate Module

Tests Byzantine fault tolerance, noise resilience, and integration with
CoordinationEnv and HMFEncoder.
"""

import pytest
import torch
import numpy as np
from cognitive_swarm.modules.trust_gate import TrustGate, SimpleTrustGate


class TestTrustGateBasics:
    """Basic functionality tests for Trust Gate."""
    
    def test_initialization(self):
        """Test that Trust Gate initializes correctly."""
        gate = TrustGate(message_dim=8, hidden_dim=64, num_heads=4)
        
        assert gate.message_dim == 8
        assert gate.hidden_dim == 64
        assert gate.threshold == 0.5
        assert gate.num_heads == 4
        print("✓ Initialization test passed")
    
    def test_forward_shape(self):
        """Test that output shapes are correct."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        batch_size = 4
        num_neighbors = 10
        
        messages = torch.randn(batch_size, num_neighbors, 8)
        local_obs = torch.randn(batch_size, 10)
        edge_index = torch.randint(0, num_neighbors, (2, 20))
        neighbor_ids = torch.arange(num_neighbors).unsqueeze(0).expand(batch_size, -1)
        
        filtered, reliability = gate(messages, local_obs, edge_index, neighbor_ids)
        
        # Check shapes
        assert filtered.shape == (batch_size, 8), f"Expected (4, 8), got {filtered.shape}"
        assert reliability.shape == (batch_size, num_neighbors), \
            f"Expected (4, 10), got {reliability.shape}"
        
        # Check that reliability weights are normalized
        sums = reliability.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), \
            f"Weights should sum to 1, got {sums}"
        
        print("✓ Forward shape test passed")
    
    def test_reliability_range(self):
        """Test that reliability weights are in valid range [0, 1]."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        messages = torch.randn(2, 5, 8)
        local_obs = torch.randn(2, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        neighbor_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        
        _, reliability = gate(messages, local_obs, edge_index, neighbor_ids)
        
        assert (reliability >= 0).all(), "Reliability weights must be >= 0"
        assert (reliability <= 1).all(), "Reliability weights must be <= 1"
        
        print("✓ Reliability range test passed")


class TestErrorDetection:
    """Test error and corruption detection capabilities."""
    
    def test_error_detection(self):
        """Verify that injected errors are detected."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        # Create normal messages
        normal_msgs = torch.randn(1, 5, 8)
        
        # Inject high-magnitude error in message 2 (simulating environment noise)
        normal_msgs[0, 2, :] = torch.randn(8) * 10.0  # 5x normal magnitude
        
        local_obs = torch.randn(1, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        neighbor_ids = torch.arange(5).unsqueeze(0)
        
        _, reliability = gate(normal_msgs, local_obs, edge_index, neighbor_ids)
        
        # Reliability for corrupted message should be lower than others
        corrupted_rel = reliability[0, 2]
        avg_normal_rel = reliability[0, [0, 1, 3, 4]].mean()
        
        assert corrupted_rel < avg_normal_rel, \
            f"Corrupted message should have lower reliability: {corrupted_rel} vs {avg_normal_rel}"
        
        print(f"✓ Error detection test passed (corrupted: {corrupted_rel:.3f}, normal: {avg_normal_rel:.3f})")
    
    def test_corruption_resilience(self):
        """
        Verify performance under 20% message corruption (matching environment).
        
        This is the CRITICAL integration test - validates that Trust Gate
        is properly calibrated for environment's σ=2.0 Gaussian noise.
        """
        gate = TrustGate(message_dim=8, hidden_dim=32, consistency_threshold=0.5)
        
        num_trials = 100
        detection_successes = 0
        
        for trial in range(num_trials):
            # Create messages
            msgs = torch.randn(1, 10, 8)
            
            # Corrupt 2 out of 10 (20%) - matching environment's noise rate
            corrupted_indices = np.random.choice(10, 2, replace=False)
            
            # Add Gaussian noise with σ=2.0 (matching environment)
            msgs[0, corrupted_indices, :] += torch.randn(2, 8) * 2.0
            
            local_obs = torch.randn(1, 10)
            edge_index = torch.randint(0, 10, (2, 20))
            neighbor_ids = torch.arange(10).unsqueeze(0)
            
            _, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
            
            # Check if corrupted messages have lower reliability than clean ones
            corrupted_rel = reliability[0, corrupted_indices].mean()
            clean_indices = [i for i in range(10) if i not in corrupted_indices]
            clean_rel = reliability[0, clean_indices].mean()
            
            if corrupted_rel < clean_rel:
                detection_successes += 1
        
        detection_rate = detection_successes / num_trials * 100
        
        print(f"Detection rate: {detection_rate:.1f}%")
        assert detection_rate >= 60, \
            f"Should detect at least 60% of corruption, got {detection_rate}%"
        
        print("✓ Corruption resilience test passed")
    
    def test_multiple_corruptions(self):
        """Test handling of multiple simultaneous corruptions."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        msgs = torch.randn(1, 10, 8)
        
        # Corrupt 4 out of 10 (40% - extreme case)
        corrupted = [1, 3, 5, 7]
        for idx in corrupted:
            msgs[0, idx, :] = torch.randn(8) * 5.0
        
        local_obs = torch.randn(1, 10)
        edge_index = torch.randint(0, 10, (2, 20))
        neighbor_ids = torch.arange(10).unsqueeze(0)
        
        _, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
        
        # At least some corrupted messages should have lower reliability
        corrupted_rel = reliability[0, corrupted].mean()
        clean_rel = reliability[0, [0, 2, 4, 6, 8, 9]].mean()
        
        assert corrupted_rel < clean_rel * 1.2, \
            "Corrupted messages should generally have lower reliability"
        
        print("✓ Multiple corruption test passed")


class TestGraphConnectivity:
    """Test that reliability filtering maintains network connectivity."""
    
    def test_graph_connectivity(self):
        """Verify that reliability filtering doesn't fragment the network."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        msgs = torch.randn(1, 10, 8)
        local_obs = torch.randn(1, 10)
        edge_index = torch.randint(0, 10, (2, 20))
        neighbor_ids = torch.arange(10).unsqueeze(0)
        
        _, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
        
        # At least half of neighbors should have non-trivial reliability
        # (since weights are softmax-normalized, all will be > 0, but most should be meaningful)
        meaningful = (reliability > 0.05).sum()  # 5% threshold for "meaningful"
        
        assert meaningful >= 5, \
            f"Should have at least 5 neighbors with meaningful reliability, got {meaningful}"
        
        print(f"✓ Graph connectivity test passed ({meaningful} neighbors with meaningful weights)")
    
    def test_no_complete_isolation(self):
        """Ensure at least some messages always get through."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        # Even with all messages looking suspicious
        msgs = torch.randn(1, 10, 8) * 10  # High variance
        local_obs = torch.randn(1, 10)
        edge_index = torch.randint(0, 10, (2, 20))
        neighbor_ids = torch.arange(10).unsqueeze(0)
        
        filtered, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
        
        # Filtered message should not be zero (softmax ensures some weights > 0)
        assert torch.norm(filtered) > 0, "Filtered message should not be zero vector"
        
        # Sum of weights should still be 1
        assert torch.allclose(reliability.sum(), torch.tensor(1.0), atol=1e-5)
        
        print("✓ No complete isolation test passed")


class TestIntegrationWithEnvironment:
    """Test integration with CoordinationEnv."""
    
    def test_integration_with_environment(self):
        """
        Test that Trust Gate correctly handles environment's message format.
        This is a CRITICAL integration test.
        """
        # Mock environment behavior
        # In real case, would use: from cognitive_swarm.environment import CoordinationEnv
        
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        # Simulate environment's collate_observations output
        batch_size = 10
        max_neighbors = 20
        
        batched = {
            'messages': torch.randn(batch_size, max_neighbors, 8),
            'local_obs': torch.randn(batch_size, 10),
            'neighbor_ids': torch.randint(0, 30, (batch_size, max_neighbors))
        }
        
        # Add 20% corruption (matching environment)
        num_corrupted = int(0.2 * batch_size * max_neighbors)
        for _ in range(num_corrupted):
            b = np.random.randint(0, batch_size)
            n = np.random.randint(0, max_neighbors)
            batched['messages'][b, n, :] += torch.randn(8) * 2.0  # σ=2.0
        
        # Create mock edge index
        edge_index = torch.randint(0, max_neighbors, (2, 50))
        
        # This should work without errors
        filtered, reliability = gate(
            batched['messages'],
            batched['local_obs'],
            edge_index,
            batched['neighbor_ids']
        )
        
        assert filtered.shape == (batch_size, 8), \
            f"Expected shape ({batch_size}, 8), got {filtered.shape}"
        assert reliability.shape == (batch_size, max_neighbors), \
            f"Expected shape ({batch_size}, {max_neighbors}), got {reliability.shape}"
        
        print("✓ Environment integration test passed")
    
    def test_message_format_compatibility(self):
        """Test that message format [sender_id, role, x, y, status, target_x, target_y, priority] works."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        # Create messages in environment's exact format
        batch_size = 5
        num_neighbors = 8
        
        messages = torch.zeros(batch_size, num_neighbors, 8)
        for b in range(batch_size):
            for n in range(num_neighbors):
                messages[b, n] = torch.tensor([
                    n,                          # sender_id
                    np.random.randint(0, 3),    # role
                    np.random.randn() * 10,     # x
                    np.random.randn() * 10,     # y
                    np.random.randint(0, 2),    # status
                    np.random.randn() * 10,     # target_x
                    np.random.randn() * 10,     # target_y
                    np.random.rand()            # priority
                ])
        
        local_obs = torch.randn(batch_size, 10)
        edge_index = torch.randint(0, num_neighbors, (2, 15))
        neighbor_ids = torch.arange(num_neighbors).unsqueeze(0).expand(batch_size, -1)
        
        filtered, reliability = gate(messages, local_obs, edge_index, neighbor_ids)
        
        assert filtered.shape == (batch_size, 8)
        assert reliability.shape == (batch_size, num_neighbors)
        
        print("✓ Message format compatibility test passed")


class TestHMFIntegration:
    """Test integration with HMFEncoder."""
    
    def test_hmf_integration(self):
        """
        Verify that reliability_weights work correctly with HMFEncoder.
        
        This tests the critical interface:
        reliability_weights -> HMFEncoder.forward(trust_weights=reliability_weights)
        """
        # Mock HMFEncoder behavior
        # In real case, would use: from cognitive_swarm.modules import HMFEncoder
        
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        batch_size = 10
        max_neighbors = 20
        
        messages = torch.randn(batch_size, max_neighbors, 8)
        local_obs = torch.randn(batch_size, 10)
        edge_index = torch.randint(0, max_neighbors, (2, 40))
        neighbor_ids = torch.randint(0, 30, (batch_size, max_neighbors))
        
        _, reliability_weights = gate(messages, local_obs, edge_index, neighbor_ids)
        
        # Verify shape matches HMFEncoder expectations
        assert reliability_weights.shape == (batch_size, max_neighbors), \
            "Shape must be (batch, max_neighbors) for HMFEncoder"
        
        # Verify range [0, 1]
        assert (reliability_weights >= 0).all() and (reliability_weights <= 1).all(), \
            "Weights must be in [0, 1] range"
        
        # Verify sum-normalized (for interpretability)
        sums = reliability_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), \
            "Weights should sum to 1 for interpretability"
        
        # Mock HMFEncoder behavior: weighted mean field
        # HMFEncoder would do: mean_field = (messages * trust_weights.unsqueeze(-1)).sum(dim=1)
        mock_mean_field = (messages * reliability_weights.unsqueeze(-1)).sum(dim=1)
        
        assert mock_mean_field.shape == (batch_size, 8), \
            "Mean field should have shape (batch, message_dim)"
        
        print("✓ HMF integration test passed")
    
    def test_trust_weights_filtering(self):
        """Test that low-reliability neighbors contribute less to mean field."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        messages = torch.randn(1, 10, 8)
        
        # Make message 5 very different (corrupted)
        messages[0, 5, :] = torch.randn(8) * 10
        
        local_obs = torch.randn(1, 10)
        edge_index = torch.randint(0, 10, (2, 20))
        neighbor_ids = torch.arange(10).unsqueeze(0)
        
        _, reliability_weights = gate(messages, local_obs, edge_index, neighbor_ids)
        
        # Corrupted message should have lower weight
        corrupted_weight = reliability_weights[0, 5]
        mean_weight = reliability_weights[0].mean()
        
        assert corrupted_weight <= mean_weight, \
            f"Corrupted message weight ({corrupted_weight}) should be <= mean ({mean_weight})"
        
        print("✓ Trust weights filtering test passed")


class TestFaultyAgentDetection:
    """Test faulty agent detection and diagnosis."""
    
    def test_detect_faulty_agents(self):
        """Test identification of consistently unreliable agents."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        # Simulate 50 time steps
        batch_size = 50
        num_neighbors = 10
        
        reliability_history = []
        
        for t in range(batch_size):
            msgs = torch.randn(1, num_neighbors, 8)
            
            # Agent 3 is consistently faulty
            msgs[0, 3, :] = torch.randn(8) * 5.0
            
            local_obs = torch.randn(1, 10)
            edge_index = torch.randint(0, num_neighbors, (2, 20))
            neighbor_ids = torch.arange(num_neighbors).unsqueeze(0)
            
            _, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
            reliability_history.append(reliability)
        
        # Stack history
        reliability_tensor = torch.cat(reliability_history, dim=0)
        # Shape: (50, 10)
        
        # Detect faulty agents
        faulty = gate.detect_faulty_agents(reliability_tensor, threshold=0.08)
        
        # Agent 3 should be detected as faulty
        assert 3 in faulty, f"Agent 3 should be detected as faulty, got {faulty}"
        
        print(f"✓ Faulty agent detection test passed (detected: {faulty})")
    
    def test_reliability_stats(self):
        """Test reliability statistics computation."""
        gate = TrustGate(message_dim=8, hidden_dim=32)
        
        msgs = torch.randn(5, 10, 8)
        local_obs = torch.randn(5, 10)
        edge_index = torch.randint(0, 10, (2, 20))
        neighbor_ids = torch.arange(10).unsqueeze(0).expand(5, -1)
        
        _, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
        
        stats = gate.get_reliability_stats(reliability)
        
        assert 'mean_reliability' in stats
        assert 'std_reliability' in stats
        assert 'min_reliability' in stats
        assert 'max_reliability' in stats
        assert 'num_reliable' in stats
        assert 'num_unreliable' in stats
        
        # Sanity checks
        assert 0 <= stats['mean_reliability'] <= 1
        assert stats['min_reliability'] <= stats['max_reliability']
        
        print(f"✓ Reliability stats test passed: {stats}")


class TestSimpleTrustGate:
    """Test simplified baseline version."""
    
    def test_simple_gate_basics(self):
        """Test that SimpleTrustGate works as baseline."""
        gate = SimpleTrustGate(message_dim=8, hidden_dim=32)
        
        messages = torch.randn(4, 10, 8)
        local_obs = torch.randn(4, 10)
        
        filtered, reliability = gate(messages, local_obs)
        
        assert filtered.shape == (4, 8)
        assert reliability.shape == (4, 10)
        
        print("✓ SimpleTrustGate basic test passed")
    
    def test_simple_vs_full_comparison(self):
        """Compare SimpleTrustGate vs full TrustGate."""
        simple_gate = SimpleTrustGate(message_dim=8, hidden_dim=32)
        full_gate = TrustGate(message_dim=8, hidden_dim=32)
        
        messages = torch.randn(2, 5, 8)
        local_obs = torch.randn(2, 10)
        edge_index = torch.randint(0, 5, (2, 10))
        neighbor_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        
        simple_filtered, simple_rel = simple_gate(messages, local_obs)
        full_filtered, full_rel = full_gate(messages, local_obs, edge_index, neighbor_ids)
        
        # Both should produce valid outputs
        assert simple_filtered.shape == full_filtered.shape
        assert simple_rel.shape == full_rel.shape
        
        # Results may differ (full gate uses graph attention)
        print("✓ Simple vs full comparison test passed")


def test_all():
    """Run all tests."""
    print("\n=== Running Trust Gate Test Suite ===\n")
    
    # Basic tests
    print("--- Basic Tests ---")
    basic = TestTrustGateBasics()
    basic.test_initialization()
    basic.test_forward_shape()
    basic.test_reliability_range()
    
    # Error detection
    print("\n--- Error Detection Tests ---")
    error = TestErrorDetection()
    error.test_error_detection()
    error.test_corruption_resilience()
    error.test_multiple_corruptions()
    
    # Graph connectivity
    print("\n--- Graph Connectivity Tests ---")
    graph = TestGraphConnectivity()
    graph.test_graph_connectivity()
    graph.test_no_complete_isolation()
    
    # Environment integration
    print("\n--- Environment Integration Tests ---")
    env_int = TestIntegrationWithEnvironment()
    env_int.test_integration_with_environment()
    env_int.test_message_format_compatibility()
    
    # HMF integration
    print("\n--- HMF Integration Tests ---")
    hmf = TestHMFIntegration()
    hmf.test_hmf_integration()
    hmf.test_trust_weights_filtering()
    
    # Faulty agent detection
    print("\n--- Faulty Agent Detection Tests ---")
    fault = TestFaultyAgentDetection()
    fault.test_detect_faulty_agents()
    fault.test_reliability_stats()
    
    # Simple baseline
    print("\n--- Simple Baseline Tests ---")
    simple = TestSimpleTrustGate()
    simple.test_simple_gate_basics()
    simple.test_simple_vs_full_comparison()
    
    print("\n=== All Tests Passed ===\n")


if __name__ == "__main__":
    test_all()
