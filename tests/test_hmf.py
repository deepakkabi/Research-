"""
Test Suite for Hierarchical Mean Field Encoder

Validates correctness of mean field aggregation, edge case handling,
and integration with the Cognitive Swarm environment.
"""

import torch
import pytest
from cognitive_swarm.modules.hmf_encoder import HMFEncoder, LearnedHMFEncoder


def test_dimensionality():
    """Verify output is fixed-size regardless of num_neighbors.
    
    This is the core property of Mean Field Theory: compression from O(N) to O(1).
    """
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Test with 10 neighbors
    states1 = torch.randn(1, 10, 6)
    roles1 = torch.randint(0, 3, (1, 10))
    mask1 = torch.ones(1, 10)
    out1 = encoder(states1, roles1, mask1)
    
    # Test with 50 neighbors (5x more)
    states2 = torch.randn(1, 50, 6)
    roles2 = torch.randint(0, 3, (1, 50))
    mask2 = torch.ones(1, 50)
    out2 = encoder(states2, roles2, mask2)
    
    # Test with 100 neighbors (10x more)
    states3 = torch.randn(1, 100, 6)
    roles3 = torch.randint(0, 3, (1, 100))
    mask3 = torch.ones(1, 100)
    out3 = encoder(states3, roles3, mask3)
    
    # All outputs should have same fixed size
    assert out1.shape == out2.shape == out3.shape == (1, 18), \
        f"Expected (1, 18), got {out1.shape}, {out2.shape}, {out3.shape}"
    
    print("✓ Dimensionality test passed: O(N) → O(1) compression verified")


def test_empty_role_group():
    """Verify handling when no neighbors of a certain role exist.
    
    Expected behavior: Mean field for missing roles should be zero vector.
    """
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # All neighbors are Scouts (role=0), no Coordinators or Support
    states = torch.randn(1, 5, 6)
    roles = torch.zeros(1, 5, dtype=torch.long)  # All role 0
    mask = torch.ones(1, 5)
    out = encoder(states, roles, mask)
    
    # Check output structure: [scout_field(0:6), coordinator_field(6:12), support_field(12:18)]
    scout_field = out[0, 0:6]
    coordinator_field = out[0, 6:12]
    support_field = out[0, 12:18]
    
    # Scout field should be non-zero (average of 5 neighbors)
    assert not torch.allclose(scout_field, torch.zeros(6)), \
        "Scout field should be non-zero when scouts exist"
    
    # Coordinator and Support fields should be zero (no neighbors of those roles)
    assert torch.allclose(coordinator_field, torch.zeros(6), atol=1e-6), \
        f"Coordinator field should be zero, got {coordinator_field}"
    assert torch.allclose(support_field, torch.zeros(6), atol=1e-6), \
        f"Support field should be zero, got {support_field}"
    
    print("✓ Empty role group test passed")


def test_trust_weighting():
    """Verify that zero-trust neighbors are excluded from mean field.
    
    Integration test with Trust Gate: untrusted neighbors should not
    contribute to the aggregated mean field.
    """
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Create 3 neighbors, all Scouts
    states = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],   # Neighbor 0: trusted
         [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],  # Neighbor 1: untrusted
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]   # Neighbor 2: trusted
    ])
    roles = torch.zeros(1, 3, dtype=torch.long)  # All Scouts
    mask = torch.ones(1, 3)
    trust = torch.tensor([[1.0, 0.0, 1.0]])  # Middle neighbor is untrusted
    
    out_with_trust = encoder(states, roles, mask, trust)
    
    # Should average only states[0] and states[2], completely ignoring states[1]
    expected = (states[0, 0] + states[0, 2]) / 2.0
    actual = out_with_trust[0, :6]  # Scout field
    
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Expected {expected}, got {actual}"
    
    # Test with all neighbors untrusted (edge case)
    trust_zero = torch.zeros(1, 3)
    out_no_trust = encoder(states, roles, mask, trust_zero)
    # Should produce near-zero output (technically small values due to epsilon)
    assert torch.allclose(out_no_trust[0, :6], torch.zeros(6), atol=1e-4), \
        "All-zero trust should produce near-zero mean field"
    
    print("✓ Trust weighting test passed")


def test_batch_processing():
    """Verify correct batched computation for multiple agents simultaneously."""
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    batch_size = 32
    max_neighbors = 20
    
    states = torch.randn(batch_size, max_neighbors, 6)
    roles = torch.randint(0, 3, (batch_size, max_neighbors))
    mask = torch.ones(batch_size, max_neighbors)
    
    out = encoder(states, roles, mask)
    
    assert out.shape == (batch_size, 18), \
        f"Expected (32, 18), got {out.shape}"
    
    # Verify each batch element is different (not broadcasting error)
    assert not torch.allclose(out[0], out[1]), \
        "Different batch elements should have different mean fields"
    
    print("✓ Batch processing test passed")


def test_no_neighbors():
    """Verify handling when agent has no neighbors at all."""
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Empty neighborhood (all masked out)
    states = torch.randn(1, 10, 6)
    roles = torch.randint(0, 3, (1, 10))
    mask = torch.zeros(1, 10)  # All neighbors masked (invalid)
    
    out = encoder(states, roles, mask)
    
    # Should produce near-zero output (small values due to epsilon in division)
    assert torch.allclose(out, torch.zeros(1, 18), atol=1e-4), \
        "No neighbors should produce near-zero mean field"
    
    print("✓ No neighbors test passed")


def test_role_separation():
    """Verify that mean fields correctly separate neighbors by role."""
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Create neighbors with distinct state values per role
    states = torch.tensor([
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # Scout
         [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],   # Coordinator  
         [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]]   # Support
    ])
    roles = torch.tensor([[0, 1, 2]])  # One of each role
    mask = torch.ones(1, 3)
    
    out = encoder(states, roles, mask)
    
    scout_field = out[0, 0:6]
    coordinator_field = out[0, 6:12]
    support_field = out[0, 12:18]
    
    # Each field should match the corresponding neighbor's state
    assert torch.allclose(scout_field, torch.ones(6), atol=1e-5)
    assert torch.allclose(coordinator_field, torch.ones(6) * 2.0, atol=1e-5)
    assert torch.allclose(support_field, torch.ones(6) * 3.0, atol=1e-5)
    
    print("✓ Role separation test passed")


def test_averaging_correctness():
    """Verify mathematical correctness of averaging operation."""
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Create 4 Scout neighbors with known values
    states = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    ])
    roles = torch.zeros(1, 4, dtype=torch.long)  # All Scouts
    mask = torch.ones(1, 4)
    
    out = encoder(states, roles, mask)
    
    # Expected: average of 4 states = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]
    expected = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
    actual = out[0, :6]
    
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"Expected {expected}, got {actual}"
    
    print("✓ Averaging correctness test passed")


def test_partial_masking():
    """Test with some neighbors masked (representing variable neighborhood sizes)."""
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # 10 potential neighbors, but only first 3 are valid
    states = torch.randn(1, 10, 6)
    roles = torch.randint(0, 3, (1, 10))
    mask = torch.zeros(1, 10)
    mask[0, :3] = 1.0  # Only first 3 are valid
    
    out = encoder(states, roles, mask)
    
    # Should only use first 3 neighbors, ignoring the masked ones
    # Verify by computing manually
    valid_states = states[0, :3]
    valid_roles = roles[0, :3]
    
    # Compute expected manually for first valid role we find
    role_to_check = valid_roles[0].item()
    role_mask_manual = (valid_roles == role_to_check).float()
    if role_mask_manual.sum() > 0:
        expected_for_role = valid_states[role_mask_manual.bool()].mean(dim=0)
        actual_for_role = out[0, role_to_check*6:(role_to_check+1)*6]
        assert torch.allclose(actual_for_role, expected_for_role, atol=1e-5)
    
    print("✓ Partial masking test passed")


def test_hierarchical_mode():
    """Test hierarchical encoder with distance-based grouping."""
    encoder = HMFEncoder(state_dim=6, num_roles=3, use_hierarchical=True)
    
    assert encoder.output_dim == 36, \
        f"Hierarchical mode should have output_dim=36, got {encoder.output_dim}"
    
    states = torch.randn(1, 10, 6)
    roles = torch.randint(0, 3, (1, 10))
    mask = torch.ones(1, 10)
    distances = torch.rand(1, 10) * 10  # Random distances 0-10
    
    out = encoder.forward_hierarchical(states, roles, mask, distances)
    
    assert out.shape == (1, 36), \
        f"Expected (1, 36) for hierarchical output, got {out.shape}"
    
    print("✓ Hierarchical mode test passed")


def test_learned_encoder():
    """Test the learnable variant of HMF Encoder."""
    encoder = LearnedHMFEncoder(state_dim=6, num_roles=3, hidden_dim=32)
    
    states = torch.randn(4, 15, 6)
    roles = torch.randint(0, 3, (4, 15))
    mask = torch.ones(4, 15)
    
    out = encoder(states, roles, mask)
    
    assert out.shape == (4, 18), \
        f"Expected (4, 18), got {out.shape}"
    
    # Verify gradients flow (learnable parameters exist)
    loss = out.sum()
    loss.backward()
    
    # Check that at least one parameter has gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in encoder.parameters())
    assert has_gradients, "Learned encoder should have trainable parameters with gradients"
    
    print("✓ Learned encoder test passed")


def test_integration_with_environment_output():
    """
    Test integration with actual environment output format.
    
    Simulates the output from cognitive_swarm.environment.CoordinationEnv
    after calling collate_observations().
    """
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Simulate environment output (batch of 8 agents)
    batch_size = 8
    max_neighbors = 30
    
    # Environment provides self_state = [x, y, health, resource, role_id, team_id]
    neighbor_states = torch.randn(batch_size, max_neighbors, 6)
    neighbor_roles = torch.randint(0, 3, (batch_size, max_neighbors))
    neighbor_mask = torch.rand(batch_size, max_neighbors) > 0.3  # ~70% valid neighbors
    
    # Simulate trust weights from Trust Gate (values in [0, 1])
    trust_weights = torch.rand(batch_size, max_neighbors)
    
    # Encode
    mean_fields = encoder(
        neighbor_states.float(),
        neighbor_roles,
        neighbor_mask.float(),
        trust_weights
    )
    
    assert mean_fields.shape == (batch_size, 18), \
        f"Expected ({batch_size}, 18), got {mean_fields.shape}"
    
    print("✓ Environment integration test passed")


def test_scalability_verification():
    """
    Verify computational complexity: should be O(N) not O(N²).
    
    Time to encode should scale linearly with number of neighbors.
    """
    import time
    
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    times = []
    neighbor_counts = [10, 50, 100, 200, 500]
    
    for n in neighbor_counts:
        states = torch.randn(1, n, 6)
        roles = torch.randint(0, 3, (1, n))
        mask = torch.ones(1, n)
        
        start = time.time()
        for _ in range(10):  # Average over 10 runs
            _ = encoder(states, roles, mask)
        elapsed = time.time() - start
        times.append(elapsed)
    
    # Check that time scaling is roughly linear (not quadratic)
    # Time for 500 neighbors should be < 50x time for 10 neighbors (linear)
    # If it were O(N²), it would be ~2500x
    ratio = times[-1] / times[0]
    assert ratio < 100, \
        f"Complexity appears super-linear: {neighbor_counts[-1]/neighbor_counts[0]}x " \
        f"neighbors took {ratio}x time (expected <100x for linear)"
    
    print(f"✓ Scalability test passed (500 neighbors took {ratio:.1f}x time vs 10 neighbors)")


# Comparison helper function
def compare_aggregation_methods():
    """
    Compare different aggregation methods for research analysis.
    
    Returns comparison table data for documentation.
    """
    results = {
        "Simple Averaging": {
            "Complexity": "O(N)",
            "Parameters": 0,
            "Pros": "Fast, interpretable, no overfitting risk",
            "Cons": "Loses individual neighbor importance"
        },
        "Learned Aggregation": {
            "Complexity": "O(N)",
            "Parameters": "3 × (6→32→1 MLPs) = ~200",
            "Pros": "Can learn importance weights, more expressive",
            "Cons": "Risk of overfitting, requires more data"
        },
        "Attention-based": {
            "Complexity": "O(N²)",
            "Parameters": "Large (attention mechanism)",
            "Pros": "Most expressive, pair-wise interactions",
            "Cons": "Defeats scalability purpose of Mean Field"
        }
    }
    return results


if __name__ == "__main__":
    """Run all tests and print summary."""
    print("=" * 60)
    print("HMF Encoder Test Suite")
    print("=" * 60)
    
    test_functions = [
        test_dimensionality,
        test_empty_role_group,
        test_trust_weighting,
        test_batch_processing,
        test_no_neighbors,
        test_role_separation,
        test_averaging_correctness,
        test_partial_masking,
        test_hierarchical_mode,
        test_learned_encoder,
        test_integration_with_environment_output,
        test_scalability_verification
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! HMF Encoder is ready for integration.")
        
        # Print comparison table
        print("\n" + "=" * 60)
        print("Aggregation Method Comparison")
        print("=" * 60)
        comparison = compare_aggregation_methods()
        for method, details in comparison.items():
            print(f"\n{method}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
