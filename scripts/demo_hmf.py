"""
Example: Using HMF Encoder with Cognitive Swarm Environment

This example demonstrates how to integrate the Hierarchical Mean Field Encoder
with the CoordinationEnv environment for scalable multi-agent training.
"""

import torch
from cognitive_swarm.modules import HMFEncoder, LearnedHMFEncoder


def example_basic_usage():
    """Basic example: Encode neighbor observations."""
    print("=" * 70)
    print("Example 1: Basic Mean Field Encoding")
    print("=" * 70)
    
    # Initialize encoder
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Simulate environment output
    batch_size = 8  # 8 agents
    max_neighbors = 20  # Up to 20 neighbors per agent
    
    # Neighbor state vectors: [x, y, health, resource, role_id, team_id]
    neighbor_states = torch.randn(batch_size, max_neighbors, 6)
    
    # Role assignments: 0=Scout, 1=Coordinator, 2=Support
    neighbor_roles = torch.randint(0, 3, (batch_size, max_neighbors))
    
    # Valid neighbor mask (some neighbors are padding)
    neighbor_mask = torch.rand(batch_size, max_neighbors) > 0.3  # 70% valid
    
    # Encode to mean fields
    mean_fields = encoder(
        neighbor_states,
        neighbor_roles,
        neighbor_mask.float()
    )
    
    print(f"Input shape: {neighbor_states.shape}")
    print(f"Output shape: {mean_fields.shape}")
    print(f"Compression: {neighbor_states.numel()} → {mean_fields.numel()} parameters")
    print(f"Reduction factor: {neighbor_states.numel() / mean_fields.numel():.1f}x")
    print()


def example_with_trust_weights():
    """Example with trust weights from Trust Gate module."""
    print("=" * 70)
    print("Example 2: Mean Field with Trust Weighting")
    print("=" * 70)
    
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Setup data
    batch_size = 4
    max_neighbors = 10
    
    neighbor_states = torch.randn(batch_size, max_neighbors, 6)
    neighbor_roles = torch.randint(0, 3, (batch_size, max_neighbors))
    neighbor_mask = torch.ones(batch_size, max_neighbors)
    
    # Trust weights from Trust Gate (simulated)
    # Some neighbors have low trust (potentially faulty)
    trust_weights = torch.rand(batch_size, max_neighbors)
    trust_weights[0, 3] = 0.0  # Agent 0's neighbor 3 is completely untrusted
    
    # Encode with trust
    mean_fields = encoder(
        neighbor_states,
        neighbor_roles,
        neighbor_mask,
        trust_weights
    )
    
    print(f"Mean field shape: {mean_fields.shape}")
    print(f"Trust weights applied: {trust_weights[0].tolist()}")
    print("Untrusted neighbors are excluded from aggregation")
    print()


def example_hierarchical_encoding():
    """Example using hierarchical near/far field grouping."""
    print("=" * 70)
    print("Example 3: Hierarchical Near/Far Field Encoding")
    print("=" * 70)
    
    encoder = HMFEncoder(state_dim=6, num_roles=3, use_hierarchical=True)
    
    batch_size = 4
    max_neighbors = 15
    
    neighbor_states = torch.randn(batch_size, max_neighbors, 6)
    neighbor_roles = torch.randint(0, 3, (batch_size, max_neighbors))
    neighbor_mask = torch.ones(batch_size, max_neighbors)
    
    # Compute distances to neighbors (Euclidean from position)
    # Assuming neighbor_states[:, :, :2] contains [x, y] positions
    ego_position = torch.zeros(batch_size, 2)  # Ego at origin
    neighbor_positions = neighbor_states[:, :, :2]
    distances = torch.norm(neighbor_positions - ego_position.unsqueeze(1), dim=-1)
    
    # Hierarchical encoding (groups by role AND distance)
    mean_fields = encoder.forward_hierarchical(
        neighbor_states,
        neighbor_roles,
        neighbor_mask,
        distances
    )
    
    print(f"Output shape: {mean_fields.shape}")
    print(f"Structure: 3 roles × 2 distances × 6 state_dim = 36")
    print("Mean fields: [Scout_near, Scout_far, Coord_near, Coord_far, Support_near, Support_far]")
    print()


def example_learned_aggregation():
    """Example using learned attention-based aggregation."""
    print("=" * 70)
    print("Example 4: Learned Attention-Based Aggregation")
    print("=" * 70)
    
    # Use learned encoder instead of simple averaging
    encoder = LearnedHMFEncoder(state_dim=6, num_roles=3, hidden_dim=32)
    
    batch_size = 4
    max_neighbors = 12
    
    neighbor_states = torch.randn(batch_size, max_neighbors, 6)
    neighbor_roles = torch.randint(0, 3, (batch_size, max_neighbors))
    neighbor_mask = torch.ones(batch_size, max_neighbors)
    
    # Forward pass
    mean_fields = encoder(neighbor_states, neighbor_roles, neighbor_mask)
    
    print(f"Output shape: {mean_fields.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters())}")
    print("Uses learned MLPs to compute attention weights for each neighbor")
    print()


def example_policy_integration():
    """Example: Integrating mean fields with a policy network."""
    print("=" * 70)
    print("Example 5: Integration with Policy Network")
    print("=" * 70)
    
    import torch.nn as nn
    
    # Policy network that uses mean field encoding
    class MeanFieldPolicy(nn.Module):
        def __init__(self, local_obs_dim=10, mean_field_dim=18, action_dim=5):
            super().__init__()
            self.mean_field_encoder = HMFEncoder(state_dim=6, num_roles=3)
            
            # Policy network
            input_dim = local_obs_dim + mean_field_dim  # 10 + 18 = 28
            self.policy = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
        
        def forward(self, local_obs, neighbor_states, neighbor_roles, neighbor_mask):
            # Encode neighbors to mean field
            mean_field = self.mean_field_encoder(
                neighbor_states, neighbor_roles, neighbor_mask
            )
            
            # Concatenate local observation with mean field
            policy_input = torch.cat([local_obs, mean_field], dim=-1)
            
            # Compute action logits
            action_logits = self.policy(policy_input)
            return action_logits
    
    # Initialize policy
    policy = MeanFieldPolicy()
    
    # Simulate inputs
    batch_size = 16
    local_obs = torch.randn(batch_size, 10)  # Local observation
    neighbor_states = torch.randn(batch_size, 25, 6)
    neighbor_roles = torch.randint(0, 3, (batch_size, 25))
    neighbor_mask = torch.ones(batch_size, 25)
    
    # Forward pass
    actions = policy(local_obs, neighbor_states, neighbor_roles, neighbor_mask)
    
    print(f"Local obs shape: {local_obs.shape}")
    print(f"Mean field shape: (16, 18)")
    print(f"Policy input shape: (16, 28)")
    print(f"Action logits shape: {actions.shape}")
    print("✓ Mean field successfully integrated into policy network")
    print()


def example_scalability_comparison():
    """Demonstrate scalability advantage of mean field."""
    print("=" * 70)
    print("Example 6: Scalability Comparison")
    print("=" * 70)
    
    encoder = HMFEncoder(state_dim=6, num_roles=3)
    
    # Test with different team sizes
    team_sizes = [10, 50, 100, 200]
    
    print("Observation dimensionality comparison:")
    print(f"{'Team Size':<12} {'Individual (O(N))':<20} {'Mean Field (O(1))':<20} {'Reduction':<12}")
    print("-" * 65)
    
    for size in team_sizes:
        # Individual processing: each agent observes all others
        individual_dim = (size - 1) * 6  # (N-1) neighbors × 6 state_dim
        
        # Mean field: fixed size regardless of team size
        mean_field_dim = 18  # 3 roles × 6 state_dim
        
        reduction = individual_dim / mean_field_dim
        
        print(f"{size:<12} {individual_dim:<20} {mean_field_dim:<20} {reduction:.1f}x")
    
    print("\nConclusion: Mean field maintains O(1) dimensionality as team scales!")
    print()


if __name__ == "__main__":
    """Run all examples."""
    
    print("\n" + "=" * 70)
    print("COGNITIVE SWARM - HMF ENCODER USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    try:
        example_basic_usage()
        example_with_trust_weights()
        example_hierarchical_encoding()
        example_learned_aggregation()
        example_policy_integration()
        example_scalability_comparison()
        
        print("=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
