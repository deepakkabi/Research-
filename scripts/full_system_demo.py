"""
Full System Demonstration - Cognitive Swarm Framework

This script demonstrates the complete Secure Decision Pipeline in action,
showing how all modules integrate to enable scalable, robust, and safe
multi-agent coordination.

Demonstrates:
1. Trust Gate filtering corrupted messages
2. Mean Field aggregation for scalability  
3. Policy network decision making
4. Safety Shield constraint verification
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_swarm import CognitiveAgent


def create_mock_environment_state(num_agents=5, grid_size=50, num_targets=3, num_protected=2):
    """
    Create a mock environment state for demonstration.
    
    Simulates the output of CoordinationEnv.
    """
    state_dict = {
        'agent_positions': np.random.rand(num_agents, 2) * grid_size,
        'protected_positions': np.random.rand(num_protected, 2) * grid_size,
        'target_positions': np.random.rand(num_targets, 2) * grid_size,
        'agent_resources': np.random.rand(num_agents) * 100,
        'target_values': np.random.rand(num_targets) * 50
    }
    return state_dict


def create_mock_observations(num_agents=5, max_neighbors=8, inject_noise=True):
    """
    Create mock observations simulating environment output.
    
    Args:
        num_agents: Number of agents in the system
        max_neighbors: Maximum neighbors per agent
        inject_noise: Whether to inject corrupted messages (simulates 20% noise)
    """
    # Local observations for each agent
    local_obs = torch.randn(num_agents, 10)
    
    # Messages from neighbors (some may be corrupted)
    messages = torch.randn(num_agents, max_neighbors, 8)
    if inject_noise:
        # Inject Gaussian noise into 20% of messages
        noise_mask = torch.rand(num_agents, max_neighbors) < 0.2
        messages[noise_mask] += torch.randn_like(messages[noise_mask]) * 2.0
    
    # Neighbor states: [x, y, health, resource, role_id, team_id]
    neighbor_states = torch.randn(num_agents, max_neighbors, 6)
    
    # Neighbor roles: Scout(0), Coordinator(1), Support(2)
    neighbor_roles = torch.randint(0, 3, (num_agents, max_neighbors))
    
    # Neighbor mask (which neighbors are valid)
    num_valid = torch.randint(3, max_neighbors, (num_agents,))
    neighbor_mask = torch.zeros(num_agents, max_neighbors)
    for i in range(num_agents):
        neighbor_mask[i, :num_valid[i]] = 1.0
    
    # Neighbor IDs
    neighbor_ids = torch.arange(max_neighbors).unsqueeze(0).expand(num_agents, -1)
    
    # Graph structure (edge index)
    edges = []
    for i in range(num_agents):
        for j in range(num_valid[i]):
            edges.append([i, j])
    edge_index = torch.tensor(edges).T if edges else torch.zeros(2, 0, dtype=torch.long)
    
    return {
        'local_obs': local_obs,
        'messages': messages,
        'neighbor_states': neighbor_states,
        'neighbor_roles': neighbor_roles,
        'neighbor_mask': neighbor_mask,
        'neighbor_ids': neighbor_ids,
        'edge_index': edge_index
    }


def run_single_step_demo():
    """
    Demonstrate a single step of the Secure Decision Pipeline.
    """
    print("\n" + "="*70)
    print("SINGLE STEP DEMONSTRATION - Secure Decision Pipeline")
    print("="*70 + "\n")
    
    # Initialize agent
    print("Initializing CognitiveAgent...")
    agent = CognitiveAgent(
        obs_dim=10,
        message_dim=8,
        state_dim=6,
        action_dim=7,
        num_roles=3,
        hidden_dim=128,
        use_beliefs=False  # V1 configuration
    )
    print(f"✓ Agent initialized with policy_input_dim={agent.get_policy_input_dim()}\n")
    
    # Create mock environment
    num_agents = 5
    print(f"Creating mock environment state ({num_agents} agents)...")
    state_dict = create_mock_environment_state(num_agents=num_agents)
    observations = create_mock_observations(num_agents=num_agents, inject_noise=True)
    print("✓ Environment created with 20% message noise injected\n")
    
    # STAGE 1-5: Forward pass through pipeline
    print("EXECUTING SECURE DECISION PIPELINE:")
    print("-" * 70)
    
    print("\n[Stage 1] OBSERVE: Receiving raw observations...")
    print(f"  - Local observations: {observations['local_obs'].shape}")
    print(f"  - Messages: {observations['messages'].shape}")
    print(f"  - Neighbor states: {observations['neighbor_states'].shape}")
    
    print("\n[Stage 2] ORIENT (Security): Trust Gate filtering messages...")
    print("[Stage 3] ORIENT (Context): Bayesian Beliefs - SKIPPED (V1)")
    print("[Stage 4] DECIDE (Scale): Mean Field aggregation...")
    print("[Stage 5] ACT (Policy): Generating action logits...")
    
    with torch.no_grad():
        action_logits, value, info = agent(
            local_obs=observations['local_obs'],
            messages=observations['messages'],
            neighbor_states=observations['neighbor_states'],
            neighbor_roles=observations['neighbor_roles'],
            neighbor_mask=observations['neighbor_mask'],
            edge_index=observations['edge_index'],
            neighbor_ids=observations['neighbor_ids']
        )
    
    print("\n✓ Pipeline execution complete!")
    print(f"\nIntermediate Outputs:")
    print(f"  - Reliability weights: {info['reliability_weights'].shape}")
    print(f"    Mean reliability: {info['reliability_weights'].mean():.3f}")
    print(f"    Min reliability: {info['reliability_weights'].min():.3f}")
    print(f"  - Mean field: {info['mean_field'].shape}")
    print(f"  - Filtered messages: {info['filtered_messages'].shape}")
    
    print(f"\nFinal Outputs:")
    print(f"  - Action logits: {action_logits.shape}")
    print(f"  - Value estimates: {value.shape}")
    print(f"    Mean value: {value.mean():.3f}")
    
    # STAGE 6: Safety verification
    print("\n[Stage 6] ACT (Safety): Verifying actions against constraints...")
    print("-" * 70)
    
    actions = []
    blocked_count = 0
    block_reasons = {}
    
    for agent_id in range(num_agents):
        final_action, violated, reason = agent.select_action(
            action_logits=action_logits[agent_id],
            state_dict=state_dict,
            agent_id=agent_id,
            deterministic=True
        )
        actions.append(final_action)
        
        if violated:
            blocked_count += 1
            block_reasons[reason] = block_reasons.get(reason, 0) + 1
            print(f"  Agent {agent_id}: Action BLOCKED - {reason}")
        else:
            print(f"  Agent {agent_id}: Action {final_action} APPROVED")
    
    print(f"\n✓ Safety verification complete!")
    print(f"  - Total actions: {num_agents}")
    print(f"  - Blocked actions: {blocked_count} ({blocked_count/num_agents*100:.1f}%)")
    if block_reasons:
        print(f"  - Block reasons: {block_reasons}")
    
    print("\n" + "="*70)
    print("SINGLE STEP DEMONSTRATION COMPLETE")
    print("="*70 + "\n")
    
    return actions, info


def run_multi_step_episode(num_steps=10):
    """
    Demonstrate a multi-step episode showing the pipeline in action.
    """
    print("\n" + "="*70)
    print(f"MULTI-STEP EPISODE - {num_steps} Steps")
    print("="*70 + "\n")
    
    # Initialize
    num_agents = 8
    agent = CognitiveAgent(
        obs_dim=10, message_dim=8, state_dim=6, action_dim=7,
        num_roles=3, use_beliefs=False
    )
    
    print(f"Running episode with {num_agents} agents for {num_steps} steps...\n")
    
    # Statistics
    total_blocked = 0
    total_actions = 0
    block_reasons_total = {}
    mean_reliability_per_step = []
    mean_value_per_step = []
    
    for step in range(num_steps):
        # Create fresh observations each step
        state_dict = create_mock_environment_state(num_agents=num_agents)
        observations = create_mock_observations(num_agents=num_agents, inject_noise=True)
        
        # Forward pass
        with torch.no_grad():
            action_logits, value, info = agent(
                local_obs=observations['local_obs'],
                messages=observations['messages'],
                neighbor_states=observations['neighbor_states'],
                neighbor_roles=observations['neighbor_roles'],
                neighbor_mask=observations['neighbor_mask'],
                edge_index=observations['edge_index'],
                neighbor_ids=observations['neighbor_ids']
            )
        
        # Action selection with safety
        step_blocked = 0
        for agent_id in range(num_agents):
            final_action, violated, reason = agent.select_action(
                action_logits[agent_id], state_dict, agent_id, deterministic=False
            )
            total_actions += 1
            if violated:
                step_blocked += 1
                total_blocked += 1
                block_reasons_total[reason] = block_reasons_total.get(reason, 0) + 1
        
        # Record statistics
        mean_reliability_per_step.append(info['reliability_weights'].mean().item())
        mean_value_per_step.append(value.mean().item())
        
        print(f"Step {step+1:2d}: "
              f"Blocked={step_blocked}/{num_agents}, "
              f"MeanReliability={info['reliability_weights'].mean():.3f}, "
              f"MeanValue={value.mean():.3f}")
    
    # Summary statistics
    print("\n" + "-"*70)
    print("EPISODE SUMMARY:")
    print("-"*70)
    print(f"Total actions: {total_actions}")
    print(f"Total blocked: {total_blocked} ({total_blocked/total_actions*100:.1f}%)")
    print(f"Block reasons: {block_reasons_total}")
    print(f"Mean reliability across episode: {np.mean(mean_reliability_per_step):.3f}")
    print(f"Mean value across episode: {np.mean(mean_value_per_step):.3f}")
    
    print("\n" + "="*70)
    print("MULTI-STEP EPISODE COMPLETE")
    print("="*70 + "\n")


def demonstrate_ablation_studies():
    """
    Demonstrate ablation studies showing the impact of each module.
    """
    print("\n" + "="*70)
    print("ABLATION STUDIES - Component Impact Analysis")
    print("="*70 + "\n")
    
    num_agents = 5
    state_dict = create_mock_environment_state(num_agents=num_agents)
    
    # Test configurations
    configs = [
        ("Full Pipeline", {'threshold': 0.5, 'safe_distance': 5.0}),
        ("No Trust Filter", {'threshold': -1.0, 'safe_distance': 5.0}),  # Accept all messages
        ("No Safety Shield", {'threshold': 0.5, 'safe_distance': 0.0}),  # No distance constraint
    ]
    
    results = {}
    
    for config_name, settings in configs:
        print(f"\nConfiguration: {config_name}")
        print("-" * 70)
        
        agent = CognitiveAgent(obs_dim=10, message_dim=8, action_dim=7, use_beliefs=False)
        agent.trust_gate.threshold = settings['threshold']
        agent.safety_module.safe_distance = settings['safe_distance']
        
        observations = create_mock_observations(num_agents=num_agents, inject_noise=True)
        
        with torch.no_grad():
            action_logits, value, info = agent(
                local_obs=observations['local_obs'],
                messages=observations['messages'],
                neighbor_states=observations['neighbor_states'],
                neighbor_roles=observations['neighbor_roles'],
                neighbor_mask=observations['neighbor_mask'],
                edge_index=observations['edge_index'],
                neighbor_ids=observations['neighbor_ids']
            )
        
        # Count blocked actions
        blocked = 0
        for agent_id in range(num_agents):
            _, violated, _ = agent.select_action(
                action_logits[agent_id], state_dict, agent_id, deterministic=True
            )
            if violated:
                blocked += 1
        
        mean_reliability = info['reliability_weights'].mean().item()
        mean_value = value.mean().item()
        
        results[config_name] = {
            'blocked': blocked,
            'reliability': mean_reliability,
            'value': mean_value
        }
        
        print(f"  Blocked actions: {blocked}/{num_agents}")
        print(f"  Mean reliability: {mean_reliability:.3f}")
        print(f"  Mean value: {mean_value:.3f}")
    
    print("\n" + "-"*70)
    print("ABLATION RESULTS SUMMARY:")
    print("-"*70)
    for config_name, metrics in results.items():
        print(f"{config_name:20s}: "
              f"Blocked={metrics['blocked']}, "
              f"Reliability={metrics['reliability']:.3f}, "
              f"Value={metrics['value']:.3f}")
    
    print("\n" + "="*70)
    print("ABLATION STUDIES COMPLETE")
    print("="*70 + "\n")


def main():
    """
    Run all demonstrations.
    """
    print("\n")
    print("="*70)
    print(" "*15 + "COGNITIVE SWARM FRAMEWORK")
    print(" "*10 + "Full System Demonstration")
    print("="*70)
    print("\nThis demonstration showcases the Secure Decision Pipeline:")
    print("  1. OBSERVE: Raw observations + messages")
    print("  2. ORIENT (Security): Trust Gate filters corrupted messages")
    print("  3. ORIENT (Context): Bayesian Beliefs [SKIPPED in V1]")
    print("  4. DECIDE (Scale): Mean Field aggregation")
    print("  5. ACT (Policy): Neural network decision making")
    print("  6. ACT (Safety): Safety Shield verification")
    
    # Run demonstrations
    try:
        # Demo 1: Single step
        actions, info = run_single_step_demo()
        
        # Demo 2: Multi-step episode
        run_multi_step_episode(num_steps=10)
        
        # Demo 3: Ablation studies
        demonstrate_ablation_studies()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
