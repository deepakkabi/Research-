# Quick demo of the system
"""
Example Usage of Multi-Agent Coordination Environment

This script demonstrates how to use the coordination environment for research.
"""

import numpy as np
from cognitive_swarm.environment import CoordinationEnv


def example_random_policy():
    """Example: Run environment with random policy."""
    print("=" * 60)
    print("Example 1: Random Policy")
    print("=" * 60)
    
    # Create environment
    env = CoordinationEnv(
        num_agents=50,
        num_adversaries=20,
        num_protected=10,
        seed=42
    )
    
    # Reset environment
    obs = env.reset(seed=42)
    print(f"Environment initialized with {len(env.agents)} agents")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Adversaries: {env.num_adversaries}")
    print(f"Protected entities: {env.num_protected}\n")
    
    # Run episode
    total_reward = 0
    episode_length = 0
    
    for step in range(200):
        # Random actions
        actions = {agent: np.random.randint(0, 7) for agent in env.agents}
        
        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Track metrics
        total_reward += sum(rewards.values())
        episode_length += 1
        
        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            active_agents = sum(1 for a in env.agents if env.agent_states[a]['active'])
            active_adversaries = sum(1 for pos in env.adversary_positions if pos[0] != -1)
            print(f"Step {step + 1}: Agents={active_agents}, Adversaries={active_adversaries}, "
                  f"Reward={sum(rewards.values()):.1f}")
        
        # Check termination
        if any(terminations.values()):
            print(f"\nEpisode terminated at step {step + 1}")
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Length: {episode_length} steps")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Average Reward per Step: {total_reward / episode_length:.2f}\n")


def example_communication_analysis():
    """Example: Analyze communication patterns and noise."""
    print("=" * 60)
    print("Example 2: Communication Analysis")
    print("=" * 60)
    
    env = CoordinationEnv(
        num_agents=20,
        comm_radius=7,
        noise_probability=0.2,
        seed=42
    )
    
    obs = env.reset(seed=42)
    
    # Track communication statistics
    total_messages = 0
    corrupted_messages = 0
    neighbor_counts = []
    
    for step in range(100):
        # All agents communicate
        actions = {agent: env.COMMUNICATE for agent in env.agents}
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Analyze this step
        for agent in env.agents:
            if agent in infos:
                # Count neighbors
                messages = obs[agent]['messages']
                neighbor_counts.append(len(messages))
                total_messages += len(messages)
                
                # Check if this agent had noisy channel
                if infos[agent]['noisy_channel']:
                    corrupted_messages += 1
    
    print(f"Communication Statistics over 100 steps:")
    print(f"  Total messages sent: {total_messages}")
    print(f"  Agents with corrupted channels: {corrupted_messages}")
    print(f"  Corruption rate: {corrupted_messages / (100 * env.num_agents):.1%}")
    print(f"  Average neighbors per agent: {np.mean(neighbor_counts):.2f}")
    print(f"  Max neighbors observed: {np.max(neighbor_counts)}")
    print()


def example_safety_constraints():
    """Example: Monitor safety constraints around protected entities."""
    print("=" * 60)
    print("Example 3: Safety Constraint Monitoring")
    print("=" * 60)
    
    env = CoordinationEnv(
        num_agents=30,
        num_adversaries=15,
        num_protected=5,
        seed=42
    )
    
    obs = env.reset(seed=42)
    
    # Track safety violations
    total_interventions = 0
    unsafe_interventions = 0
    min_safe_distance = 3.0  # Minimum safe distance for interventions
    
    for step in range(100):
        # Simple policy: intervention if adversary nearby, otherwise random
        actions = {}
        
        for agent in env.agents:
            # Check for nearby adversaries in local observation
            local_grid = obs[agent]['local_grid']
            has_nearby_adversary = local_grid[:, :, 1].sum() > 0
            
            # Check distance to protected entities
            distances = env.get_protected_distances(agent)
            min_distance = np.min(distances)
            
            if has_nearby_adversary and min_distance >= min_safe_distance:
                actions[agent] = env.INTERVENTION
                total_interventions += 1
            elif has_nearby_adversary and min_distance < min_safe_distance:
                actions[agent] = env.HOLD  # Too close to protected entity
                unsafe_interventions += 1
            else:
                actions[agent] = np.random.randint(0, 4)  # Random movement
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        if any(terms.values()):
            break
    
    print(f"Safety Monitoring Results:")
    print(f"  Total intervention attempts: {total_interventions}")
    print(f"  Prevented unsafe interventions: {unsafe_interventions}")
    print(f"  Safety compliance rate: {total_interventions / (total_interventions + unsafe_interventions):.1%}")
    print()


def example_batching():
    """Example: Demonstrate observation batching for neural networks."""
    print("=" * 60)
    print("Example 4: Observation Batching")
    print("=" * 60)
    
    env = CoordinationEnv(num_agents=10, seed=42)
    obs = env.reset(seed=42)
    
    # Convert to list for batching
    obs_list = list(obs.values())
    
    # Batch observations
    batched = env.collate_observations(obs_list, max_neighbors=20)
    
    print(f"Batched Observation Shapes:")
    for key, value in batched.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    print(f"\nThis batched format is ready for neural network input.")
    print(f"Example: policy_network(batched['local_obs'])")
    print()


def example_role_distribution():
    """Example: Analyze agent role distribution and behavior."""
    print("=" * 60)
    print("Example 5: Role-Based Analysis")
    print("=" * 60)
    
    env = CoordinationEnv(num_agents=100, seed=42)
    obs = env.reset(seed=42)
    
    # Count roles
    role_names = {0: "Scout", 1: "Coordinator", 2: "Support"}
    role_counts = {0: 0, 1: 0, 2: 0}
    
    for agent in env.agents:
        role = env.agent_roles[agent]
        role_counts[role] += 1
    
    print(f"Agent Role Distribution:")
    for role_id, count in role_counts.items():
        percentage = count / env.num_agents * 100
        print(f"  {role_names[role_id]}: {count} ({percentage:.1f}%)")
    
    print(f"\nExpected distribution:")
    print(f"  Scouts: 50%")
    print(f"  Coordinators: 30%")
    print(f"  Support: 20%")
    print()


def run_all_examples():
    """Run all example demonstrations."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Multi-Agent Coordination Environment Examples  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    examples = [
        example_random_policy,
        example_communication_analysis,
        example_safety_constraints,
        example_batching,
        example_role_distribution,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}\n")
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Modify these examples for your research needs")
    print("2. Integrate with neural network modules (Trust Gate, Policy, etc.)")
    print("3. Run experiments with different parameters")
    print("4. Analyze coordination and safety metrics")
    print()


if __name__ == "__main__":
    run_all_examples()
