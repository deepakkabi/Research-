"""
Training Data Generation for Bayesian Belief Module

This script collects labeled observations from scripted adversaries
in the coordination environment to train the neural likelihood networks.

Usage:
    python scripts/generate_training_data.py --num_episodes 100 --output data/belief_training_data.pt
"""

import torch
import numpy as np
import argparse
import os
from typing import Tuple, List, Dict
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_adversary_data(
    adversary_type: int,
    num_samples: int = 1000,
    obs_dim: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic observations for a specific adversary type.
    
    This is used when you don't have access to the actual environment
    or want to augment real data with synthetic examples.
    
    Args:
        adversary_type: 0 (Aggressive), 1 (Defensive), 2 (Deceptive), 3 (Recon)
        num_samples: Number of samples to generate
        obs_dim: Observation dimension (default=10)
    
    Returns:
        observations: (num_samples, obs_dim)
        labels: (num_samples,) - all equal to adversary_type
    """
    observations = []
    
    for _ in range(num_samples):
        obs = np.zeros(obs_dim)
        
        # Position (random, not used for type inference)
        obs[0] = np.random.uniform(0, 100)  # x
        obs[1] = np.random.uniform(0, 100)  # y
        
        if adversary_type == 0:  # Aggressive
            # Close distance, high speed, frequent actions
            obs[2] = np.random.uniform(0, 5)      # distance_to_nearest_friendly
            obs[3] = np.random.uniform(0.7, 1.0)  # movement_speed
            obs[4] = np.random.uniform(0.6, 1.0)  # action_frequency
            obs[5] = np.random.uniform(0, 2)      # time_since_last_action (short)
            obs[6] = np.random.uniform(0.7, 1.0)  # health_status (confident)
            obs[7] = np.random.uniform(0.5, 1.0)  # resource_level
            
        elif adversary_type == 1:  # Defensive
            # Far distance, low speed, infrequent actions
            obs[2] = np.random.uniform(10, 20)    # distance_to_nearest_friendly
            obs[3] = np.random.uniform(0.0, 0.3)  # movement_speed
            obs[4] = np.random.uniform(0.0, 0.3)  # action_frequency
            obs[5] = np.random.uniform(3, 10)     # time_since_last_action (long)
            obs[6] = np.random.uniform(0.5, 1.0)  # health_status
            obs[7] = np.random.uniform(0.3, 0.8)  # resource_level
            
        elif adversary_type == 2:  # Deceptive
            # Variable distance (approach-retreat), high speed, erratic
            # Distance oscillates between close and medium
            phase = np.random.uniform(0, 2 * np.pi)
            obs[2] = 8.0 + 5.0 * np.sin(phase)    # distance (3-13 range)
            obs[3] = np.random.uniform(0.6, 0.9)  # movement_speed (high)
            obs[4] = np.random.uniform(0.4, 0.7)  # action_frequency (moderate)
            obs[5] = np.random.uniform(1, 5)      # time_since_last_action
            obs[6] = np.random.uniform(0.6, 0.9)  # health_status
            obs[7] = np.random.uniform(0.4, 0.9)  # resource_level
            
        elif adversary_type == 3:  # Reconnaissance
            # Maintains safe distance 10-15, high speed, wide coverage
            obs[2] = np.random.uniform(10, 15)    # distance_to_nearest_friendly
            obs[3] = np.random.uniform(0.7, 1.0)  # movement_speed (high)
            obs[4] = np.random.uniform(0.3, 0.6)  # action_frequency (moderate)
            obs[5] = np.random.uniform(1, 4)      # time_since_last_action
            obs[6] = np.random.uniform(0.8, 1.0)  # health_status (cautious)
            obs[7] = np.random.uniform(0.6, 1.0)  # resource_level
        
        else:
            raise ValueError(f"Invalid adversary_type: {adversary_type}")
        
        # Remaining features (role_id, team_id)
        obs[8] = adversary_type  # role_id matches type
        obs[9] = 1.0  # team_id (adversary team)
        
        observations.append(obs)
    
    observations = np.array(observations)
    labels = np.full(num_samples, adversary_type, dtype=np.int64)
    
    return torch.tensor(observations, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def generate_training_data_from_environment(
    num_episodes: int = 100,
    episode_length: int = 500,
    obs_dim: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect training data from the actual coordination environment.
    
    NOTE: This function assumes you have access to CoordinationEnv.
    If not available, use generate_synthetic_adversary_data() instead.
    
    Args:
        num_episodes: Number of episodes to collect
        episode_length: Steps per episode
        obs_dim: Observation dimension
    
    Returns:
        observations: (N, obs_dim) - collected observations
        labels: (N,) - adversary type labels
    """
    try:
        from cognitive_swarm.environment import CoordinationEnv
    except ImportError:
        print("WARNING: CoordinationEnv not available. Using synthetic data instead.")
        return generate_balanced_synthetic_data(
            samples_per_type=num_episodes * episode_length // 4
        )
    
    # Initialize environment
    env = CoordinationEnv(
        num_agents=100,
        num_adversaries=30,
        grid_size=100
    )
    
    all_observations = []
    all_labels = []
    
    print(f"Collecting data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        for step in range(episode_length):
            # Extract adversary observations
            # NOTE: This assumes your environment has a method to get adversary info
            # You may need to adapt this based on your actual environment API
            
            try:
                # Get adversary observations and types
                adv_observations, adv_types = extract_adversary_info(env, obs)
                
                if len(adv_observations) > 0:
                    all_observations.append(adv_observations)
                    all_labels.append(adv_types)
            except Exception as e:
                print(f"Warning: Could not extract adversary info at episode {episode}, step {step}: {e}")
            
            # Step environment with random actions
            actions = {agent: env.action_space.sample() for agent in env.agents}
            obs, rewards, dones, infos = env.step(actions)
            
            if all(dones.values()):
                break
        
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")
    
    # Concatenate all collected data
    if len(all_observations) > 0:
        observations = torch.cat(all_observations, dim=0)
        labels = torch.cat(all_labels, dim=0)
    else:
        print("WARNING: No data collected from environment. Using synthetic data.")
        return generate_balanced_synthetic_data(samples_per_type=5000)
    
    print(f"\nCollected {observations.size(0)} observations")
    
    return observations, labels


def extract_adversary_info(env, obs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract adversary observations and type labels from environment.
    
    NOTE: This is a placeholder. You need to implement this based on
    your actual environment's API.
    
    Args:
        env: CoordinationEnv instance
        obs: Current observations
    
    Returns:
        adv_obs: (num_adversaries, obs_dim)
        adv_types: (num_adversaries,)
    """
    # PLACEHOLDER IMPLEMENTATION
    # Replace this with actual environment API calls
    
    # Example: If your environment stores adversary info
    # adv_obs = env.get_adversary_observations()
    # adv_types = env.get_adversary_types()
    
    # For now, return empty tensors (will trigger synthetic data generation)
    return torch.empty(0, 10), torch.empty(0, dtype=torch.long)


def generate_balanced_synthetic_data(
    samples_per_type: int = 5000,
    obs_dim: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate balanced synthetic dataset with equal samples per type.
    
    Args:
        samples_per_type: Number of samples per adversary type
        obs_dim: Observation dimension
    
    Returns:
        observations: (samples_per_type * 4, obs_dim)
        labels: (samples_per_type * 4,)
    """
    print(f"\nGenerating balanced synthetic dataset...")
    print(f"  Samples per type: {samples_per_type}")
    print(f"  Total samples: {samples_per_type * 4}")
    
    all_obs = []
    all_labels = []
    
    for adversary_type in range(4):
        type_name = ['Aggressive', 'Defensive', 'Deceptive', 'Reconnaissance'][adversary_type]
        print(f"  Generating {type_name} (type {adversary_type})...")
        
        obs, labels = generate_synthetic_adversary_data(
            adversary_type=adversary_type,
            num_samples=samples_per_type,
            obs_dim=obs_dim
        )
        
        all_obs.append(obs)
        all_labels.append(labels)
    
    # Concatenate
    observations = torch.cat(all_obs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Shuffle
    perm = torch.randperm(observations.size(0))
    observations = observations[perm]
    labels = labels[perm]
    
    print(f"\n✓ Generated {observations.size(0)} total samples")
    
    return observations, labels


def split_train_val_test(
    observations: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split data into train/validation/test sets.
    
    Args:
        observations: (N, obs_dim)
        labels: (N,)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
    
    Returns:
        splits: Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    N = observations.size(0)
    
    # Shuffle
    perm = torch.randperm(N)
    observations = observations[perm]
    labels = labels[perm]
    
    # Split indices
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    
    splits = {
        'train': (observations[:train_end], labels[:train_end]),
        'val': (observations[train_end:val_end], labels[train_end:val_end]),
        'test': (observations[val_end:], labels[val_end:])
    }
    
    print(f"\nDataset split:")
    print(f"  Train: {splits['train'][0].size(0)} samples ({train_ratio:.0%})")
    print(f"  Val:   {splits['val'][0].size(0)} samples ({val_ratio:.0%})")
    print(f"  Test:  {splits['test'][0].size(0)} samples ({test_ratio:.0%})")
    
    return splits


def analyze_dataset(observations: torch.Tensor, labels: torch.Tensor):
    """
    Print dataset statistics.
    
    Args:
        observations: (N, obs_dim)
        labels: (N,)
    """
    print("\n" + "=" * 70)
    print("Dataset Analysis")
    print("=" * 70)
    
    # Overall stats
    print(f"\nTotal samples: {observations.size(0)}")
    print(f"Observation dimension: {observations.size(1)}")
    
    # Per-type distribution
    print(f"\nClass distribution:")
    type_names = ['Aggressive', 'Defensive', 'Deceptive', 'Reconnaissance']
    for type_idx in range(4):
        count = (labels == type_idx).sum().item()
        percentage = 100.0 * count / labels.size(0)
        print(f"  Type {type_idx} ({type_names[type_idx]:15s}): {count:6d} ({percentage:5.2f}%)")
    
    # Feature statistics (key features only)
    print(f"\nKey feature statistics:")
    feature_names = ['x_pos', 'y_pos', 'distance', 'speed', 'action_freq']
    for idx, name in enumerate(feature_names[:5]):
        mean = observations[:, idx].mean().item()
        std = observations[:, idx].std().item()
        min_val = observations[:, idx].min().item()
        max_val = observations[:, idx].max().item()
        print(f"  {name:12s}: mean={mean:6.2f}, std={std:6.2f}, min={min_val:6.2f}, max={max_val:6.2f}")


def save_dataset(
    splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    output_dir: str = 'data'
):
    """
    Save train/val/test splits to disk.
    
    Args:
        splits: Dictionary with 'train', 'val', 'test' splits
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, (obs, labels) in splits.items():
        filepath = os.path.join(output_dir, f'belief_{split_name}_data.pt')
        torch.save({'observations': obs, 'labels': labels}, filepath)
        print(f"Saved {split_name} data to {filepath}")
    
    # Also save combined file for backward compatibility
    all_obs = torch.cat([splits['train'][0], splits['val'][0], splits['test'][0]], dim=0)
    all_labels = torch.cat([splits['train'][1], splits['val'][1], splits['test'][1]], dim=0)
    
    combined_path = os.path.join(output_dir, 'belief_training_data.pt')
    torch.save({'observations': all_obs, 'labels': all_labels}, combined_path)
    print(f"Saved combined data to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate training data for Bayesian Belief Module')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to collect (if using environment)')
    parser.add_argument('--samples_per_type', type=int, default=5000,
                       help='Samples per type (if using synthetic data)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for data files')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Force use of synthetic data (don\'t try environment)')
    parser.add_argument('--obs_dim', type=int, default=10,
                       help='Observation dimension')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bayesian Belief Module - Training Data Generation")
    print("=" * 70)
    
    # Generate data
    if args.use_synthetic:
        print("\nUsing synthetic data generation (--use_synthetic flag set)")
        observations, labels = generate_balanced_synthetic_data(
            samples_per_type=args.samples_per_type,
            obs_dim=args.obs_dim
        )
    else:
        print("\nAttempting to collect from environment...")
        observations, labels = generate_training_data_from_environment(
            num_episodes=args.num_episodes,
            obs_dim=args.obs_dim
        )
    
    # Analyze dataset
    analyze_dataset(observations, labels)
    
    # Split into train/val/test
    splits = split_train_val_test(observations, labels)
    
    # Save
    save_dataset(splits, output_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Data generation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Inspect the generated data to verify correctness")
    print("2. Run training: python scripts/train_beliefs.py")
    print("3. Validate the trained model")


if __name__ == "__main__":
    main()
