"""
Evaluation Script for Cognitive Swarm Framework

This script evaluates trained agents on test scenarios and generates
performance metrics and visualizations.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/agent.pt --episodes 100
"""

import torch
import numpy as np
import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_swarm import CognitiveAgent


def create_mock_test_scenario(num_agents: int = 8) -> Dict:
    """Create a mock test scenario for evaluation."""
    return {
        'local_obs': torch.randn(num_agents, 10),
        'messages': torch.randn(num_agents, 8, 8),
        'neighbor_states': torch.randn(num_agents, 8, 6),
        'neighbor_roles': torch.randint(0, 3, (num_agents, 8)),
        'neighbor_mask': torch.ones(num_agents, 8),
        'state_dict': {
            'agent_positions': np.random.rand(num_agents, 2) * 50,
            'protected_positions': np.random.rand(2, 2) * 50,
            'target_positions': np.random.rand(3, 2) * 50,
            'agent_resources': np.random.rand(num_agents) * 100,
            'target_values': np.random.rand(3) * 50
        }
    }


def evaluate_episode(agent: CognitiveAgent, scenario: Dict) -> Dict[str, float]:
    """Evaluate agent on a single episode/scenario."""
    agent.eval()
    
    with torch.no_grad():
        action_logits, value, info = agent(
            local_obs=scenario['local_obs'],
            messages=scenario['messages'],
            neighbor_states=scenario['neighbor_states'],
            neighbor_roles=scenario['neighbor_roles'],
            neighbor_mask=scenario['neighbor_mask']
        )
    
    # Collect metrics
    num_agents = scenario['local_obs'].size(0)
    blocked_count = 0
    
    for agent_id in range(num_agents):
        _, violated, _ = agent.select_action(
            action_logits[agent_id],
            scenario['state_dict'],
            agent_id
        )
        if violated:
            blocked_count += 1
    
    return {
        'mean_value': value.mean().item(),
        'mean_reliability': info['reliability_weights'].mean().item(),
        'safety_block_rate': blocked_count / num_agents,
        'action_entropy': -(torch.softmax(action_logits, dim=-1) * 
                           torch.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean().item()
    }


def run_evaluation(
    agent: CognitiveAgent,
    num_episodes: int = 100,
    num_agents: int = 8
) -> Dict[str, List[float]]:
    """Run full evaluation over multiple episodes."""
    
    results = {
        'mean_value': [],
        'mean_reliability': [],
        'safety_block_rate': [],
        'action_entropy': []
    }
    
    print(f"Running evaluation over {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        scenario = create_mock_test_scenario(num_agents)
        episode_metrics = evaluate_episode(agent, scenario)
        
        for key, value in episode_metrics.items():
            results[key].append(value)
        
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")
    
    return results


def print_summary(results: Dict[str, List[float]]):
    """Print evaluation summary statistics."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    for metric, values in results.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric:25s}: {mean:.4f} ± {std:.4f}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Cognitive Swarm Agent')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--agents', type=int, default=8,
                       help='Number of agents per episode')
    parser.add_argument('--use_beliefs', action='store_true',
                       help='Enable Bayesian beliefs')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Cognitive Swarm - Agent Evaluation")
    print("=" * 70)
    
    # Initialize agent
    agent = CognitiveAgent(
        obs_dim=10,
        message_dim=8,
        state_dim=6,
        action_dim=7,
        use_beliefs=args.use_beliefs
    )
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        agent.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    else:
        print("No checkpoint provided, using randomly initialized agent")
    
    # Run evaluation
    results = run_evaluation(agent, args.episodes, args.agents)
    
    # Print summary
    print_summary(results)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
