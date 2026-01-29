# Main training loop
"""
Training Script - PPO for Cognitive Agent

Simple Proximal Policy Optimization (PPO) training loop for the CognitiveAgent.
This demonstrates how the agent can be trained in a multi-agent RL setting.

Note: This is a simplified training loop for demonstration. For production,
consider using frameworks like RLlib, Stable-Baselines3, or CleanRL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_swarm import CognitiveAgent


class PPOTrainer:
    """
    Simple PPO trainer for CognitiveAgent.
    
    Implements the core PPO algorithm:
    - Clipped surrogate objective
    - Value function loss
    - Entropy bonus for exploration
    """
    
    def __init__(self, 
                 agent: CognitiveAgent,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer for both policy and value networks
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.safety_violations = deque(maxlen=100)
        self.trust_scores = deque(maxlen=100)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def ppo_update(self, trajectories, num_epochs=4, batch_size=64):
        """
        Perform PPO update using collected trajectories.
        
        Args:
            trajectories: List of (obs, action, log_prob, value, reward, done, info)
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size
        """
        # Unpack trajectories
        obs_list = [t[0] for t in trajectories]
        actions = torch.tensor([t[1] for t in trajectories], dtype=torch.long)
        old_log_probs = torch.tensor([t[2] for t in trajectories], dtype=torch.float32)
        values = [t[3] for t in trajectories]
        rewards = [t[4] for t in trajectories]
        dones = [t[5] for t in trajectories]
        
        # Compute advantages and returns
        next_value = 0  # Simplified: assume episode ends
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(num_epochs):
            # Mini-batch updates
            num_samples = len(trajectories)
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = [obs_list[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Recompute action logits and values
                # Note: This is simplified - in practice, you'd need to reconstruct
                # full observations including messages, neighbor states, etc.
                # For demo purposes, we'll use mock data
                
                # Mock forward pass (in real training, use actual observations)
                batch_size_actual = len(batch_indices)
                mock_local_obs = torch.randn(batch_size_actual, 10)
                mock_messages = torch.randn(batch_size_actual, 5, 8)
                mock_neighbor_states = torch.randn(batch_size_actual, 5, 6)
                mock_neighbor_roles = torch.randint(0, 3, (batch_size_actual, 5))
                mock_neighbor_mask = torch.ones(batch_size_actual, 5)
                
                action_logits, value_pred, _ = self.agent(
                    local_obs=mock_local_obs,
                    messages=mock_messages,
                    neighbor_states=mock_neighbor_states,
                    neighbor_roles=mock_neighbor_roles,
                    neighbor_mask=mock_neighbor_mask
                )
                
                # Compute new log probs
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred_clipped = values[batch_indices[0]] + torch.clamp(
                    value_pred.squeeze() - values[batch_indices[0]],
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss1 = (value_pred.squeeze() - batch_returns) ** 2
                value_loss2 = (value_pred_clipped - batch_returns) ** 2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # Return average losses
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def train_episode(self, max_steps=100):
        """
        Run a single training episode.
        
        In practice, this would interact with the actual environment.
        For demonstration, we use mock data.
        """
        trajectories = []
        episode_reward = 0
        safety_violations = 0
        trust_scores = []
        
        for step in range(max_steps):
            # Mock observation (in practice, get from environment)
            local_obs = torch.randn(1, 10)
            messages = torch.randn(1, 5, 8)
            neighbor_states = torch.randn(1, 5, 6)
            neighbor_roles = torch.randint(0, 3, (1, 5))
            neighbor_mask = torch.ones(1, 5)
            
            # Forward pass
            with torch.no_grad():
                action_logits, value, info = self.agent(
                    local_obs=local_obs,
                    messages=messages,
                    neighbor_states=neighbor_states,
                    neighbor_roles=neighbor_roles,
                    neighbor_mask=neighbor_mask
                )
            
            # Sample action
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Mock state dict for safety
            state_dict = {
                'agent_positions': np.array([[step, step]]),
                'protected_positions': np.array([[50, 50]]),
                'target_positions': np.array([[25, 25]]),
                'agent_resources': np.array([100]),
                'target_values': np.array([50])
            }
            
            # Apply safety shield
            final_action, violated, _ = self.agent.select_action(
                action_logits[0], state_dict, agent_id=0, deterministic=False
            )
            
            # Mock reward (in practice, get from environment)
            reward = np.random.randn() + (0 if violated else 1)
            done = (step == max_steps - 1)
            
            # Store trajectory
            obs_data = {
                'local_obs': local_obs,
                'messages': messages,
                'neighbor_states': neighbor_states,
                'neighbor_roles': neighbor_roles,
                'neighbor_mask': neighbor_mask
            }
            trajectories.append((
                obs_data,
                final_action,
                log_prob.item(),
                value.item(),
                reward,
                done,
                info
            ))
            
            episode_reward += reward
            if violated:
                safety_violations += 1
            trust_scores.append(info['reliability_weights'].mean().item())
        
        # Record statistics
        self.episode_rewards.append(episode_reward)
        self.safety_violations.append(safety_violations)
        self.trust_scores.append(np.mean(trust_scores))
        
        return trajectories, episode_reward, safety_violations
    
    def train(self, num_episodes=100, update_interval=10):
        """
        Main training loop.
        """
        print("\n" + "="*70)
        print("TRAINING COGNITIVE AGENT WITH PPO")
        print("="*70 + "\n")
        
        print(f"Configuration:")
        print(f"  - Episodes: {num_episodes}")
        print(f"  - Update interval: {update_interval} episodes")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - GAE lambda: {self.gae_lambda}")
        print(f"  - Clip epsilon: {self.clip_epsilon}\n")
        
        all_trajectories = []
        
        for episode in range(num_episodes):
            # Collect episode
            trajectories, episode_reward, violations = self.train_episode()
            all_trajectories.extend(trajectories)
            
            # Update policy
            if (episode + 1) % update_interval == 0:
                losses = self.ppo_update(all_trajectories)
                all_trajectories = []  # Clear buffer
                
                # Print statistics
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Reward: {np.mean(list(self.episode_rewards)):.2f} ± {np.std(list(self.episode_rewards)):.2f}")
                print(f"  Violations: {np.mean(list(self.safety_violations)):.2f}")
                print(f"  Trust: {np.mean(list(self.trust_scores)):.3f}")
                print(f"  Policy loss: {losses['policy_loss']:.4f}")
                print(f"  Value loss: {losses['value_loss']:.4f}")
                print(f"  Entropy: {losses['entropy']:.4f}")
                print()
        
        print("="*70)
        print("TRAINING COMPLETE")
        print("="*70 + "\n")


def main():
    """
    Main training script.
    """
    # Initialize agent
    agent = CognitiveAgent(
        obs_dim=10,
        message_dim=8,
        state_dim=6,
        action_dim=7,
        num_roles=3,
        hidden_dim=128,
        use_beliefs=False
    )
    
    # Initialize trainer
    trainer = PPOTrainer(
        agent=agent,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Train
    trainer.train(num_episodes=100, update_interval=10)
    
    # Save checkpoint
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "cognitive_agent.pt"
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'episode_rewards': list(trainer.episode_rewards),
        'safety_violations': list(trainer.safety_violations),
        'trust_scores': list(trainer.trust_scores)
    }, checkpoint_path)
    
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
