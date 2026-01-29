"""
Bayesian Belief Module for Opponent Strategy Inference - V2

UPGRADED VERSION 2 IMPLEMENTATION:
- 4 strategy types (Aggressive, Defensive, Deceptive, Reconnaissance)
- Neural likelihood networks (learned from data)
- Every-step updates (maximum accuracy)
- Training capability included
- Backward compatible with V1

Research Context:
This module enables decision-making under uncertainty by maintaining
belief distributions over opponent strategy types. Uses Bayes' rule
to update beliefs as observations accumulate.

Academic Application:
- Opponent modeling in multi-agent systems
- Bayesian inference under partial observability
- Strategic reasoning in competitive settings

V2 Improvements over V1:
1. 4 types instead of 2 (more granular modeling)
2. Learned neural likelihoods (adaptive to data)
3. Training method for supervised learning
4. Every-step updates (higher accuracy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import os


class BayesianBeliefModule(nn.Module):
    """
    Bayesian strategy inference using learned neural likelihoods.
    
    V2 Features:
    1. 4 strategy types (Aggressive, Defensive, Deceptive, Reconnaissance)
    2. Neural likelihood networks (trainable)
    3. Supervised training method
    4. Every-step updates
    5. Backward compatible with V1
    
    Strategy Types:
    - Type 0 (Aggressive): Direct movement toward targets, high action frequency
      * Observable: close distance (<5), high speed (>0.7), frequent actions (>0.6)
    - Type 1 (Defensive): Maintains distance, reactive only, clustered
      * Observable: far distance (>10), low speed (<0.3), infrequent actions (<0.3)
    - Type 2 (Deceptive): Approach-retreat cycles, erratic movement
      * Observable: high distance variance, frequent direction changes
    - Type 3 (Reconnaissance): Perimeter scanning, maintains safe distance
      * Observable: distance 10-15, high speed, wide area coverage
    """
    
    def __init__(
        self, 
        obs_dim: int = 10, 
        num_types: int = 4,  # V2: 4 types (was 2 in V1)
        hidden_dim: int = 64,
        use_neural: bool = True,  # V2: Neural networks (was heuristics in V1)
        update_frequency: int = 1  # V2: Every step (was 5 in V1)
    ):
        """
        Initialize Bayesian Belief Module.
        
        Args:
            obs_dim: Dimension of observation vector (default=10 from environment)
            num_types: Number of strategy types (2 for V1 compat, 4 for V2)
            hidden_dim: Hidden layer size for neural networks
            use_neural: If True, use neural networks; if False, use V1 heuristics
            update_frequency: Update beliefs every N steps (1 for V2, 5 for V1)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.num_types = num_types
        self.hidden_dim = hidden_dim
        self.use_neural = use_neural
        
        # Type names (supports both V1 and V2)
        if num_types == 2:
            self.type_names = ['Aggressive', 'Defensive']
        elif num_types == 4:
            self.type_names = ['Aggressive', 'Defensive', 'Deceptive', 'Reconnaissance']
        else:
            self.type_names = [f'Type_{i}' for i in range(num_types)]
        
        # Update frequency
        self.update_frequency = update_frequency
        self.steps_since_update = 0
        
        # Cache for previous belief
        self._cached_belief = None
        
        # V2: Neural likelihood networks (one per type)
        if use_neural:
            self.likelihood_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),  # Regularization
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, 1),
                    nn.Softplus()  # Ensure positive likelihood
                ) for _ in range(num_types)
            ])
            
            # Optional: Learned prior network (instead of uniform)
            self.use_learned_prior = False  # Can be enabled for advanced usage
            self.prior_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_types),
                nn.Softmax(dim=-1)
            )
        else:
            # V1 compatibility: No neural networks
            self.likelihood_nets = None
            self.prior_net = None
    
    def _compute_likelihood_heuristic(self, obs: torch.Tensor, type_id: int) -> torch.Tensor:
        """
        V1 BACKWARD COMPATIBILITY: Hand-crafted likelihood heuristics.
        
        Only used when use_neural=False (V1 mode).
        
        Args:
            obs: (batch, obs_dim) - observations
            type_id: Type index
            
        Returns:
            likelihood: (batch,) - P(obs | type)
        """
        batch_size = obs.size(0)
        
        # Extract features (assuming environment structure)
        distance = obs[:, 2] if obs.size(1) > 2 else torch.zeros(batch_size, device=obs.device)
        speed = obs[:, 3] if obs.size(1) > 3 else torch.zeros(batch_size, device=obs.device)
        action_freq = obs[:, 4] if obs.size(1) > 4 else torch.zeros(batch_size, device=obs.device)
        
        # V1 heuristics for 2 types
        if type_id == 0:  # Aggressive
            dist_likelihood = torch.sigmoid(-(distance - 5.0))
            speed_likelihood = torch.sigmoid(speed - 0.5)
            action_likelihood = torch.sigmoid(action_freq - 0.5)
            likelihood = dist_likelihood * speed_likelihood * action_likelihood
            
        elif type_id == 1:  # Defensive
            dist_likelihood = torch.sigmoid(distance - 10.0)
            speed_likelihood = torch.sigmoid(-(speed - 0.3))
            action_likelihood = torch.sigmoid(-(action_freq - 0.3))
            likelihood = dist_likelihood * speed_likelihood * action_likelihood
            
        elif type_id == 2:  # Deceptive (V2 only)
            # High variance in distance, frequent direction changes
            # For heuristic mode, approximate as medium distance + high speed
            dist_likelihood = torch.sigmoid(-(torch.abs(distance - 8.0) - 3.0))
            speed_likelihood = torch.sigmoid(speed - 0.6)
            likelihood = dist_likelihood * speed_likelihood
            
        elif type_id == 3:  # Reconnaissance (V2 only)
            # Maintains distance 10-15, high speed
            dist_likelihood = torch.sigmoid(-(torch.abs(distance - 12.5) - 2.5))
            speed_likelihood = torch.sigmoid(speed - 0.7)
            likelihood = dist_likelihood * speed_likelihood
            
        else:
            raise ValueError(f"Invalid type_id: {type_id}")
        
        return likelihood + 1e-8
    
    def _compute_likelihood_neural(self, obs: torch.Tensor, type_id: int) -> torch.Tensor:
        """
        V2: Compute likelihood using neural network.
        
        Args:
            obs: (batch, obs_dim) - observations
            type_id: Type index
            
        Returns:
            likelihood: (batch,) - P(obs | type) from neural network
        """
        likelihood = self.likelihood_nets[type_id](obs)  # (batch, 1)
        return likelihood.squeeze(-1)  # (batch,)
    
    def forward(
        self,
        observations: torch.Tensor,
        prior_belief: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update belief distribution over strategy types using Bayes' rule.
        
        BACKWARD COMPATIBLE: Same signature as V1.
        
        Args:
            observations: (batch, seq_len, obs_dim) or (batch, obs_dim)
                         Observation sequence or single observation
            prior_belief: (batch, num_types) - prior belief distribution
                         If None, starts with uniform distribution
        
        Returns:
            posterior_belief: (batch, num_types) - updated probability distribution
        """
        # Handle single observation (add sequence dimension)
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)  # (batch, 1, obs_dim)
        
        batch_size = observations.size(0)
        seq_len = observations.size(1)
        
        # Initialize prior belief
        if prior_belief is None:
            if self.use_neural and self.use_learned_prior:
                # Use learned prior network (optional V2 feature)
                belief = self.prior_net(observations[:, 0, :])
            else:
                # Uniform prior (default for both V1 and V2)
                belief = torch.ones(batch_size, self.num_types, device=observations.device)
                belief = belief / self.num_types
        else:
            belief = prior_belief.clone()
        
        # Sequential Bayesian update over observation sequence
        for t in range(seq_len):
            obs_t = observations[:, t, :]  # (batch, obs_dim)
            
            # Compute likelihood for each strategy type
            likelihoods = []
            for type_id in range(self.num_types):
                if self.use_neural and self.likelihood_nets is not None:
                    # V2: Neural likelihood
                    likelihood = self._compute_likelihood_neural(obs_t, type_id)
                else:
                    # V1: Heuristic likelihood
                    likelihood = self._compute_likelihood_heuristic(obs_t, type_id)
                
                likelihoods.append(likelihood)
            
            # Stack likelihoods: (batch, num_types)
            likelihoods = torch.stack(likelihoods, dim=1)
            
            # Bayes' rule: posterior ∝ prior × likelihood
            belief = belief * likelihoods
            
            # Normalize to ensure probabilities sum to 1
            belief_sum = belief.sum(dim=1, keepdim=True)
            belief = belief / (belief_sum + 1e-8)
        
        return belief
    
    def get_likely_type(self, belief: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the most probable strategy type from belief distribution.
        
        BACKWARD COMPATIBLE: Same signature as V1.
        
        Args:
            belief: (batch, num_types) - probability distribution
            
        Returns:
            type_indices: (batch,) - index of most likely type
            type_probs: (batch,) - probability of most likely type
        """
        type_probs, type_indices = belief.max(dim=1)
        return type_indices, type_probs
    
    def get_type_name(self, type_idx: int) -> str:
        """Get human-readable name for a strategy type."""
        if 0 <= type_idx < len(self.type_names):
            return self.type_names[type_idx]
        return f"Unknown_{type_idx}"
    
    def compute_expected_value(
        self,
        belief: torch.Tensor,
        q_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expected Q-value over belief distribution.
        
        BACKWARD COMPATIBLE: Same signature as V1.
        
        Args:
            belief: (batch, num_types) - probability distribution over types
            q_values: (batch, num_actions, num_types) - Q-values per action per type
        
        Returns:
            expected_q: (batch, num_actions) - expected Q-value over beliefs
        """
        # Expand belief for broadcasting: (batch, 1, num_types)
        belief_expanded = belief.unsqueeze(1)
        
        # Expected Q-value: E[Q(s,a)] = Σ_type P(type) * Q(s, a | type)
        expected_q = (q_values * belief_expanded).sum(dim=-1)
        
        return expected_q
    
    def should_update(self) -> bool:
        """
        Check if belief should be updated this step.
        
        Returns True every `update_frequency` steps.
        """
        self.steps_since_update += 1
        if self.steps_since_update >= self.update_frequency:
            self.steps_since_update = 0
            return True
        return False
    
    def reset_update_counter(self):
        """Reset the update counter (e.g., at episode start)."""
        self.steps_since_update = 0
        self._cached_belief = None
    
    def get_belief_entropy(self, belief: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of belief distribution.
        
        Args:
            belief: (batch, num_types) - probability distribution
            
        Returns:
            entropy: (batch,) - entropy in nats
        """
        log_belief = torch.log(belief + 1e-8)
        entropy = -(belief * log_belief).sum(dim=1)
        return entropy
    
    # ========================================================================
    # V2 NEW FEATURE: Training Method
    # ========================================================================
    
    def train_likelihood(
        self,
        observations: torch.Tensor,
        true_types: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train likelihood networks using supervised learning.
        
        NEW IN V2: Enables learning from labeled data.
        
        Args:
            observations: (N, obs_dim) - labeled observations
            true_types: (N,) - ground truth types (0, 1, 2, or 3 for 4-type model)
            optimizer: torch optimizer (e.g., Adam)
            epochs: number of training iterations
            batch_size: mini-batch size
            verbose: print progress
        
        Returns:
            history: Dictionary with 'loss' and 'accuracy' lists
        """
        if not self.use_neural:
            raise ValueError("Cannot train: module is in heuristic mode (use_neural=False)")
        
        # Training mode
        self.train()
        
        # History tracking
        history = {'loss': [], 'accuracy': []}
        
        N = observations.size(0)
        num_batches = (N + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            
            # Shuffle data
            perm = torch.randperm(N)
            observations = observations[perm]
            true_types = true_types[perm]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, N)
                
                obs_batch = observations[start_idx:end_idx]
                types_batch = true_types[start_idx:end_idx]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute likelihoods for all types
                likelihoods = []
                for type_idx in range(self.num_types):
                    likelihood = self._compute_likelihood_neural(obs_batch, type_idx)
                    likelihoods.append(likelihood)
                
                likelihoods = torch.stack(likelihoods, dim=1)  # (batch, num_types)
                
                # Supervised loss: maximize likelihood of true type
                # Using negative log-likelihood
                log_likelihoods = torch.log(likelihoods + 1e-8)
                loss = F.nll_loss(log_likelihoods, types_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item() * obs_batch.size(0)
                predictions = likelihoods.argmax(dim=1)
                epoch_correct += (predictions == types_batch).sum().item()
            
            # Epoch statistics
            avg_loss = epoch_loss / N
            accuracy = epoch_correct / N
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")
        
        # Back to eval mode
        self.eval()
        
        return history
    
    def validate(
        self,
        observations: torch.Tensor,
        true_types: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Validate the trained model on a held-out dataset.
        
        NEW IN V2: Evaluate model performance.
        
        Args:
            observations: (N, obs_dim) - validation observations
            true_types: (N,) - ground truth types
            verbose: print results
        
        Returns:
            metrics: Dictionary with validation metrics
        """
        self.eval()
        
        with torch.no_grad():
            # Compute likelihoods
            likelihoods = []
            for type_idx in range(self.num_types):
                if self.use_neural:
                    likelihood = self._compute_likelihood_neural(observations, type_idx)
                else:
                    likelihood = self._compute_likelihood_heuristic(observations, type_idx)
                likelihoods.append(likelihood)
            
            likelihoods = torch.stack(likelihoods, dim=1)  # (N, num_types)
            
            # Predictions
            predictions = likelihoods.argmax(dim=1)
            
            # Metrics
            accuracy = (predictions == true_types).float().mean().item()
            
            # Per-class accuracy
            per_class_acc = {}
            for type_idx in range(self.num_types):
                mask = true_types == type_idx
                if mask.sum() > 0:
                    class_acc = (predictions[mask] == true_types[mask]).float().mean().item()
                    per_class_acc[self.get_type_name(type_idx)] = class_acc
        
        metrics = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        }
        
        if verbose:
            print(f"\nValidation Results:")
            print(f"Overall Accuracy: {accuracy:.2%}")
            for type_name, acc in per_class_acc.items():
                print(f"  {type_name}: {acc:.2%}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        NEW IN V2: Save trained likelihood networks.
        
        Args:
            filepath: Path to save model (e.g., 'checkpoints/belief_module.pt')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'state_dict': self.state_dict(),
            'num_types': self.num_types,
            'obs_dim': self.obs_dim,
            'hidden_dim': self.hidden_dim,
            'use_neural': self.use_neural,
            'update_frequency': self.update_frequency,
            'type_names': self.type_names
        }
        
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """
        Load a trained model.
        
        NEW IN V2: Load pre-trained likelihood networks.
        
        Args:
            filepath: Path to saved model
            device: Device to load model to
        
        Returns:
            module: Loaded BayesianBeliefModule
        """
        state = torch.load(filepath, map_location=device)
        
        # Create module with saved config
        module = cls(
            obs_dim=state['obs_dim'],
            num_types=state['num_types'],
            hidden_dim=state['hidden_dim'],
            use_neural=state['use_neural'],
            update_frequency=state['update_frequency']
        )
        
        # Load weights
        module.load_state_dict(state['state_dict'])
        module.eval()
        
        print(f"Model loaded from {filepath}")
        return module


def test_v2_features():
    """Quick test to verify V2 features work correctly."""
    print("=" * 70)
    print("Testing Bayesian Belief Module V2 Features")
    print("=" * 70)
    
    # Test 1: V2 initialization (4 types, neural)
    print("\n[Test 1] V2 Initialization (4 types, neural)")
    module_v2 = BayesianBeliefModule(obs_dim=10, num_types=4, use_neural=True)
    print(f"✓ Initialized V2 with {module_v2.num_types} types:")
    for i, name in enumerate(module_v2.type_names):
        print(f"  Type {i}: {name}")
    
    # Test 2: V1 backward compatibility (2 types, heuristics)
    print("\n[Test 2] V1 Backward Compatibility (2 types, heuristics)")
    module_v1 = BayesianBeliefModule(obs_dim=10, num_types=2, use_neural=False, update_frequency=5)
    print(f"✓ Initialized V1 compat mode with {module_v1.num_types} types")
    
    # Test 3: Forward pass works for both
    print("\n[Test 3] Forward Pass Compatibility")
    obs = torch.randn(2, 5, 10)  # batch=2, seq=5
    
    belief_v2 = module_v2(obs)
    belief_v1 = module_v1(obs)
    
    print(f"V2 belief shape: {belief_v2.shape} (expected: [2, 4])")
    print(f"V1 belief shape: {belief_v1.shape} (expected: [2, 2])")
    assert belief_v2.shape == (2, 4)
    assert belief_v1.shape == (2, 2)
    print("✓ Forward pass works for both V1 and V2")
    
    # Test 4: Training capability (V2 only)
    print("\n[Test 4] Training Capability (V2)")
    # Generate synthetic training data
    train_obs = torch.randn(100, 10)
    train_labels = torch.randint(0, 4, (100,))
    
    optimizer = torch.optim.Adam(module_v2.parameters(), lr=0.001)
    
    print("Training for 5 epochs...")
    history = module_v2.train_likelihood(
        train_obs, train_labels, optimizer, epochs=5, verbose=False
    )
    
    print(f"✓ Training completed")
    print(f"  Initial loss: {history['loss'][0]:.4f}")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Final accuracy: {history['accuracy'][-1]:.2%}")
    
    # Test 5: Save and load
    print("\n[Test 5] Save and Load Model")
    save_path = '/tmp/test_belief_module.pt'
    module_v2.save_model(save_path)
    
    loaded_module = BayesianBeliefModule.load_model(save_path)
    
    # Verify loaded module produces same output
    belief_original = module_v2(obs)
    belief_loaded = loaded_module(obs)
    
    diff = torch.abs(belief_original - belief_loaded).max()
    print(f"Max difference after load: {diff.item():.6f}")
    assert diff < 1e-5, "Loaded model should produce identical results"
    print("✓ Save and load works correctly")
    
    print("\n" + "=" * 70)
    print("All V2 tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_v2_features()
