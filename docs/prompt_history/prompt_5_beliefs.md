# Prompt 5: Bayesian Beliefs
PROMPT 5: BAYESIAN BELIEFS
ACADEMIC RESEARCH DISCLAIMER:
This is for research on decision-making under uncertainty and opponent
modeling in multi-agent systems.

---

You are implementing Bayesian strategy inference for multi-agent systems under 
uncertainty. This module infers opposing agent strategies ("types") from partial 
observations, enabling strategic decision-making despite incomplete information.

### RESEARCH CONTEXT

Standard MARL assumes full observability or treats opposing agents as stationary. 
In reality, other teams have hidden strategies that must be inferred.

**Bayesian Stochastic Games**: Model other teams as having hidden "types" (e.g., 
Strategy_A, Strategy_B, Strategy_C). Agent maintains a belief distribution over types 
and updates it using Bayes' rule as observations arrive.

**Research application**: Decision-making under uncertainty, opponent modeling,
partial observability in competitive multi-agent settings.

**NOTE**: This module is OPTIONAL for Version 1. Consider using simpler heuristics
for initial implementation. Full Bayesian inference can be added for journal version.

### STRATEGY TYPE DEFINITIONS

**CRITICAL**: Define what each strategy type means for implementation:

**Strategy_A (Aggressive Advance)**:
- Characteristics: Direct movement toward targets, high action frequency
- Observable patterns: Short distance to targets, frequent INTERVENTION actions
- Movement: Straight-line paths, minimal evasion
- Training data: Simulated agents with high aggression parameter

**Strategy_B (Defensive Hold)**:
- Characteristics: Maintain position, form perimeter, reactive only
- Observable patterns: Low movement variance, cluster formation
- Movement: Minimal displacement, high cohesion with teammates
- Training data: Simulated agents with defensive posture

**Strategy_C (Deceptive Feint)**:
- Characteristics: False attacks, unpredictable movement, misdirection
- Observable patterns: Approach-retreat cycles, erratic trajectories
- Movement: High entropy paths, frequent direction changes
- Training data: Scripted feint behaviors

**Strategy_D (Reconnaissance)**:
- Characteristics: High mobility, avoid engagement, information gathering
- Observable patterns: Perimeter scanning, maintain safe distance
- Movement: Wide coverage patterns, retreat when approached
- Training data: Scout-type scripted behaviors

### MATHEMATICAL SPECIFICATION

**Strategy Types**: T = {Strategy_A, Strategy_B, Strategy_C, Strategy_D}

**Belief State**: β_t = [P(T=A), P(T=B), P(T=C), P(T=D)]

**Bayes Update**:
β_{t+1}(type) ∝ β_t(type) · P(observation | type)
Where P(observation | type) is the likelihood function.

**Policy Decision**:
Instead of choosing action for one strategy type, choose action that maximizes 
expected utility over belief distribution:
a* = argmax_a Σ_{type} β_t(type) · Q(s, a | type)

### IMPLEMENTATION SPECIFICATION
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianBeliefModule(nn.Module):
    """
    Bayesian strategy inference for multi-agent systems.
    
    Research application: Decision-making under partial observability,
    opponent modeling, handling strategic uncertainty.
    
    NOTE: This is OPTIONAL for Version 1. Can be simplified or skipped.
    """
    
    def __init__(self, obs_dim=10, num_types=4, hidden_dim=64):
        """
        Args:
            obs_dim: Dimension of observation vector (default=10, from environment)
            num_types: Number of strategy types to track (4 types defined above)
            hidden_dim: Hidden layer size for likelihood network
        """
        super().__init__()
        self.num_types = num_types
        self.type_names = ['Strategy_A', 'Strategy_B', 'Strategy_C', 'Strategy_D']
        
        # Likelihood network: P(observation | type)
        # Separate network for each type
        self.likelihood_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive likelihood
            ) for _ in range(num_types)
        ])
        
        # Alternative: Single network with type as input
        # self.likelihood_net = nn.Sequential(
        #     nn.Linear(obs_dim + num_types, hidden_dim),
        #     ...
        # )
        
        # Optional: Prior network (learned from data instead of uniform)
        self.prior_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                observations: torch.Tensor,   # (batch, seq_len, obs_dim)
                prior_belief: torch.Tensor = None  # (batch, num_types)
               ) -> torch.Tensor:
        """
        Update belief distribution over strategy types using Bayes rule.
        
        Returns:
            posterior_belief: (batch, num_types) - updated probabilities
        """
        batch_size = observations.size(0)
        seq_len = observations.size(1)
        
        # Initialize prior
        if prior_belief is None:
            # Start with uniform prior
            belief = torch.ones(batch_size, self.num_types) / self.num_types
            belief = belief.to(observations.device)
        else:
            belief = prior_belief.clone()
        
        # Sequential Bayes update
        for t in range(seq_len):
            obs_t = observations[:, t, :]  # (batch, obs_dim)
            
            # Compute likelihood for each type
            likelihoods = []
            for type_idx in range(self.num_types):
                likelihood = self.likelihood_nets[type_idx](obs_t)
                # Shape: (batch, 1)
                likelihoods.append(likelihood)
            
            likelihoods = torch.cat(likelihoods, dim=-1)
            # Shape: (batch, num_types)
            
            # Bayes update: posterior ∝ prior * likelihood
            belief = belief * likelihoods
            
            # Normalize (ensure probabilities sum to 1)
            belief = belief / (belief.sum(dim=-1, keepdim=True) + 1e-8)
        
        return belief
    
    def compute_expected_value(self, 
                                belief: torch.Tensor,    # (batch, num_types)
                                q_values: torch.Tensor    # (batch, num_actions, num_types)
                               ) -> torch.Tensor:
        """
        Compute expected Q-value over belief distribution.
        
        Args:
            belief: Probability distribution over types
            q_values: Q-values for each action conditioned on each type
        
        Returns:
            expected_q: (batch, num_actions) - expected value over beliefs
        """
        # Q-values: (batch, num_actions, num_types)
        # Belief: (batch, num_types)
        
        # Expand belief for broadcasting
        belief_expanded = belief.unsqueeze(1)  # (batch, 1, num_types)
        
        # Expected Q = Σ belief(type) * Q(a | type)
        expected_q = (q_values * belief_expanded).sum(dim=-1)
        # Shape: (batch, num_actions)
        
        return expected_q
    
    def train_likelihood(self, observations, true_types, optimizer, epochs=100):
        """
        Pre-train likelihood networks using labeled data.
        
        Args:
            observations: (N, obs_dim) - labeled observations
            true_types: (N,) - ground truth types (0, 1, 2, or 3)
            optimizer: torch optimizer
            epochs: number of training iterations
        
        This is important: likelihood networks need to be trained on data
        where you KNOW the strategy type (e.g., from simulations with scripted behaviors).
        """
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute likelihood for each type
            likelihoods = []
            for type_idx in range(self.num_types):
                likelihood = self.likelihood_nets[type_idx](observations)
                likelihoods.append(likelihood)
            
            likelihoods = torch.cat(likelihoods, dim=-1)
            # Shape: (N, num_types)
            
            # Supervised loss: maximize likelihood of true type
            log_likelihoods = torch.log(likelihoods + 1e-8)
            loss = F.nll_loss(log_likelihoods, true_types)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```

### GENERATING TRAINING DATA

To train likelihood networks, you need labeled observations:
```python
def generate_strategy_data(strategy_type, num_samples=1000):
    """
    Generate observations from agents following a specific strategy.
    
    Args:
        strategy_type: 0 (Aggressive), 1 (Defensive), 2 (Deceptive), 3 (Recon)
        num_samples: Number of samples to generate
    
    Returns:
        observations: (num_samples, obs_dim=10) - labeled data
        labels: (num_samples,) - all equal to strategy_type
    """
    obs_dim = 10
    observations = []
    
    for _ in range(num_samples):
        if strategy_type == 0:  # Aggressive
            # High speed toward targets, direct paths
            speed = np.random.uniform(0.8, 1.0)
            directness = np.random.uniform(0.7, 1.0)  # Low path entropy
            distance_to_target = np.random.uniform(0, 5)  # Close to targets
            action_frequency = np.random.uniform(0.7, 1.0)
            # ... construct 10-dim observation vector
            obs = np.array([speed, directness, distance_to_target, action_frequency, ...])
            
        elif strategy_type == 1:  # Defensive
            # Low speed, high clustering with teammates
            speed = np.random.uniform(0.0, 0.3)
            clustering = np.random.uniform(0.7, 1.0)  # Stay close to team
            distance_to_target = np.random.uniform(10, 20)  # Far from targets
            action_frequency = np.random.uniform(0.0, 0.3)
            obs = np.array([speed, clustering, distance_to_target, action_frequency, ...])
            
        elif strategy_type == 2:  # Deceptive
            # High path entropy, approach-retreat patterns
            path_entropy = np.random.uniform(0.7, 1.0)
            direction_changes = np.random.randint(5, 10)
            distance_variance = np.random.uniform(5, 15)  # Erratic distance
            obs = np.array([path_entropy, direction_changes, distance_variance, ...])
            
        elif strategy_type == 3:  # Reconnaissance
            # High mobility, perimeter scanning, retreat when approached
            coverage =
np.random.uniform(0.7, 1.0) # Wide area coverage retreat_frequency = np.random.uniform(0.5, 1.0) safe_distance = np.random.uniform(10, 15) obs = np.array([coverage, retreat_frequency, safe_distance, ...])
    observations.append(obs)
observations = np.array(observations)
labels = np.full(num_samples, strategy_type)
return torch.tensor(observations, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

### DESIGN DECISIONS TO EXPLAIN

1. **Why 4 types (Strategy A/B/C/D)?**
   - Based on common strategic patterns in multi-agent coordination
   - Alternatives: Continuous type space (harder), hierarchical types
   - These 4 cover main behavioral modes: aggressive, defensive, deceptive, scouting

2. **Why separate networks per type vs. single conditional network?**
   - Separate: More parameters, more expressiveness, easier to train
   - Single: Shared features, better generalization, fewer parameters
   - For Version 1: Use separate (simpler to implement)

3. **How to handle belief collapse?**
   - If belief converges to P(Type_A) = 1.0, agent becomes inflexible
   - Solution 1: Add entropy regularization to beliefs
   - Solution 2: Inject small noise into beliefs each step
   - Solution 3: Decay beliefs over time (other team can change strategy)

4. **Simplification for Version 1:**
   - Use only 2 types (Aggressive vs Defensive) instead of 4
   - Use hand-crafted likelihood functions instead of neural networks
   - Update beliefs every 10 steps instead of every step

### TESTING REQUIREMENTS
```python
import pytest
def test_belief_update(): """Verify that beliefs update correctly according to Bayes rule""" module = BayesianBeliefModule(obs_dim=10, num_types=4)
# Start with uniform prior
prior = torch.ones(1, 4) / 4
# Observation that strongly indicates Strategy_A (type 0)
obs = torch.randn(1, 1, 10)
# Manually set likelihood to favor type 0
with torch.no_grad():
    module.likelihood_nets[0][-2].weight.fill_(1.0)
    module.likelihood_nets[0][-2].bias.fill_(1.0)
    for i in [1, 2, 3]:
        module.likelihood_nets[i][-2].weight.fill_(0.1)
        module.likelihood_nets[i][-2].bias.fill_(0.1)
posterior = module(obs, prior)
# Posterior should favor type 0
assert posterior[0, 0] > 0.5
assert posterior.sum().item() == pytest.approx(1.0, abs=1e-5)
print("✓ Belief update test passed")
def test_belief_convergence(): """Verify that belief converges after repeated consistent observations""" module = BayesianBeliefModule(obs_dim=10, num_types=4)
# Generate sequence of observations from Strategy_A type
obs_sequence = torch.randn(1, 20, 10)
belief = module(obs_sequence)
# After 20 consistent observations, should have strong belief
max_belief = belief.max()
assert max_belief > 0.7
print("✓ Belief convergence test passed")
def test_expected_value(): """Verify expected Q-value computation""" module = BayesianBeliefModule(obs_dim=10, num_types=4)
belief = torch.tensor([[0.4, 0.3, 0.2, 0.1]])  # Favor type 0
q_values = torch.randn(1, 7, 4)  # 7 actions (matching environment), 4 types
# Manually set Q-values
q_values[0, 0, :] = torch.tensor([10, 0, 0, 0])  # Action 0 good for type 0
q_values[0, 1, :] = torch.tensor([0, 10, 0, 0])  # Action 1 good for type 1
expected_q = module.compute_expected_value(belief, q_values)
# Action 0 should have higher expected value (since belief favors type 0)
assert expected_q[0, 0] > expected_q[0, 1]
print("✓ Expected value test passed")

### OUTPUT FORMAT

Provide:
1. Complete `bayesian_beliefs.py`
2. Test file `test_beliefs.py`
3. Data generation script `generate_training_data.py`
4. Ablation suggestion:
   - Compare: Perfect info (know true type) vs Bayesian beliefs vs Uniform beliefs
   - Measure: Task success rate under each condition

### CRITICAL NOTE - SIMPLIFICATION FOR VERSION 1

This module is COMPUTATIONALLY EXPENSIVE (requires separate forward passes 
for each type). For Version 1, **strongly consider**:

**Option 1**: Skip entirely, use reactive policy without opponent modeling
**Option 2**: Use only 2 types (Aggressive vs Defensive)
**Option 3**: Use hand-crafted heuristics instead of learned likelihoods:
```python
def heuristic_likelihood(obs, type): if type == 0: # Aggressive return 1.0 if obs[2] < 5 else 0.1 # Close to targets elif type == 1: # Defensive return 1.0 if obs[2] > 10 else 0.1 # Far from targets

Save full neural implementation for journal version.

### ACADEMIC CONTEXT

This work addresses decision-making under uncertainty in multi-agent systems:
- Bayesian opponent modeling
- Partial observability in competitive settings
- Strategy inference from incomplete information

Related work: Bayesian Stochastic Games (Gmytrasiewicz & Doshi), 
Opponent Modeling in MARL (He et al.), Hidden Type Inference
