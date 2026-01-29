# Prompt 4: Trust Gate
PROMPT 4: TRUST GATE
ACADEMIC RESEARCH DISCLAIMER:
This is for research on Byzantine fault tolerance and robust communication
in distributed multi-agent systems.

---

You are implementing a reliability verification module for multi-agent systems. 
This component detects and filters erroneous messages (from faulty agents 
or communication failures) before they reach the decision-making policy.

### RESEARCH CONTEXT

Standard MARL assumes all communication is reliable. This fails when:
- Agents experience hardware/software faults (send incorrect information)
- Communication channels are corrupted (replaced with noise)
- Messages are corrupted during transmission

**Zero Trust Principle**: "Never trust, always verify" - every message must be 
validated against local observations.

**Research application**: Byzantine fault tolerance in distributed multi-agent systems.

### INTEGRATION WITH ENVIRONMENT NOISE

**CRITICAL CALIBRATION NOTE**:

The Environment (Prompt 1) injects Gaussian noise with σ=2.0 into 20% of messages.

**Noise characteristics**:
- Real messages typically have magnitude in range [0, 10]
- Gaussian noise N(0, σ=2.0) has ~95% of values in [-4, +4]
- Signal-to-noise ratio: Real messages are 2-5x larger than noise

**Consistency threshold calibration**:
- `threshold = 0.5` is calibrated for this noise level
- **If environment changes σ**, retune threshold:
  - Higher σ (more noise) → lower threshold (more tolerant)
  - Lower σ (less noise) → higher threshold (more strict)

**Recommended threshold values**:
```python
# Based on noise standard deviation:
if σ == 1.0:
    threshold = 0.7  # Noise is very distinguishable
elif σ == 2.0:
    threshold = 0.5  # Default (current setting)
elif σ == 5.0:
    threshold = 0.3  # Noise harder to distinguish
```

### MATHEMATICAL SPECIFICATION

Given agent i receiving messages M = {m_1, m_2, ..., m_K} from neighbors:

**Step 1: Consistency Checking**
For each message m_j, compute consistency score:
c_j = similarity(m_j, predict_message_j(local_obs_i))
Where:
•	predict_message_j = what agent i EXPECTS to hear from neighbor j
•	similarity = cosine similarity or Euclidean distance

**Step 2: Reliability Weight Computation**
τ_j = { c_j if c_j > threshold 0.0 otherwise (message rejected) }

**Step 3: Graph Attention (Structural Reliability)**
Use Graph Attention Network to compute importance weights based on graph topology:
α_j = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))
Where:
•	h_i, h_j = agent embeddings
•	|| = concatenation
•	a, W = learnable parameters

**Step 4: Combined Reliability Score**
final_weight_j = τ_j * α_j
filtered_messages = Σ (final_weight_j * m_j)

### IMPLEMENTATION SPECIFICATION
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple

class TrustGate(nn.Module):
    """
    Message reliability verification for multi-agent systems.
    
    Research application: Byzantine fault tolerance, handling communication
    failures and faulty agents in distributed coordination.
    
    Integrates with Environment's noise injection (σ=2.0, 20% corruption rate).
    """
    
    def __init__(self, message_dim=8, hidden_dim=64, num_heads=4, consistency_threshold=0.5):
        """
        Args:
            message_dim: Dimension of message vectors (default=8, from environment)
            hidden_dim: Dimension for GAT hidden layers
            num_heads: Number of attention heads in GAT
            consistency_threshold: Minimum consistency score to accept message
                                   (calibrated for environment's σ=2.0 noise)
        """
        super().__init__()
        self.message_dim = message_dim
        self.threshold = consistency_threshold
        
        # Component 1: Consistency predictor (what should neighbors say?)
        # This is a neural network that predicts expected messages from local obs
        self.message_predictor = nn.Sequential(
            nn.Linear(message_dim + 10, hidden_dim),  # +10 for local context
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Component 2: Graph Attention Network (structural reliability)
        self.gat = GATConv(
            in_channels=message_dim,
            out_channels=message_dim,
            heads=num_heads,
            concat=False,
            dropout=0.1
        )
        
        # Component 3: Combine consistency + attention
        self.fusion = nn.Linear(2, 1)  # Combine two reliability signals
    
    def forward(self, 
                messages: torch.Tensor,           # (batch, num_neighbors, msg_dim=8)
                local_obs: torch.Tensor,          # (batch, obs_dim=10)
                edge_index: torch.Tensor,         # (2, num_edges) - graph structure
                neighbor_ids: torch.Tensor        # (batch, num_neighbors)
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter messages based on reliability.
        
        Returns:
            filtered_messages: (batch, msg_dim) - weighted sum of reliable messages
            reliability_weights: (batch, num_neighbors) - reliability score for each neighbor
                                (these feed into HMFEncoder as trust_weights)
        """
        batch_size = messages.size(0)
        num_neighbors = messages.size(1)
        
        # === STEP 1: Consistency Checking ===
        # Predict what each neighbor SHOULD say based on local observations
        # Expand local_obs to match each neighbor
        local_obs_expanded = local_obs.unsqueeze(1).expand(-1, num_neighbors, -1)
        
        # Concatenate with messages for context
        predictor_input = torch.cat([messages, local_obs_expanded], dim=-1)
        
        expected_messages = self.message_predictor(predictor_input)
        # Shape: (batch, num_neighbors, msg_dim)
        
        # Compute consistency: cosine similarity
        consistency_scores = F.cosine_similarity(
            messages, 
            expected_messages, 
            dim=-1
        )  # Shape: (batch, num_neighbors)
        
        # Apply threshold
        consistency_weights = torch.where(
            consistency_scores > self.threshold,
            consistency_scores,
            torch.zeros_like(consistency_scores)
        )
        
        # === STEP 2: Graph Attention (Structural Reliability) ===
        # Reshape for GAT: (batch * num_neighbors, msg_dim)
        messages_flat = messages.view(-1, self.message_dim)
        
        # Apply GAT
        gat_output = self.gat(messages_flat, edge_index)
        # Shape: (batch * num_neighbors, msg_dim)
        
        # Reshape back
        gat_output = gat_output.view(batch_size, num_neighbors, self.message_dim)
        
        # Compute attention weights (use norm as proxy for importance)
        attention_weights = torch.norm(gat_output, dim=-1)
        # Normalize
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # === STEP 3: Combine Reliability Signals ===
        # Stack consistency and attention
        reliability_signals = torch.stack([consistency_weights, attention_weights], dim=-1)
        # Shape: (batch, num_neighbors, 2)
        
        # Learned fusion
        combined_reliability = self.fusion(reliability_signals).squeeze(-1)
        # Shape: (batch, num_neighbors)
        
        # Normalize (so weights sum to 1)
        reliability_weights = F.softmax(combined_reliability, dim=-1)
        
        # === STEP 4: Filter Messages ===
        # Weighted sum
        filtered_messages = (messages * reliability_weights.unsqueeze(-1)).sum(dim=1)
        # Shape: (batch, msg_dim)
        
        return filtered_messages, reliability_weights
    
    def detect_faulty_agents(self, reliability_weights, threshold=0.1):
        """
        Identify agents with consistently low reliability scores.
        
        Args:
            reliability_weights: (batch, num_neighbors) - history of reliability scores
            threshold: If average reliability < threshold, flag as faulty
        
        Returns:
            faulty_ids: List of neighbor indices to isolate
        """
        avg_reliability = reliability_weights.mean(dim=0)  # Average over batch (time)
        faulty = (avg_reliability < threshold).nonzero(as_tuple=True)[0]
        return faulty.tolist()
```

### DESIGN DECISIONS TO EXPLAIN

1. **Why combine consistency AND attention?**
   - Consistency: Content-based reliability (does message make sense?)
   - Attention: Structural reliability (is sender in a reliable position?)
   - Both are needed: a faulty agent in a critical position might send plausible errors

2. **Why cosine similarity for consistency?**
   - Alternative: Euclidean distance
   - Alternative: Learned distance metric
   - Trade-offs: interpretability vs. expressiveness

3. **Why threshold=0.5?**
   - Too high: false positives (reject good messages)
   - Too low: false negatives (accept bad messages)
   - Calibrated for environment's σ=2.0 noise
   - Should this be learned or fixed?

4. **What if ALL messages are flagged as unreliable?**
   - Current: Agent acts on local observations only
   - Alternative: Lower threshold dynamically
   - Alternative: Request retransmission

### TESTING REQUIREMENTS
```python
import numpy as np

def test_error_detection():
    """Verify that injected errors are detected"""
    gate = TrustGate(message_dim=8, hidden_dim=32)
    
    # Create normal messages
    normal_msgs = torch.randn(1, 5, 8)
    
    # Inject error in message 3 (simulating environment noise)
    normal_msgs[0, 2, :] = torch.randn(8) * 10  # High noise
    
    local_obs = torch.randn(1, 10)
    edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,0]])  # Ring topology
    neighbor_ids = torch.arange(5).unsqueeze(0)
    
    filtered, reliability = gate(normal_msgs, local_obs, edge_index, neighbor_ids)
    
    # Reliability for message 2 should be low
    assert reliability[0, 2] < 0.2
    print("✓ Error detection test passed")

def test_corruption_resilience():
    """Verify performance under 20% message corruption (matching environment)"""
    gate = TrustGate(message_dim=8, hidden_dim=32)
    
    # Simulate 100 time steps with 20% corruption
    success_rate = 0
    for _ in range(100):
        msgs = torch.randn(1, 10, 8)
        
        # Corrupt 2 out of 10 (20%) - matching environment's noise rate
        corrupted = np.random.choice(10, 2, replace=False)
        msgs[0, corrupted, :] = torch.randn(2, 8) * 2.0  # σ=2.0, matching environment
        
        local_obs = torch.randn(1, 10)
        edge_index = torch.randint(0, 10, (2, 20))
        neighbor_ids = torch.arange(10).unsqueeze(0)
        
        _, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
        
        # Check if corrupted messages have low reliability
        if reliability[0, corrupted].mean() < 0.3:
            success_rate += 1
    
    print(f"Detection rate: {success_rate}%")
    assert success_rate > 70  # Should detect at least 70% of corruption
    print("✓ Corruption resilience test passed")

def test_graph_connectivity():
    """Verify that reliability filtering doesn't fragment the network"""
    gate = TrustGate(message_dim=8, hidden_dim=32)
    
    msgs = torch.randn(1, 10, 8)
    local_obs = torch.randn(1, 10)
    edge_index = torch.randint(0, 10, (2, 20))
    neighbor_ids = torch.arange(10).unsqueeze(0)
    
    _, reliability = gate(msgs, local_obs, edge_index, neighbor_ids)
    
    # At least half of neighbors should have non-zero reliability
    non_zero = (reliability > 0.01).sum()
    assert non_zero >= 5
    print("✓ Graph connectivity test passed")

def test_integration_with_environment():
    """
    Test that Trust Gate correctly handles environment's message format.
    This is a critical integration test.
    """
    from coordination_env import CoordinationEnv  # Your environment
    
    env = CoordinationEnv(num_agents=10)
    obs = env.reset()
    
    # Get batched observations
    obs_list = list(obs.values())
    batched = env.collate_observations(obs_list, max_neighbors=20)
    
    gate = TrustGate(message_dim=8, hidden_dim=32)
    
    # This should work without errors
    filtered, reliability = gate(
        batched['messages'],
        batched['local_obs'],
        edge_index=torch.randint(0, 20, (2, 40)),  # Mock graph
        neighbor_ids=batched['neighbor_ids']
    )
    
    assert filtered.shape == (10, 8)  # batch_size=10, message_dim=8
    assert reliability.shape == (10, 20)  # batch_size=10, max_neighbors=20
    print("✓ Environment integration test passed")
```

### OUTPUT FORMAT

Provide:
1. Complete `trust_gate.py`
2. Test file `test_trust.py`
3. Sensitivity analysis:
   - How does detection rate change with threshold?
   - Plot: threshold (x-axis) vs. false positive/negative rates (y-axis)
4. Comparison table:
   | Method | Detection Rate | False Positives | Computation |
   |--------|---------------|----------------|-------------|
   | Consistency only | 65% | 10% | O(N) |
   | GAT only | 70% | 15% | O(N²) |
   | Combined (ours) | 85% | 5% | O(N²) |

### CRITICAL INTEGRATION NOTE

This module produces `reliability_weights` that feed into:
- `HMFEncoder.forward(trust_weights=reliability_weights)` - only reliable neighbors contribute to mean field
- Later analysis (identify faulty agents over time)

**Output format MUST match HMFEncoder expectations**:
- Shape: (batch, max_neighbors)
- Range: [0, 1] (soft weights, not binary)
- Sum normalized (for interpretability)

### ACADEMIC CONTEXT

This work addresses Byzantine fault tolerance in multi-agent systems:
- Handling faulty agents in distributed coordination
- Communication failure resilience
- Graph-based reliability verification

Related work: Byzantine Fault Tolerance (Lamport et al.), Active Defense 
Multi-Agent Communication (ADMAC), Robust MARL surveys
