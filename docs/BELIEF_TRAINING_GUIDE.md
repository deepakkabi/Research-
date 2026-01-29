# Bayesian Beliefs Module V2 - Training Guide

## Overview

This guide explains how to train the neural likelihood networks in the Bayesian Belief Module V2 for accurate opponent strategy inference.

## Quick Start

```bash
# 1. Generate training data
python scripts/generate_training_data.py --samples_per_type 5000

# 2. Train the model
python scripts/train_beliefs.py --epochs 100

# 3. Use the trained model
python
>>> from bayesian_beliefs_v2 import BayesianBeliefModule
>>> module = BayesianBeliefModule.load_model('checkpoints/belief_module.pt')
```

## Step-by-Step Training Process

### Step 1: Generate Training Data

The neural networks need labeled observations to learn P(observation | strategy_type).

#### Option A: Use Synthetic Data (Recommended for Testing)

```bash
python scripts/generate_training_data.py \
    --samples_per_type 5000 \
    --use_synthetic \
    --output_dir data
```

This generates:
- 5,000 samples per type (20,000 total for 4 types)
- Balanced across all strategy types
- Synthetic but realistic observations
- Saved to `data/belief_train_data.pt`, `data/belief_val_data.pt`, `data/belief_test_data.pt`

#### Option B: Collect from Environment (For Production)

```bash
python scripts/generate_training_data.py \
    --num_episodes 100 \
    --output_dir data
```

**Requirements:**
- Access to `CoordinationEnv`
- Environment must expose adversary types
- Longer collection time but more realistic data

**What gets collected:**
```python
# For each timestep where adversaries are present:
observation = [
    x_pos,                          # [0]
    y_pos,                          # [1]
    distance_to_nearest_friendly,  # [2] ← KEY FEATURE
    movement_speed,                 # [3] ← KEY FEATURE  
    action_frequency,               # [4] ← KEY FEATURE
    time_since_last_action,         # [5]
    health_status,                  # [6]
    resource_level,                 # [7]
    role_id,                        # [8]
    team_id                         # [9]
]
label = adversary_type  # 0, 1, 2, or 3
```

#### Data Quality Requirements

**Minimum dataset size:**
- Training: 10,000 observations (2,500 per type)
- Validation: 3,000 observations
- Test: 3,000 observations

**Balance:** Each type should have roughly equal representation (within 10%)

**Feature coverage:** Observations should cover the full range of behaviors:

| Type | Distance Range | Speed Range | Action Freq Range |
|------|---------------|-------------|------------------|
| 0 (Aggressive) | 0-5 | 0.7-1.0 | 0.6-1.0 |
| 1 (Defensive) | 10-20 | 0.0-0.3 | 0.0-0.3 |
| 2 (Deceptive) | 3-13 | 0.6-0.9 | 0.4-0.7 |
| 3 (Reconnaissance) | 10-15 | 0.7-1.0 | 0.3-0.6 |

### Step 2: Train the Model

```bash
python scripts/train_beliefs.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.001 \
    --hidden_dim 64 \
    --output_dir checkpoints
```

**Training Parameters:**

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `--epochs` | 100 | Training iterations | Increase if loss still decreasing |
| `--batch_size` | 256 | Mini-batch size | Larger = faster, smaller = more stable |
| `--lr` | 0.001 | Learning rate | Reduce if training unstable |
| `--hidden_dim` | 64 | Network size | Increase for complex patterns |
| `--num_types` | 4 | Strategy types | Match your task |

**What happens during training:**
1. Loads train/val/test data from `data_dir`
2. Initializes 4 separate likelihood networks (one per type)
3. Trains using supervised cross-entropy loss
4. Validates after each epoch
5. Saves best model to `checkpoints/belief_module.pt`
6. Generates training curves and confusion matrix

**Expected training time:**
- CPU: ~5-10 minutes for 100 epochs with 20K samples
- GPU: ~1-2 minutes for 100 epochs with 20K samples

### Step 3: Monitor Training

Training produces several outputs:

#### 1. Console Output
```
Epoch 10/100 | Loss: 0.8234 | Acc: 0.65
Epoch 20/100 | Loss: 0.5621 | Acc: 0.78
Epoch 30/100 | Loss: 0.4123 | Acc: 0.85
...
Epoch 100/100 | Loss: 0.2341 | Acc: 0.92

Validation Results:
Overall Accuracy: 90.5%
  Aggressive: 92.3%
  Defensive: 91.1%
  Deceptive: 88.7%
  Reconnaissance: 89.9%
```

#### 2. Training Curves (`logs/training_history.png`)
- **Loss curve**: Should decrease steadily
- **Accuracy curve**: Should increase to >80%

**Good training:**
```
Loss: Smooth decrease from ~1.4 to <0.3
Accuracy: Steady increase to >85%
```

**Poor training (needs tuning):**
```
Loss: Oscillates or plateaus early
Accuracy: Stays below 70%
```

#### 3. Confusion Matrix (`logs/confusion_matrix.png`)
Shows which types are confused with each other.

**Ideal confusion matrix:**
```
             Pred: Agg  Def  Dec  Rec
True: Agg    [950   20   15   15]
      Def    [ 25  940   20   15]
      Dec    [ 30   25  920   25]
      Rec    [ 20   15   25  940]
```

**Problematic confusion:**
- High off-diagonal values = types are hard to distinguish
- One type consistently misclassified = need more training data for that type

#### 4. Training Summary (`checkpoints/training_summary.txt`)
Complete record of training configuration and results.

### Step 4: Validate the Model

After training, check test set performance:

```python
from bayesian_beliefs_v2 import BayesianBeliefModule
import torch

# Load model
module = BayesianBeliefModule.load_model('checkpoints/belief_module.pt')

# Load test data
data = torch.load('data/belief_test_data.pt')
test_obs = data['observations']
test_labels = data['labels']

# Validate
metrics = module.validate(test_obs, test_labels, verbose=True)

# Check results
print(f"Test Accuracy: {metrics['accuracy']:.1%}")
for type_name, acc in metrics['per_class_accuracy'].items():
    print(f"  {type_name}: {acc:.1%}")
```

**Success criteria:**
- ✅ Overall accuracy > 80%
- ✅ Each type accuracy > 70%
- ✅ No type significantly worse than others
- ✅ Confusion matrix shows strong diagonal

### Step 5: Integrate with CognitiveAgent

```python
from cognitive_swarm.agents import CognitiveAgent
from bayesian_beliefs_v2 import BayesianBeliefModule

# Load trained module
belief_module = BayesianBeliefModule.load_model('checkpoints/belief_module.pt')

# Create agent with beliefs
agent = CognitiveAgent(
    obs_dim=10,
    action_dim=7,
    use_beliefs=True,
    num_strategy_types=4  # Must match trained model
)

# Replace with trained module
agent.belief_module = belief_module

# Now agent uses trained likelihoods for inference!
```

## Troubleshooting

### Problem: Training accuracy stays low (<70%)

**Possible causes:**
1. **Insufficient data**: Collect more samples
2. **Imbalanced data**: Ensure equal samples per type
3. **Bad learning rate**: Try `--lr 0.0001` or `--lr 0.01`
4. **Network too small**: Try `--hidden_dim 128`

**Solutions:**
```bash
# Collect more data
python scripts/generate_training_data.py --samples_per_type 10000

# Adjust learning rate
python scripts/train_beliefs.py --lr 0.0001 --epochs 150

# Increase network capacity
python scripts/train_beliefs.py --hidden_dim 128 --epochs 100
```

### Problem: Validation accuracy much lower than training

**Cause:** Overfitting

**Solutions:**
1. Reduce network size: `--hidden_dim 32`
2. Add more dropout (edit `bayesian_beliefs_v2.py`, increase dropout from 0.1 to 0.2)
3. Collect more diverse training data
4. Early stopping (stop when validation accuracy plateaus)

### Problem: One type has much lower accuracy

**Cause:** Insufficient or ambiguous training data for that type

**Solutions:**
1. Collect more samples for that specific type
2. Check if features clearly distinguish that type
3. Verify data labels are correct
4. Consider merging similar types (e.g., combine Deceptive and Reconnaissance if too similar)

### Problem: Training is too slow

**Solutions:**
1. Reduce batch size: `--batch_size 128`
2. Use GPU: `--device cuda`
3. Reduce dataset size (use fewer samples)
4. Reduce epochs: `--epochs 50`

### Problem: Model predicts same type for everything

**Cause:** Data imbalance or network collapsed

**Solutions:**
1. Verify data balance: each type should have ~25% of samples
2. Reinitialize and retrain with fresh random weights
3. Check labels are correct (not all the same)

## Advanced Topics

### Custom Network Architecture

Edit `bayesian_beliefs_v2.py` to modify the likelihood networks:

```python
# In __init__():
self.likelihood_nets = nn.ModuleList([
    nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),  # Increase dropout
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),  # Add batch norm
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim // 2),  # Add layer
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
        nn.Softplus()
    ) for _ in range(num_types)
])
```

### Learned Priors

Enable learned priors instead of uniform:

```python
# In bayesian_beliefs_v2.py, set:
self.use_learned_prior = True

# The prior_net will now be used to initialize beliefs
# based on the first observation
```

### Transfer Learning

If you have a pre-trained model and want to fine-tune on new data:

```python
# Load pre-trained model
module = BayesianBeliefModule.load_model('checkpoints/belief_module_v1.pt')

# Fine-tune on new data
new_obs = torch.load('data/new_training_data.pt')
optimizer = optim.Adam(module.parameters(), lr=0.0001)  # Lower LR

history = module.train_likelihood(
    new_obs['observations'],
    new_obs['labels'],
    optimizer,
    epochs=20  # Fewer epochs
)

# Save fine-tuned model
module.save_model('checkpoints/belief_module_v2_finetuned.pt')
```

### Ensemble Methods

Train multiple models and average their predictions:

```python
# Train 3 models with different random seeds
models = []
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    module = BayesianBeliefModule(num_types=4)
    # ... train ...
    models.append(module)

# Inference with ensemble
def ensemble_forward(obs):
    beliefs = []
    for model in models:
        belief = model(obs)
        beliefs.append(belief)
    
    # Average beliefs
    ensemble_belief = torch.stack(beliefs).mean(dim=0)
    return ensemble_belief
```

## Performance Benchmarks

Expected performance on synthetic data:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Training Accuracy | >80% | >85% | >90% |
| Validation Accuracy | >75% | >82% | >88% |
| Test Accuracy | >75% | >80% | >85% |
| Per-type Accuracy | >70% | >75% | >82% |
| Training Time (CPU) | <10 min | <5 min | <3 min |

Expected performance on real environment data:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Training Accuracy | >70% | >75% | >80% |
| Validation Accuracy | >65% | >70% | >75% |
| Test Accuracy | >65% | >68% | >72% |

Note: Real data is harder due to noise and ambiguous behaviors.

## Deployment Checklist

Before using in production:

- [ ] Generated sufficient training data (>10K samples)
- [ ] Verified data balance (each type 20-30% of dataset)
- [ ] Training accuracy > 80%
- [ ] Validation accuracy > 75%
- [ ] Test accuracy > 75%
- [ ] No type has accuracy < 70%
- [ ] Confusion matrix shows clear diagonal pattern
- [ ] Model saved to `checkpoints/belief_module.pt`
- [ ] Integration test with CognitiveAgent passes
- [ ] Inference time < 5ms per update
- [ ] Documented any data preprocessing steps

## Citation

If you use this module in research, please cite:

```bibtex
@article{your_paper,
  title={Bayesian Opponent Modeling in Multi-Agent Systems},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review test cases in `tests/test_beliefs_v2.py`
3. Examine training logs in `logs/training_summary.txt`
4. Visualize confusion matrix in `logs/confusion_matrix.png`

---

**Last Updated:** January 2026  
**Version:** 2.0  
**Status:** Production Ready
