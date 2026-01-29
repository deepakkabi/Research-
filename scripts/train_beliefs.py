"""
Training Script for Bayesian Belief Module V2

This script trains the neural likelihood networks using the collected
labeled observations.

Usage:
    python scripts/train_beliefs.py --data data/belief_training_data.pt --epochs 100
"""

import torch
import torch.optim as optim
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_swarm.modules.bayesian_beliefs import BayesianBeliefModule


def load_data(data_dir: str = 'data') -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load train/val/test data.
    
    Args:
        data_dir: Directory containing data files
    
    Returns:
        data_splits: Dictionary with 'train', 'val', 'test' data
    """
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        filepath = os.path.join(data_dir, f'belief_{split_name}_data.pt')
        
        if os.path.exists(filepath):
            data = torch.load(filepath)
            splits[split_name] = (data['observations'], data['labels'])
            print(f"Loaded {split_name}: {data['observations'].size(0)} samples")
        else:
            print(f"Warning: {filepath} not found, skipping {split_name} split")
    
    # Fallback: Try loading combined file and split it
    if not splits:
        combined_path = os.path.join(data_dir, 'belief_training_data.pt')
        if os.path.exists(combined_path):
            print(f"Loading combined data from {combined_path}")
            data = torch.load(combined_path)
            obs = data['observations']
            labels = data['labels']
            
            # Split 70/15/15
            N = obs.size(0)
            train_end = int(0.7 * N)
            val_end = int(0.85 * N)
            
            splits['train'] = (obs[:train_end], labels[:train_end])
            splits['val'] = (obs[train_end:val_end], labels[train_end:val_end])
            splits['test'] = (obs[val_end:], labels[val_end:])
            
            print(f"Split into train/val/test: {train_end}/{val_end-train_end}/{N-val_end}")
        else:
            raise FileNotFoundError(f"No data files found in {data_dir}")
    
    return splits


def plot_training_history(
    history: Dict[str, list],
    val_metrics: Dict[str, float],
    output_path: str = 'logs/training_history.png'
):
    """
    Plot training loss and accuracy curves.
    
    Args:
        history: Training history from train_likelihood()
        val_metrics: Validation metrics
        output_path: Path to save plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history['loss'], label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot([100 * acc for acc in history['accuracy']], label='Training Accuracy')
    ax2.axhline(y=100 * val_metrics['accuracy'], color='r', linestyle='--', 
                label=f'Val Accuracy: {val_metrics["accuracy"]:.1%}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to {output_path}")


def plot_confusion_matrix(
    module: BayesianBeliefModule,
    observations: torch.Tensor,
    labels: torch.Tensor,
    output_path: str = 'logs/confusion_matrix.png'
):
    """
    Plot confusion matrix for the trained model.
    
    Args:
        module: Trained BayesianBeliefModule
        observations: Test observations
        labels: True labels
        output_path: Path to save plot
    """
    import numpy as np
    
    module.eval()
    num_types = module.num_types
    confusion_matrix = np.zeros((num_types, num_types), dtype=int)
    
    with torch.no_grad():
        # Get predictions
        likelihoods = []
        for type_idx in range(num_types):
            likelihood = module._compute_likelihood_neural(observations, type_idx)
            likelihoods.append(likelihood)
        
        likelihoods = torch.stack(likelihoods, dim=1)
        predictions = likelihoods.argmax(dim=1)
        
        # Build confusion matrix
        for true_label in range(num_types):
            for pred_label in range(num_types):
                mask = (labels == true_label) & (predictions == pred_label)
                confusion_matrix[true_label, pred_label] = mask.sum().item()
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Labels
    type_names = module.type_names
    ax.set_xticks(range(num_types))
    ax.set_yticks(range(num_types))
    ax.set_xticklabels(type_names, rotation=45, ha='right')
    ax.set_yticklabels(type_names)
    
    # Add text annotations
    for i in range(num_types):
        for j in range(num_types):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="black" if confusion_matrix[i, j] < confusion_matrix.max()/2 else "white")
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")


def evaluate_model(
    module: BayesianBeliefModule,
    test_obs: torch.Tensor,
    test_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the trained model.
    
    Args:
        module: Trained module
        test_obs: Test observations
        test_labels: Test labels
    
    Returns:
        metrics: Evaluation metrics
    """
    print("\n" + "=" * 70)
    print("Model Evaluation on Test Set")
    print("=" * 70)
    
    metrics = module.validate(test_obs, test_labels, verbose=True)
    
    # Additional metrics
    module.eval()
    with torch.no_grad():
        # Get predictions
        likelihoods = []
        for type_idx in range(module.num_types):
            likelihood = module._compute_likelihood_neural(test_obs, type_idx)
            likelihoods.append(likelihood)
        
        likelihoods = torch.stack(likelihoods, dim=1)
        predictions = likelihoods.argmax(dim=1)
        
        # Confidence statistics
        max_likelihood = likelihoods.max(dim=1)[0]
        avg_confidence = max_likelihood.mean().item()
        
        # Entropy (uncertainty)
        probs = likelihoods / (likelihoods.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        avg_entropy = entropy.mean().item()
        
        metrics['avg_confidence'] = avg_confidence
        metrics['avg_entropy'] = avg_entropy
    
    print(f"\nAdditional Metrics:")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"  Average entropy: {avg_entropy:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Bayesian Belief Module')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--num_types', type=int, default=4,
                       help='Number of strategy types')
    parser.add_argument('--obs_dim', type=int, default=10,
                       help='Observation dimension')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directory to save trained model')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bayesian Belief Module V2 - Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num types: {args.num_types}")
    print(f"  Device: {args.device}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    data_splits = load_data(args.data_dir)
    
    train_obs, train_labels = data_splits['train']
    val_obs, val_labels = data_splits['val']
    test_obs, test_labels = data_splits['test']
    
    # Move to device
    device = torch.device(args.device)
    train_obs = train_obs.to(device)
    train_labels = train_labels.to(device)
    val_obs = val_obs.to(device)
    val_labels = val_labels.to(device)
    test_obs = test_obs.to(device)
    test_labels = test_labels.to(device)
    
    # Initialize module
    print(f"\nInitializing Bayesian Belief Module V2...")
    module = BayesianBeliefModule(
        obs_dim=args.obs_dim,
        num_types=args.num_types,
        hidden_dim=args.hidden_dim,
        use_neural=True,  # V2: Neural networks
        update_frequency=1  # V2: Every step
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in module.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(module.parameters(), lr=args.lr)
    
    # Train
    print(f"\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    history = module.train_likelihood(
        observations=train_obs,
        true_types=train_labels,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True
    )
    
    # Validation
    print(f"\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    
    val_metrics = module.validate(val_obs, val_labels, verbose=True)
    
    # Test
    test_metrics = evaluate_model(module, test_obs, test_labels)
    
    # Plot results
    print(f"\nGenerating visualizations...")
    plot_training_history(history, val_metrics, 'logs/training_history.png')
    plot_confusion_matrix(module, test_obs, test_labels, 'logs/confusion_matrix.png')
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'belief_module.pt')
    module.save_model(model_path)
    
    # Save training summary
    summary_path = os.path.join(args.output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Bayesian Belief Module V2 - Training Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Hidden dim: {args.hidden_dim}\n")
        f.write(f"  Num types: {args.num_types}\n\n")
        
        f.write(f"Training Data:\n")
        f.write(f"  Train: {train_obs.size(0)} samples\n")
        f.write(f"  Val: {val_obs.size(0)} samples\n")
        f.write(f"  Test: {test_obs.size(0)} samples\n\n")
        
        f.write(f"Final Training Results:\n")
        f.write(f"  Loss: {history['loss'][-1]:.4f}\n")
        f.write(f"  Accuracy: {history['accuracy'][-1]:.2%}\n\n")
        
        f.write(f"Validation Results:\n")
        f.write(f"  Accuracy: {val_metrics['accuracy']:.2%}\n")
        for type_name, acc in val_metrics['per_class_accuracy'].items():
            f.write(f"    {type_name}: {acc:.2%}\n")
        f.write("\n")
        
        f.write(f"Test Results:\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.2%}\n")
        for type_name, acc in test_metrics['per_class_accuracy'].items():
            f.write(f"    {type_name}: {acc:.2%}\n")
        f.write(f"  Avg confidence: {test_metrics['avg_confidence']:.4f}\n")
        f.write(f"  Avg entropy: {test_metrics['avg_entropy']:.4f}\n")
    
    print(f"Training summary saved to {summary_path}")
    
    # Success criteria check
    print(f"\n" + "=" * 70)
    print("Success Criteria Check")
    print("=" * 70)
    
    success = True
    criteria = [
        ("Training converged", history['loss'][-1] < history['loss'][0] * 0.5, True),
        ("Validation accuracy > 80%", val_metrics['accuracy'] > 0.80, True),
        ("Test accuracy > 75%", test_metrics['accuracy'] > 0.75, True),
        ("All types > 70% accuracy", all(acc > 0.70 for acc in test_metrics['per_class_accuracy'].values()), True)
    ]
    
    for criterion, result, required in criteria:
        status = "✓ PASS" if result else ("✗ FAIL" if required else "⚠ WARN")
        print(f"  {status}: {criterion}")
        if required and not result:
            success = False
    
    if success:
        print(f"\n🎉 Training successful! All criteria met.")
    else:
        print(f"\n⚠️  Training completed but some criteria not met.")
        print(f"   Consider: increasing epochs, adjusting learning rate, or collecting more data")
    
    print(f"\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review training curves: logs/training_history.png")
    print(f"2. Review confusion matrix: logs/confusion_matrix.png")
    print(f"3. Load model in your agent:")
    print(f"   >>> from bayesian_beliefs_v2 import BayesianBeliefModule")
    print(f"   >>> module = BayesianBeliefModule.load_model('{model_path}')")
    print(f"4. Run integration tests with CognitiveAgent")


if __name__ == "__main__":
    main()
