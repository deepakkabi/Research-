"""
Test Suite for Bayesian Belief Module V2

Tests cover:
1. V1 backward compatibility (2 types, heuristics)
2. V2 features (4 types, neural networks)
3. Training capability
4. Integration scenarios
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cognitive_swarm.modules.bayesian_beliefs import BayesianBeliefModule


class TestBackwardCompatibility:
    """Test V1 backward compatibility."""
    
    def test_v1_initialization(self):
        """Test V1-style initialization (2 types, heuristics)."""
        module = BayesianBeliefModule(
            obs_dim=10,
            num_types=2,
            use_neural=False,
            update_frequency=5
        )
        
        assert module.num_types == 2
        assert not module.use_neural
        assert module.update_frequency == 5
        assert module.type_names == ['Aggressive', 'Defensive']
        print("✓ V1 initialization test passed")
    
    def test_v1_forward_pass(self):
        """Test that V1 forward pass still works."""
        module = BayesianBeliefModule(num_types=2, use_neural=False)
        
        obs = torch.randn(4, 5, 10)  # batch=4, seq=5
        belief = module(obs)
        
        assert belief.shape == (4, 2)
        assert belief.sum(dim=1).allclose(torch.ones(4), atol=1e-5)
        print("✓ V1 forward pass test passed")
    
    def test_v1_heuristic_inference(self):
        """Test V1 heuristic-based inference still works."""
        module = BayesianBeliefModule(num_types=2, use_neural=False)
        
        # Aggressive observation
        aggressive_obs = torch.zeros(1, 1, 10)
        aggressive_obs[0, 0, 2] = 2.0
        aggressive_obs[0, 0, 3] = 0.9
        aggressive_obs[0, 0, 4] = 0.8
        
        belief = module(aggressive_obs)
        type_idx, _ = module.get_likely_type(belief)
        
        assert type_idx.item() == 0  # Should infer Aggressive
        print("✓ V1 heuristic inference test passed")


class TestV2Initialization:
    """Test V2 initialization and configuration."""
    
    def test_v2_default_initialization(self):
        """Test V2-style initialization (4 types, neural)."""
        module = BayesianBeliefModule()  # Defaults to V2
        
        assert module.num_types == 4
        assert module.use_neural == True
        assert module.update_frequency == 1
        assert len(module.type_names) == 4
        assert module.likelihood_nets is not None
        print("✓ V2 default initialization test passed")
    
    def test_v2_custom_initialization(self):
        """Test V2 with custom parameters."""
        module = BayesianBeliefModule(
            obs_dim=15,
            num_types=4,
            hidden_dim=128,
            use_neural=True
        )
        
        assert module.obs_dim == 15
        assert module.hidden_dim == 128
        assert len(module.likelihood_nets) == 4
        print("✓ V2 custom initialization test passed")
    
    def test_four_type_names(self):
        """Test that 4 type names are correct."""
        module = BayesianBeliefModule(num_types=4)
        
        expected_names = ['Aggressive', 'Defensive', 'Deceptive', 'Reconnaissance']
        assert module.type_names == expected_names
        
        for i, name in enumerate(expected_names):
            assert module.get_type_name(i) == name
        
        print("✓ Four type names test passed")


class TestV2ForwardPass:
    """Test V2 forward pass with neural networks."""
    
    def test_neural_forward_pass(self):
        """Test forward pass with neural networks."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        obs = torch.randn(8, 10, 10)  # batch=8, seq=10
        belief = module(obs)
        
        assert belief.shape == (8, 4)
        assert belief.sum(dim=1).allclose(torch.ones(8), atol=1e-5)
        assert (belief >= 0).all()
        assert (belief <= 1).all()
        print("✓ Neural forward pass test passed")
    
    def test_neural_likelihood_outputs(self):
        """Test that neural likelihoods produce valid outputs."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        module.eval()
        
        obs = torch.randn(5, 10)
        
        for type_idx in range(4):
            likelihood = module._compute_likelihood_neural(obs, type_idx)
            
            assert likelihood.shape == (5,)
            assert (likelihood > 0).all()  # Softplus ensures positive
        
        print("✓ Neural likelihood outputs test passed")
    
    def test_batch_processing_v2(self):
        """Test batch processing with 4 types."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        batch_sizes = [1, 4, 16, 32]
        for batch_size in batch_sizes:
            obs = torch.randn(batch_size, 5, 10)
            belief = module(obs)
            
            assert belief.shape == (batch_size, 4)
            assert belief.sum(dim=1).allclose(torch.ones(batch_size), atol=1e-5)
        
        print("✓ Batch processing V2 test passed")


class TestTraining:
    """Test training capability."""
    
    def test_training_basic(self):
        """Test basic training loop."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        # Generate synthetic training data
        train_obs = torch.randn(200, 10)
        train_labels = torch.randint(0, 4, (200,))
        
        optimizer = optim.Adam(module.parameters(), lr=0.01)
        
        # Train for a few epochs
        history = module.train_likelihood(
            train_obs, train_labels, optimizer, epochs=5, verbose=False
        )
        
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 5
        assert len(history['accuracy']) == 5
        
        print("✓ Basic training test passed")
    
    def test_training_convergence(self):
        """Test that training reduces loss."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        # Generate separable synthetic data
        train_obs = torch.zeros(400, 10)
        train_labels = torch.zeros(400, dtype=torch.long)
        
        # Create clearly separable data
        for i in range(4):
            start_idx = i * 100
            end_idx = (i + 1) * 100
            
            train_obs[start_idx:end_idx, 2] = 5.0 * i  # Different distances
            train_obs[start_idx:end_idx, 3] = 0.2 * (i + 1)  # Different speeds
            train_labels[start_idx:end_idx] = i
        
        optimizer = optim.Adam(module.parameters(), lr=0.01)
        
        history = module.train_likelihood(
            train_obs, train_labels, optimizer, epochs=20, verbose=False
        )
        
        # Loss should decrease
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        
        assert final_loss < initial_loss
        assert history['accuracy'][-1] > 0.5  # Should learn something
        
        print(f"✓ Training convergence test passed (loss: {initial_loss:.3f} → {final_loss:.3f})")
    
    def test_training_with_heuristics_fails(self):
        """Test that training fails when use_neural=False."""
        module = BayesianBeliefModule(num_types=2, use_neural=False)
        
        train_obs = torch.randn(100, 10)
        train_labels = torch.randint(0, 2, (100,))
        
        with pytest.raises(ValueError, match="Cannot train"):
            module.train_likelihood(train_obs, train_labels, None, epochs=1)
        
        print("✓ Training with heuristics fails correctly")


class TestValidation:
    """Test validation functionality."""
    
    def test_validation_metrics(self):
        """Test that validation returns correct metrics."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        val_obs = torch.randn(100, 10)
        val_labels = torch.randint(0, 4, (100,))
        
        metrics = module.validate(val_obs, val_labels, verbose=False)
        
        assert 'accuracy' in metrics
        assert 'per_class_accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert len(metrics['per_class_accuracy']) == 4
        
        print("✓ Validation metrics test passed")
    
    def test_validation_per_class(self):
        """Test per-class accuracy computation."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        # Create perfectly separable data
        val_obs = torch.zeros(40, 10)
        val_labels = torch.zeros(40, dtype=torch.long)
        
        for i in range(4):
            start = i * 10
            end = (i + 1) * 10
            val_obs[start:end, 2] = 10.0 * i  # Unique feature per class
            val_labels[start:end] = i
        
        # Train briefly
        optimizer = optim.Adam(module.parameters(), lr=0.1)
        module.train_likelihood(val_obs, val_labels, optimizer, epochs=50, verbose=False)
        
        # Validate
        metrics = module.validate(val_obs, val_labels, verbose=False)
        
        # Should achieve high accuracy on this simple data
        assert metrics['accuracy'] > 0.7
        
        print(f"✓ Per-class validation test passed (acc: {metrics['accuracy']:.2%})")


class TestSaveLoad:
    """Test model saving and loading."""
    
    def test_save_and_load(self):
        """Test saving and loading model."""
        module_original = BayesianBeliefModule(num_types=4, use_neural=True)
        
        # Train briefly to get non-random weights
        train_obs = torch.randn(50, 10)
        train_labels = torch.randint(0, 4, (50,))
        optimizer = optim.Adam(module_original.parameters(), lr=0.01)
        module_original.train_likelihood(train_obs, train_labels, optimizer, epochs=5, verbose=False)
        
        # Save
        save_path = '/tmp/test_belief_v2.pt'
        module_original.save_model(save_path)
        
        # Load
        module_loaded = BayesianBeliefModule.load_model(save_path)
        
        # Verify same configuration
        assert module_loaded.num_types == module_original.num_types
        assert module_loaded.obs_dim == module_original.obs_dim
        assert module_loaded.hidden_dim == module_original.hidden_dim
        
        # Verify same outputs
        test_obs = torch.randn(5, 10)
        belief_original = module_original(test_obs)
        belief_loaded = module_loaded(test_obs)
        
        assert torch.allclose(belief_original, belief_loaded, atol=1e-5)
        
        print("✓ Save and load test passed")
    
    def test_loaded_model_is_eval_mode(self):
        """Test that loaded model is in eval mode."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        save_path = '/tmp/test_belief_eval.pt'
        module.save_model(save_path)
        
        loaded_module = BayesianBeliefModule.load_model(save_path)
        
        assert not loaded_module.training  # Should be in eval mode
        
        print("✓ Loaded model eval mode test passed")


class TestIntegrationV2:
    """Test V2 integration scenarios."""
    
    def test_four_type_inference(self):
        """Test inference with 4 distinct types."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        # Create 4 distinct observation patterns
        obs_patterns = torch.zeros(4, 1, 10)
        
        # Type 0 (Aggressive): close, fast
        obs_patterns[0, 0, 2] = 2.0
        obs_patterns[0, 0, 3] = 0.9
        
        # Type 1 (Defensive): far, slow
        obs_patterns[1, 0, 2] = 18.0
        obs_patterns[1, 0, 3] = 0.1
        
        # Type 2 (Deceptive): medium, fast
        obs_patterns[2, 0, 2] = 8.0
        obs_patterns[2, 0, 3] = 0.8
        
        # Type 3 (Reconnaissance): far, fast
        obs_patterns[3, 0, 2] = 12.0
        obs_patterns[3, 0, 3] = 0.9
        
        # Get beliefs
        beliefs = module(obs_patterns)
        
        assert beliefs.shape == (4, 4)
        
        # Each should sum to 1
        for i in range(4):
            assert beliefs[i].sum().item() == pytest.approx(1.0, abs=1e-5)
        
        print("✓ Four type inference test passed")
    
    def test_integration_with_agent_api(self):
        """Test that API matches what CognitiveAgent expects."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        # Simulate agent usage
        obs = torch.randn(4, 10)  # batch of 4 agents
        
        # Update belief (agent's perspective)
        belief = module(obs, prior_belief=None)
        
        # Get Q-values (simulated from agent's policy network)
        num_actions = 7
        q_values = torch.randn(4, num_actions, 4)  # (batch, actions, types)
        
        # Compute expected Q-values
        expected_q = module.compute_expected_value(belief, q_values)
        
        # Select actions
        actions = expected_q.argmax(dim=1)
        
        assert belief.shape == (4, 4)
        assert expected_q.shape == (4, num_actions)
        assert actions.shape == (4,)
        
        print("✓ Integration with agent API test passed")
    
    def test_episodic_usage_v2(self):
        """Test V2 in episodic setting with 4 types."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        module.reset_update_counter()
        
        belief = None
        episode_length = 30
        
        for t in range(episode_length):
            obs = torch.randn(1, 10)
            belief = module(obs, prior_belief=belief)
            
            # Should always be valid
            assert belief.shape == (1, 4)
            assert belief.sum().item() == pytest.approx(1.0, abs=1e-5)
        
        print("✓ Episodic usage V2 test passed")


class TestExpectedValueV2:
    """Test expected value computation with 4 types."""
    
    def test_expected_value_four_types(self):
        """Test expected Q-value with 4 types."""
        module = BayesianBeliefModule(num_types=4, use_neural=True)
        
        # Belief favoring type 0
        belief = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
        
        # Q-values
        num_actions = 7
        q_values = torch.zeros(1, num_actions, 4)
        q_values[0, 0, 0] = 10.0  # Action 0 good for type 0
        q_values[0, 1, 1] = 10.0  # Action 1 good for type 1
        
        expected_q = module.compute_expected_value(belief, q_values)
        
        # Expected Q for action 0: 0.7 * 10.0 = 7.0
        # Expected Q for action 1: 0.1 * 10.0 = 1.0
        
        assert expected_q[0, 0] == pytest.approx(7.0, abs=1e-4)
        assert expected_q[0, 1] == pytest.approx(1.0, abs=1e-4)
        assert expected_q.argmax(dim=1).item() == 0
        
        print("✓ Expected value four types test passed")


def run_all_tests():
    """Run all test classes."""
    print("=" * 70)
    print("RUNNING BAYESIAN BELIEF MODULE V2 TEST SUITE")
    print("=" * 70)
    
    test_classes = [
        TestBackwardCompatibility,
        TestV2Initialization,
        TestV2ForwardPass,
        TestTraining,
        TestValidation,
        TestSaveLoad,
        TestIntegrationV2,
        TestExpectedValueV2
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"Running {test_class.__name__}")
        print('='*70)
        
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nV2 Features Verified:")
        print("  ✓ Backward compatible with V1")
        print("  ✓ 4-type strategy inference")
        print("  ✓ Neural likelihood networks")
        print("  ✓ Training capability")
        print("  ✓ Save/load functionality")
        print("  ✓ Integration with CognitiveAgent API")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed\n")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
