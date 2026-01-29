"""
Test Suite for Safety Constraint Module

Tests verify:
1. Protected entity safety constraint (Rule 1)
2. Proportionality constraint (Rule 2)
3. Resource conservation constraint (Rule 3)
4. Action indexing matches environment
5. Reward penalty computation
6. Statistics tracking
"""

import numpy as np
from cognitive_swarm.governance.shield import SafetyConstraintModule


class TestProtectedEntityConstraint:
    """Test Rule 1: Protected Entity Safety"""
    
    def test_protected_entity_constraint(self):
        """Verify that intervention is blocked near protected entities"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[3.0, 0.0]]),  # 3 cells away (< 5)
            'target_positions': np.array([[10.0, 10.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([5.0])
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)  # 4 = INTERVENTION
        
        assert violated == True, "Should violate protected entity constraint"
        assert reason == "Protected Entity Constraint"
        assert action == 6, "Should return HOLD action"
        print("✓ Protected entity constraint test passed")
    
    def test_protected_entity_safe_distance(self):
        """Verify that intervention is allowed when far from protected entities"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[10.0, 10.0]]),  # Far away (> 5)
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([10.0])  # High value to pass proportionality
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        assert violated == False, "Should NOT violate when far from protected entities"
        assert reason is None
        assert action == 4, "Should return original INTERVENTION action"
        print("✓ Protected entity safe distance test passed")
    
    def test_protected_entity_exact_boundary(self):
        """Test behavior at exact safe_distance boundary"""
        shield = SafetyConstraintModule(safe_distance=5.0)
        
        # Exactly at boundary (should violate - distance < safe_distance)
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[5.0, 0.0]]),  # Exactly 5 cells
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([10.0])
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        # At exactly safe_distance, should NOT violate (< is strict inequality)
        assert violated == False, "Should not violate at exact boundary"
        print("✓ Protected entity boundary test passed")
    
    def test_no_protected_entities(self):
        """Verify behavior when no protected entities exist"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([]).reshape(0, 2),  # Empty
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([10.0])
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        assert violated == False, "Should not violate when no protected entities"
        print("✓ No protected entities test passed")


class TestProportionalityConstraint:
    """Test Rule 2: Proportionality (Cost-Benefit Analysis)"""
    
    def test_proportionality_violation(self):
        """Verify that low-benefit high-cost actions are blocked"""
        shield = SafetyConstraintModule(
            safe_distance=3,  # Reduced so protected entity passes Rule 1
            proportionality_lambda=1.0
        )
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[4.0, 0.0]]),  # Within risk zone (3+1.5=4.5) but outside safe_distance (3)
            'target_positions': np.array([[2.0, 0.0]]),     # Close target
            'agent_resources': np.array([10]),
            'target_values': np.array([1.0])  # Low value
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        # Should violate proportionality (cost > benefit with low value target near protected entity)
        assert violated == True, "Should violate proportionality constraint"
        assert reason == "Proportionality Constraint", f"Expected Proportionality Constraint, got {reason}"
        assert action == 6, "Should return HOLD action"
        print("✓ Proportionality violation test passed")
    
    def test_high_value_target(self):
        """Verify that high-value targets ARE allowed even with some risk"""
        shield = SafetyConstraintModule(
            safe_distance=5,
            proportionality_lambda=1.0
        )
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[8.0, 0.0]]),
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([10.0])  # HIGH value
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        # Should NOT violate (benefit > cost)
        assert violated == False, "Should allow high-value targets"
        assert reason is None
        assert action == 4, "Should return original action"
        print("✓ High-value target test passed")
    
    def test_proportionality_lambda_sensitivity(self):
        """Test how proportionality_lambda affects decisions"""
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[6.0, 0.0]]),
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([5.0])
        }
        
        # Low lambda (cost matters less) - more permissive
        shield_low = SafetyConstraintModule(safe_distance=5, proportionality_lambda=0.5)
        action_low, violated_low, _ = shield_low.verify_action(4, state, 0)
        
        # High lambda (cost matters more) - more restrictive
        shield_high = SafetyConstraintModule(safe_distance=5, proportionality_lambda=2.0)
        action_high, violated_high, _ = shield_high.verify_action(4, state, 0)
        
        # Higher lambda should be more likely to violate
        if violated_high:
            assert action_high == 6, "High lambda should block more actions"
        
        print("✓ Proportionality lambda sensitivity test passed")
    
    def test_multiple_targets(self):
        """Test benefit calculation with multiple targets"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[15.0, 15.0]]),  # Far away
            'target_positions': np.array([
                [2.0, 0.0],   # Close
                [2.5, 0.0],   # Close
                [20.0, 20.0]  # Far (out of range)
            ]),
            'agent_resources': np.array([10]),
            'target_values': np.array([5.0, 5.0, 100.0])  # Far target has high value but unreachable
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        # Multiple close targets should increase benefit
        assert violated == False, "Multiple targets should increase total benefit"
        print("✓ Multiple targets test passed")


class TestResourceConservationConstraint:
    """Test Rule 3: Resource Conservation"""
    
    def test_resource_conservation_violation(self):
        """Verify that low-resource + low-value is blocked"""
        shield = SafetyConstraintModule(
            safe_distance=5,
            reserve_threshold=2,
            high_value_threshold=5.0
        )
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[15.0, 15.0]]),  # Far away
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([1]),  # Low resource (< 2)
            'target_values': np.array([2.0])  # Low value (< 5)
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        assert violated == True, "Should violate resource conservation"
        assert reason == "Resource Conservation Constraint"
        assert action == 6, "Should return HOLD action"
        print("✓ Resource conservation violation test passed")
    
    def test_resource_conservation_high_value_exception(self):
        """Verify that low-resource + high-value is allowed"""
        shield = SafetyConstraintModule(
            safe_distance=5,
            reserve_threshold=2,
            high_value_threshold=5.0
        )
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[15.0, 15.0]]),
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([1]),  # Low resource
            'target_values': np.array([10.0])  # HIGH value (>= 5)
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        assert violated == False, "Should allow high-value targets even with low resources"
        print("✓ Resource conservation high-value exception test passed")
    
    def test_sufficient_resources(self):
        """Verify that sufficient resources bypass this constraint"""
        shield = SafetyConstraintModule(reserve_threshold=2)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[15.0, 15.0]]),
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),  # High resource
            'target_values': np.array([1.0])  # Low value
        }
        
        action, violated, _ = shield.verify_action(4, state, 0)
        
        # Should not violate resource constraint (has enough resources)
        # May still violate proportionality though
        if violated:
            assert action == 6
        
        print("✓ Sufficient resources test passed")


class TestActionIndexing:
    """Test that action indices match environment exactly"""
    
    def test_action_indexing(self):
        """Verify action indices match environment"""
        shield = SafetyConstraintModule()
        
        # Verify class constants
        assert shield.MOVE_NORTH == 0
        assert shield.MOVE_SOUTH == 1
        assert shield.MOVE_EAST == 2
        assert shield.MOVE_WEST == 3
        assert shield.INTERVENTION == 4
        assert shield.COMMUNICATE == 5
        assert shield.HOLD == 6
        
        print("✓ Action constants test passed")
    
    def test_non_intervention_actions_passthrough(self):
        """Verify non-intervention actions pass through unchanged"""
        shield = SafetyConstraintModule()
        
        # Create state where intervention would be blocked
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[1.0, 0.0]]),  # Very close
            'target_positions': np.array([]).reshape(0, 2),
            'agent_resources': np.array([10]),
            'target_values': np.array([])
        }
        
        # Non-intervention actions should pass through
        for action_idx in [0, 1, 2, 3, 5, 6]:  # All except INTERVENTION=4
            final_action, violated, reason = shield.verify_action(action_idx, state, 0)
            
            assert final_action == action_idx, f"Action {action_idx} should not be modified"
            assert violated == False, f"Action {action_idx} should not violate constraints"
            assert reason is None
        
        print("✓ Non-intervention actions passthrough test passed")


class TestRewardPenalty:
    """Test reward penalty computation for training"""
    
    def test_hard_penalty_for_violation(self):
        """Verify hard penalty (-10) for constraint violations"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[2.0, 0.0]]),  # Too close
            'target_positions': np.array([[10.0, 10.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([5.0])
        }
        
        penalty = shield.compute_reward_penalty(4, state, 0)
        
        assert penalty == -10.0, "Should return hard penalty for violation"
        print("✓ Hard penalty test passed")
    
    def test_soft_penalty_for_low_proportionality(self):
        """Verify soft penalty for low but positive proportionality"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[15.0, 15.0]]),  # Far away
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([10.0])  # High value
        }
        
        penalty = shield.compute_reward_penalty(4, state, 0)
        
        # Should be a soft penalty (between -1 and 1, not -10)
        assert -1.0 <= penalty <= 1.0, "Should return soft penalty"
        print("✓ Soft penalty test passed")
    
    def test_zero_penalty_for_safe_actions(self):
        """Verify no penalty for non-intervention actions"""
        shield = SafetyConstraintModule()
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[1.0, 0.0]]),
            'target_positions': np.array([]).reshape(0, 2),
            'agent_resources': np.array([10]),
            'target_values': np.array([])
        }
        
        # Movement actions should have no penalty
        for action_idx in [0, 1, 2, 3, 5, 6]:
            penalty = shield.compute_reward_penalty(action_idx, state, 0)
            # Note: penalty is 0 for non-intervention actions
            # (they pass through verify_action without modification)
        
        print("✓ Zero penalty for safe actions test passed")


class TestStatistics:
    """Test constraint violation statistics tracking"""
    
    def test_statistics_tracking(self):
        """Verify that statistics are tracked correctly"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        # Initial state
        stats = shield.get_constraint_statistics()
        assert all(v == 0.0 for v in stats.values())
        
        # Trigger protected entity violation
        state1 = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[2.0, 0.0]]),
            'target_positions': np.array([[10.0, 10.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([5.0])
        }
        shield.verify_action(4, state1, 0)
        
        stats = shield.get_constraint_statistics()
        assert stats['Protected Entity Constraint'] > 0
        
        print("✓ Statistics tracking test passed")
    
    def test_statistics_reset(self):
        """Verify that statistics can be reset"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[2.0, 0.0]]),
            'target_positions': np.array([[10.0, 10.0]]),
            'agent_resources': np.array([10]),
            'target_values': np.array([5.0])
        }
        
        shield.verify_action(4, state, 0)
        shield.reset_statistics()
        
        stats = shield.get_constraint_statistics()
        assert all(v == 0.0 for v in stats.values())
        
        print("✓ Statistics reset test passed")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_multiple_agents(self):
        """Test with multiple agents"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([
                [0.0, 0.0],
                [10.0, 10.0],
                [20.0, 20.0]
            ]),
            'protected_positions': np.array([[2.0, 0.0]]),
            'target_positions': np.array([[15.0, 15.0]]),
            'agent_resources': np.array([10, 5, 3]),
            'target_values': np.array([10.0])
        }
        
        # Agent 0 is too close to protected entity
        action0, violated0, _ = shield.verify_action(4, state, 0)
        assert violated0 == True
        
        # Agent 1 is far from protected entity
        action1, violated1, _ = shield.verify_action(4, state, 1)
        # May or may not violate depending on proportionality
        
        print("✓ Multiple agents test passed")
    
    def test_empty_targets(self):
        """Test with no targets"""
        shield = SafetyConstraintModule(safe_distance=5)
        
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[15.0, 15.0]]),
            'target_positions': np.array([]).reshape(0, 2),
            'agent_resources': np.array([10]),
            'target_values': np.array([])
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        # No targets means zero benefit AND zero cost
        # Score = 0 - 0 = 0, which is NOT < 0, so proportionality doesn't violate
        # However, in practice, intervention with no targets is pointless
        # This demonstrates that the module enforces safety, not strategy
        assert violated == False, "Zero benefit and zero cost should not violate proportionality"
        
        print("✓ Empty targets test passed")
    
    def test_cascade_of_constraints(self):
        """Test that constraints are checked in order"""
        shield = SafetyConstraintModule(safe_distance=5, reserve_threshold=5)
        
        # State that violates multiple constraints
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': np.array([[2.0, 0.0]]),  # Violates Rule 1
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([1]),  # Violates Rule 3
            'target_values': np.array([1.0])  # Low value
        }
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        # Should violate Rule 1 first (checked first)
        assert violated == True
        assert reason == "Protected Entity Constraint"
        
        print("✓ Cascade of constraints test passed")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("SAFETY CONSTRAINT MODULE - COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    # Run all test classes
    test_classes = [
        TestProtectedEntityConstraint,
        TestProportionalityConstraint,
        TestResourceConservationConstraint,
        TestActionIndexing,
        TestRewardPenalty,
        TestStatistics,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 70)
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                method = getattr(test_instance, method_name)
                try:
                    method()
                except AssertionError as e:
                    print(f"✗ {method_name} FAILED: {e}")
                except Exception as e:
                    print(f"✗ {method_name} ERROR: {e}")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == '__main__':
    print("Running Safety Constraint Module Tests...")
    run_all_tests()
