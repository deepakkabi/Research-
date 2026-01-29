"""
Integration Example: Safety Shield in Action

This example demonstrates how the SafetyConstraintModule integrates with
the Cognitive Swarm framework to enforce safety constraints.
"""

import numpy as np
from cognitive_swarm.governance import SafetyConstraintModule


def demonstrate_protected_entity_constraint():
    """Example 1: Protected Entity Safety Constraint"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Protected Entity Safety Constraint")
    print("="*70)
    
    shield = SafetyConstraintModule(safe_distance=5.0)
    
    # Scenario: Agent too close to protected entity
    state = {
        'agent_positions': np.array([[0.0, 0.0]]),
        'protected_positions': np.array([[3.0, 0.0]]),  # Only 3 cells away!
        'target_positions': np.array([[10.0, 10.0]]),
        'agent_resources': np.array([10]),
        'target_values': np.array([5.0])
    }
    
    print("\nScenario: Agent at [0,0], Protected entity at [3,0] (distance: 3 cells)")
    print(f"Safe distance threshold: {shield.safe_distance} cells")
    print("Proposed action: INTERVENTION (4)")
    
    action, violated, reason = shield.verify_action(4, state, 0)
    
    print(f"\n✗ Action BLOCKED!")
    print(f"  Reason: {reason}")
    print(f"  Returned action: {action} (HOLD)")
    print("\nExplanation: Too close to protected entity. Action would risk harm.")


def demonstrate_proportionality_constraint():
    """Example 2: Proportionality Constraint"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Proportionality Constraint")
    print("="*70)
    
    shield = SafetyConstraintModule(
        safe_distance=3.0,
        proportionality_lambda=1.0
    )
    
    # Scenario: Low-value target near protected entity
    state = {
        'agent_positions': np.array([[0.0, 0.0]]),
        'protected_positions': np.array([[4.0, 0.0]]),  # Within risk zone
        'target_positions': np.array([[2.0, 0.0]]),
        'agent_resources': np.array([10]),
        'target_values': np.array([1.0])  # Low value
    }
    
    print("\nScenario: Low-value target (value: 1.0) near protected entity")
    print("Agent at [0,0], Target at [2,0], Protected at [4,0]")
    
    benefit = shield._estimate_benefit(state, 0)
    cost = shield._estimate_cost(state, 0)
    score = benefit - shield.lambda_prop * cost
    
    print(f"\nCost-Benefit Analysis:")
    print(f"  Expected Benefit: {benefit:.2f}")
    print(f"  Potential Cost: {cost:.2f}")
    print(f"  Proportionality Score: {score:.2f}")
    print(f"  Threshold: 0.0")
    
    action, violated, reason = shield.verify_action(4, state, 0)
    
    print(f"\n✗ Action BLOCKED!")
    print(f"  Reason: {reason}")
    print("\nExplanation: Potential cost exceeds expected benefit.")


def demonstrate_high_value_target():
    """Example 3: High-Value Target Exception"""
    print("\n" + "="*70)
    print("EXAMPLE 3: High-Value Target (Action Allowed)")
    print("="*70)
    
    shield = SafetyConstraintModule(
        safe_distance=3.0,
        proportionality_lambda=1.0
    )
    
    # Scenario: High-value target with acceptable risk
    state = {
        'agent_positions': np.array([[0.0, 0.0]]),
        'protected_positions': np.array([[8.0, 0.0]]),  # Far away
        'target_positions': np.array([[2.0, 0.0]]),
        'agent_resources': np.array([10]),
        'target_values': np.array([10.0])  # HIGH value
    }
    
    print("\nScenario: High-value target (value: 10.0)")
    print("Agent at [0,0], Target at [2,0], Protected at [8,0]")
    
    benefit = shield._estimate_benefit(state, 0)
    cost = shield._estimate_cost(state, 0)
    score = benefit - shield.lambda_prop * cost
    
    print(f"\nCost-Benefit Analysis:")
    print(f"  Expected Benefit: {benefit:.2f}")
    print(f"  Potential Cost: {cost:.2f}")
    print(f"  Proportionality Score: {score:.2f}")
    print(f"  Threshold: 0.0")
    
    action, violated, reason = shield.verify_action(4, state, 0)
    
    print(f"\n✓ Action ALLOWED!")
    print(f"  Returned action: {action} (INTERVENTION)")
    print("\nExplanation: High-value target justifies minimal risk.")


def demonstrate_resource_conservation():
    """Example 4: Resource Conservation Constraint"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Resource Conservation Constraint")
    print("="*70)
    
    shield = SafetyConstraintModule(
        safe_distance=5.0,
        reserve_threshold=2,
        high_value_threshold=5.0
    )
    
    # Scenario: Low resources + low-value target
    state = {
        'agent_positions': np.array([[0.0, 0.0]]),
        'protected_positions': np.array([[15.0, 15.0]]),  # Far away
        'target_positions': np.array([[2.0, 0.0]]),
        'agent_resources': np.array([1]),  # LOW resources
        'target_values': np.array([2.0])  # Low value
    }
    
    print("\nScenario: Agent has only 1 resource remaining")
    print(f"Reserve threshold: {shield.reserve_threshold}")
    print(f"Target value: 2.0 (below high-value threshold: {shield.high_value_threshold})")
    
    action, violated, reason = shield.verify_action(4, state, 0)
    
    print(f"\n✗ Action BLOCKED!")
    print(f"  Reason: {reason}")
    print("\nExplanation: Must preserve resources for high-value opportunities.")


def demonstrate_lambda_sensitivity():
    """Example 5: Proportionality Lambda Sensitivity"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Proportionality Lambda Sensitivity")
    print("="*70)
    
    state = {
        'agent_positions': np.array([[0.0, 0.0]]),
        'protected_positions': np.array([[4.0, 0.0]]),
        'target_positions': np.array([[2.0, 0.0]]),
        'agent_resources': np.array([10]),
        'target_values': np.array([5.0])
    }
    
    print("\nSame scenario, different risk tolerances:")
    print("Target value: 5.0, Protected entity at moderate risk")
    
    lambdas = [0.5, 1.0, 1.5, 2.0]
    
    for lam in lambdas:
        shield = SafetyConstraintModule(
            safe_distance=3.0,
            proportionality_lambda=lam
        )
        
        benefit = shield._estimate_benefit(state, 0)
        cost = shield._estimate_cost(state, 0)
        score = benefit - lam * cost
        
        action, violated, reason = shield.verify_action(4, state, 0)
        
        status = "✗ BLOCKED" if violated else "✓ ALLOWED"
        print(f"\nλ = {lam:.1f}: Score = {score:.2f} → {status}")


def demonstrate_statistics_tracking():
    """Example 6: Statistics Tracking"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Statistics Tracking")
    print("="*70)
    
    shield = SafetyConstraintModule(safe_distance=5.0)
    
    # Simulate 10 action checks
    scenarios = [
        {'protected_positions': np.array([[2.0, 0.0]]), 'target_values': np.array([5.0])},  # Too close
        {'protected_positions': np.array([[10.0, 0.0]]), 'target_values': np.array([10.0])},  # Safe
        {'protected_positions': np.array([[3.0, 0.0]]), 'target_values': np.array([5.0])},  # Too close
        {'protected_positions': np.array([[15.0, 0.0]]), 'target_values': np.array([8.0])},  # Safe
        {'protected_positions': np.array([[4.0, 0.0]]), 'target_values': np.array([5.0])},  # Too close
    ]
    
    print("\nRunning 5 simulated action checks...")
    
    for i, scenario in enumerate(scenarios):
        state = {
            'agent_positions': np.array([[0.0, 0.0]]),
            'protected_positions': scenario['protected_positions'],
            'target_positions': np.array([[2.0, 0.0]]),
            'agent_resources': np.array([10]),
            'target_values': scenario['target_values']
        }
        action, violated, reason = shield.verify_action(4, state, 0)
    
    stats = shield.get_constraint_statistics()
    
    print(f"\nConstraint Violation Statistics:")
    print(f"  Total checks: {shield.total_checks}")
    for constraint, rate in stats.items():
        print(f"  {constraint}: {rate*100:.1f}%")


def main():
    """Run all demonstration examples"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SAFETY SHIELD INTEGRATION EXAMPLES" + " "*19 + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Demonstrating safety constraints in multi-agent RL systems" + " "*7 + "║")
    print("╚" + "="*68 + "╝")
    
    demonstrate_protected_entity_constraint()
    demonstrate_proportionality_constraint()
    demonstrate_high_value_target()
    demonstrate_resource_conservation()
    demonstrate_lambda_sensitivity()
    demonstrate_statistics_tracking()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The Safety Shield provides:
✓ Hard constraints that block unsafe actions
✓ Interpretable decisions based on clear rules
✓ Tunable parameters for different risk profiles
✓ Comprehensive statistics for research analysis

Integration is simple:
1. Create shield with desired parameters
2. Check each action before execution
3. Use returned safe action
4. Optionally add reward penalties for training

See cognitive_swarm/governance/README.md for full documentation.
    """)


if __name__ == '__main__':
    main()
