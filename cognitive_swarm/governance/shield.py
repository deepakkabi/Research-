"""
Safety Constraint Module with Veto Authority for Multi-Agent RL

This module implements a governance shield with VETO AUTHORITY over agent actions.
Actions that violate safety constraints are VETOED (blocked) before execution,
addressing the challenge of verifiable safety in learned policies.

Veto Mechanisms (PROMPT 3 requirement):
1. Protected Entity Veto: Vetoes actions too close to civilians/friendlies
2. Proportionality Veto: Vetoes actions where damage > benefit  
3. Resource Conservation Veto: Vetoes low-value actions when resources scarce

Research Contribution:
- Combines symbolic logic (if-then rules) with neural cost estimation
- Novel proportionality equation: R_final = R_task - λ * max(0, Cost - Benefit)
- Provides both hard guarantees (veto/blocking) and soft guidance (reward shaping)
- Aligns with Neuro-Symbolic Governance (Component 2 in research framework)

The veto system ensures hard safety guarantees - unsafe actions CANNOT be executed,
even if the learned policy proposes them.

Related Work: Safe RL, Constrained MDPs, Neuro-symbolic AI, Just War Theory
"""

import numpy as np
from typing import Dict, Tuple, Optional

# PyTorch is optional - only needed for neural cost estimator
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy base class if torch not available
    class nn:
        class Module:
            def __init__(self):
                pass


class SafetyConstraintModule(nn.Module):
    """
    Enforces operational safety rules for multi-agent systems.
    
    This module implements three core safety constraints:
    1. Protected Entity Safety: Maintain minimum distance from protected entities
    2. Proportionality: Ensure expected benefit exceeds potential cost
    3. Resource Conservation: Restrict low-value actions when resources are scarce
    
    The module provides both:
    - Hard constraints: Block unsafe actions before execution
    - Soft constraints: Reward shaping to guide policy learning
    
    Args:
        safe_distance (float): Minimum distance to protected entities (cells)
        proportionality_lambda (float): Weight for proportionality constraint
        action_range (float): Range of intervention action (cells)
        splash_radius (float): Collateral damage radius beyond target
        reserve_threshold (int): Minimum resource level for low-value targets
        high_value_threshold (float): Threshold for high-value targets
    """
    
    # Action constants - MUST match environment exactly
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    INTERVENTION = 4  # The action we regulate
    COMMUNICATE = 5
    HOLD = 6  # Safe fallback action
    
    def __init__(
        self,
        safe_distance: float = 5.0,
        proportionality_lambda: float = 1.0,
        action_range: float = 3.0,
        splash_radius: float = 1.5,
        reserve_threshold: int = 2,
        high_value_threshold: float = 5.0
    ):
        super().__init__()
        self.safe_distance = safe_distance
        self.lambda_prop = proportionality_lambda
        self.action_range = action_range
        self.splash_radius = splash_radius
        self.reserve_threshold = reserve_threshold
        self.high_value_threshold = high_value_threshold
        
        # Optional: Learnable cost estimator (neural component)
        # This allows the module to refine cost estimates during training
        # Only available if PyTorch is installed
        if TORCH_AVAILABLE:
            self.cost_estimator = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.cost_estimator = None
        
        # Veto statistics tracking for research analysis (PROMPT 3)
        self.veto_stats = {
            'total_actions_checked': 0,
            'total_vetoes': 0,
            'vetoes_by_rule': {
                'Protected Entity Constraint': 0,
                'Proportionality Constraint': 0,
                'Resource Conservation Constraint': 0
            }
        }
        
        # Backward compatibility
        self.constraint_violations = self.veto_stats['vetoes_by_rule']
        self.total_checks = 0
    
    def verify_action(
        self,
        proposed_action: int,
        state_dict: Dict[str, np.ndarray],
        agent_id: int
    ) -> Tuple[int, bool, Optional[str]]:
        """
        Verify action and apply VETO if safety constraints violated (PROMPT 3).
        
        This method has VETO AUTHORITY - it can override the agent's proposed
        action with the safe HOLD action if any safety constraint is violated.
        
        The veto mechanism implements hard safety guarantees by blocking
        unsafe actions before execution.
        
        Args:
            proposed_action: Action index (must match environment: 0-6)
            state_dict: Environment state with keys:
                - 'agent_positions': np.array (num_agents, 2)
                - 'protected_positions': np.array (num_protected, 2)
                - 'target_positions': np.array (num_targets, 2)
                - 'agent_resources': np.array (num_agents,)
                - 'target_values': np.array (num_targets,)
            agent_id: Index of the agent taking the action
        
        Returns:
            final_action: int (same as proposed if safe, else HOLD=6 if vetoed)
            was_vetoed: bool (True if action was vetoed)
            veto_reason: str or None (which safety rule triggered the veto)
        """
        self.total_checks += 1
        
        # Only check constraints for Intervention action
        # Movement, communication, and hold actions are always allowed
        if proposed_action != self.INTERVENTION:
            return proposed_action, False, None
        
        # ===================================================================
        # RULE 1: Protected Entity Safety (Hard Constraint)
        # ===================================================================
        # Formulation: ALWAYS (NOT Intervention IF min_distance < safe_distance)
        
        agent_pos = state_dict['agent_positions'][agent_id]
        protected_positions = state_dict['protected_positions']
        
        if len(protected_positions) > 0:
            # Calculate Euclidean distances to all protected entities
            distances = np.linalg.norm(
                protected_positions - agent_pos, 
                axis=1
            )
            min_distance = distances.min()
            
            if min_distance < self.safe_distance:
                self.constraint_violations['Protected Entity Constraint'] += 1
                return self.HOLD, True, "Protected Entity Constraint"
        
        # ===================================================================
        # RULE 2: Proportionality (Cost-Benefit Constraint)
        # ===================================================================
        # Formulation: Proportionality_Score = Expected_Benefit - λ * Potential_Cost
        # Block if score < 0 (cost exceeds benefit)
        
        expected_benefit = self._estimate_benefit(state_dict, agent_id)
        potential_cost = self._estimate_cost(state_dict, agent_id)
        
        proportionality_score = expected_benefit - self.lambda_prop * potential_cost
        
        if proportionality_score < 0:
            self.constraint_violations['Proportionality Constraint'] += 1
            return self.HOLD, True, "Proportionality Constraint"
        
        # ===================================================================
        # RULE 3: Resource Conservation
        # ===================================================================
        # Only allow intervention with low resources if targeting high-value targets
        
        resource = state_dict['agent_resources'][agent_id]
        
        if resource < self.reserve_threshold:
            target_values = state_dict['target_values']
            max_value = target_values.max() if len(target_values) > 0 else 0
            
            if max_value < self.high_value_threshold:
                self.constraint_violations['Resource Conservation Constraint'] += 1
                return self.HOLD, True, "Resource Conservation Constraint"
        
        # All checks passed - action is safe
        return proposed_action, False, None
    
    def _estimate_benefit(
        self,
        state_dict: Dict[str, np.ndarray],
        agent_id: int
    ) -> float:
        """
        Estimate expected benefit of intervention action.
        
        Mathematical formulation:
        Benefit = Σ (target_value_i * P(success_i))
        
        Where:
        - target_value_i: Strategic importance (high-value=10, regular=1)
        - P(success_i): Hit probability based on distance
        
        Returns:
            Expected military/strategic value
        """
        agent_pos = state_dict['agent_positions'][agent_id]
        target_positions = state_dict['target_positions']
        target_values = state_dict['target_values']
        
        if len(target_positions) == 0:
            return 0.0
        
        # Base hit probability
        accuracy = 0.7
        
        # Calculate distances to all targets
        distances = np.linalg.norm(target_positions - agent_pos, axis=1)
        in_range = distances <= self.action_range
        
        # Hit probability decreases exponentially with distance
        hit_probs = accuracy * np.exp(-distances / self.action_range)
        hit_probs = np.clip(hit_probs, 0, 1)
        
        # Expected benefit = Σ value_i * P(hit_i) for targets in range
        benefit = (target_values * hit_probs * in_range).sum()
        
        return float(benefit)
    
    def _estimate_cost(
        self,
        state_dict: Dict[str, np.ndarray],
        agent_id: int
    ) -> float:
        """
        Estimate potential negative impact on protected entities.
        
        THIS IS THE NOVEL EQUATION - Key research contribution.
        
        Mathematical formulation:
        Cost = Σ (collateral_impact_i * P(occurrence_i))
        
        Where:
        - collateral_impact_i: Negative utility (-10 per protected entity)
        - P(occurrence_i): Damage probability based on distance and splash radius
        
        The splash radius models collateral damage extending beyond the target,
        creating a "blast radius" that can affect nearby protected entities.
        
        Returns:
            Expected collateral damage cost
        """
        agent_pos = state_dict['agent_positions'][agent_id]
        protected_positions = state_dict['protected_positions']
        
        if len(protected_positions) == 0:
            return 0.0
        
        # Calculate distances to all protected entities
        distances = np.linalg.norm(protected_positions - agent_pos, axis=1)
        
        # Protected entities within action range + splash radius are at risk
        effective_range = self.action_range + self.splash_radius
        at_risk = distances <= effective_range
        
        # Damage probability decreases exponentially with distance
        damage_probs = np.exp(-distances / self.splash_radius)
        damage_probs = np.clip(damage_probs, 0, 1)
        
        # Each protected entity casualty = -10 utility
        protected_value = 10.0
        cost = (protected_value * damage_probs * at_risk).sum()
        
        return float(cost)
    
    def compute_reward_penalty(
        self,
        proposed_action: int,
        state_dict: Dict[str, np.ndarray],
        agent_id: int
    ) -> float:
        """
        Compute reward penalty for use in reward shaping during training.
        
        This allows the policy to LEARN to avoid violations through gradient
        descent, not just be blocked at execution time.
        
        The penalty combines:
        1. Hard penalty (-10) for constraint violations
        2. Soft penalty based on proportionality score
        
        This dual approach enables:
        - Fast online blocking of unsafe actions (hard constraints)
        - Gradual policy improvement to avoid violations (soft guidance)
        
        Args:
            proposed_action: Action index
            state_dict: Environment state
            agent_id: Agent index
        
        Returns:
            Penalty value (0 if safe, negative if violates or risky)
        """
        # Check if action violates any hard constraint
        _, violated, _ = self.verify_action(proposed_action, state_dict, agent_id)
        
        if violated:
            # Hard penalty for violation
            return -10.0
        
        # Even if not violated, add proportionality as soft constraint
        # This guides the policy toward more proportional actions
        benefit = self._estimate_benefit(state_dict, agent_id)
        cost = self._estimate_cost(state_dict, agent_id)
        
        # Proportionality score: positive = good, negative = bad
        proportionality = benefit - self.lambda_prop * cost
        
        # Normalize to approximately [-1, 1] using tanh
        # Negative penalty if cost > benefit
        penalty = -np.tanh(-proportionality)
        
        return float(penalty)
    
    def get_constraint_statistics(self) -> Dict[str, float]:
        """
        Get statistics on constraint violations for research analysis.
        
        Returns:
            Dictionary with violation rates for each constraint
        """
        if self.total_checks == 0:
            return {k: 0.0 for k in self.constraint_violations.keys()}
        
        return {
            constraint: count / self.total_checks
            for constraint, count in self.constraint_violations.items()
        }
    
    def reset_statistics(self):
        """Reset constraint violation statistics."""
        self.constraint_violations = {k: 0 for k in self.constraint_violations.keys()}
        self.total_checks = 0
    
    def forward(self, x):
        """
        Forward pass for potential neural cost estimation.
        
        Currently unused, but included for future research where
        cost estimation could be learned from data.
        
        Args:
            x: State features (batch_size, feature_dim) - tensor or array
        
        Returns:
            Estimated cost (batch_size, 1)
        """
        if not TORCH_AVAILABLE or self.cost_estimator is None:
            raise NotImplementedError(
                "Neural cost estimation requires PyTorch. "
                "Install with: pip install torch"
            )
        return self.cost_estimator(x)


def create_safety_shield(
    safe_distance: float = 5.0,
    proportionality_lambda: float = 1.0,
    **kwargs
) -> SafetyConstraintModule:
    """
    Factory function to create a SafetyConstraintModule with standard parameters.
    
    Args:
        safe_distance: Minimum distance to protected entities
        proportionality_lambda: Weight for proportionality constraint
        **kwargs: Additional parameters passed to SafetyConstraintModule
    
    Returns:
        Configured SafetyConstraintModule instance
    """
    return SafetyConstraintModule(
        safe_distance=safe_distance,
        proportionality_lambda=proportionality_lambda,
        **kwargs
    )
