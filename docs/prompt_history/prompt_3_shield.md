# Prompt 3: Shield
PROMPT 3: SAFETY CONSTRAINTS
ACADEMIC RESEARCH DISCLAIMER:
This is for multi-agent reinforcement learning research focused on safety
and constraint satisfaction in learned policies.

---

You are implementing a Safety Constraint Module for multi-agent reinforcement 
learning research. This module enforces operational rules as HARD CONSTRAINTS 
that cannot be violated.

### RESEARCH CONTEXT

Standard RL uses reward shaping: penalize unsafe actions but don't prohibit them. 
This allows violations under optimization pressure.

**Our approach**: Use symbolic logic to create a mathematical "firewall" that 
blocks unsafe actions BEFORE execution.

**Research contribution**: Formalization of "proportionality" constraints as 
differentiable functions.

### OPERATIONAL RULES

You must implement these rules as deterministic logic:

**Rule 1: Protected Entity Safety (Hard Constraint)**
ALWAYS (NOT Intervention IF min_protected_distance < safe_distance)
Where:
- safe_distance = 5 cells (configurable)
- min_protected_distance = closest protected entity to action location
- protected_entities = elements in environment that must not be harmed

**Rule 2: Proportionality (Cost-Benefit Constraint)**
Proportionality_Score = Expected_Benefit - Potential_Cost
IF Proportionality_Score < 0: BLOCK Intervention action

Mathematical formulation:
Benefit = Σ (target_value_i * P(success_i)) Cost = Σ (collateral_impact_i * P(occurrence_i))
Where:
•	target_value_i = strategic importance (high-value = 10, regular = 1)
•	collateral_impact_i = estimated negative impact on protected entities
•	P(success_i) = probability of successful action

**Rule 3: Resource Conservation**
IF resource_remaining < reserve_threshold: ALLOW Intervention ONLY IF high_value_target

### ACTION SPACE INTEGRATION

**CRITICAL**: Must match environment action indices:
```python
MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_EAST = 2
MOVE_WEST = 3
INTERVENTION = 4  # The action we regulate
COMMUNICATE = 5
HOLD = 6  # Safe fallback action
```

### IMPLEMENTATION SPECIFICATION
```python
import torch
import torch.nn as nn
import numpy as np

class SafetyConstraintModule(nn.Module):
    """
    Enforces operational safety rules for multi-agent systems.
    
    Research application: Proving that RL agents can satisfy hard constraints.
    Novel contribution: Proportionality equation formalizing cost-benefit analysis.
    """
    
    def __init__(self, safe_distance=5, proportionality_lambda=1.0):
        """
        Args:
            safe_distance: Minimum distance to protected entities (cells)
            proportionality_lambda: Weight for proportionality constraint
        """
        super().__init__()
        self.safe_distance = safe_distance
        self.lambda_prop = proportionality_lambda
        
        # Optional: Learnable cost estimator (neural component)
        self.cost_estimator = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def verify_action(self, proposed_action, state_dict, agent_id):
        """
        Check if proposed action satisfies constraints.
        
        Args:
            proposed_action: int (action index, must match environment)
            state_dict: dict with keys:
                - 'agent_positions': np.array (num_agents, 2)
                - 'protected_positions': np.array (num_protected, 2)
                - 'target_positions': np.array (num_targets, 2)
                - 'agent_resources': np.array (num_agents,)
                - 'target_values': np.array (num_targets,)
            agent_id: int
        
        Returns:
            final_action: int (same as proposed if safe, else HOLD)
            constraint_violated: bool
            violation_reason: str (which rule was violated)
        """
        # Action constants (MUST match environment)
        MOVE_ACTIONS = [0, 1, 2, 3]
        INTERVENTION_ACTION = 4
        COMMUNICATE_ACTION = 5
        HOLD_ACTION = 6
        
        # Only check constraints for Intervention action
        if proposed_action != INTERVENTION_ACTION:
            return proposed_action, False, None
        
        # === CHECK RULE 1: Protected Entity Safety ===
        agent_pos = state_dict['agent_positions'][agent_id]
        protected_positions = state_dict['protected_positions']
        
        if len(protected_positions) > 0:
            distances = np.linalg.norm(protected_positions - agent_pos, axis=1)
            min_distance = distances.min()
            
            if min_distance < self.safe_distance:
                return HOLD_ACTION, True, "Protected Entity Constraint"
        
        # === CHECK RULE 2: Proportionality ===
        expected_benefit = self._estimate_benefit(state_dict, agent_id)
        potential_cost = self._estimate_cost(state_dict, agent_id)
        
        proportionality_score = expected_benefit - self.lambda_prop * potential_cost
        
        if proportionality_score < 0:
            return HOLD_ACTION, True, "Proportionality Constraint"
        
        # === CHECK RULE 3: Resource Conservation ===
        resource = state_dict['agent_resources'][agent_id]
        reserve_threshold = 2
        
        if resource < reserve_threshold:
            target_values = state_dict['target_values']
            max_value = target_values.max() if len(target_values) > 0 else 0
            
            if max_value < 5:
                return HOLD_ACTION, True, "Resource Conservation Constraint"
        
        # All checks passed
        return proposed_action, False, None
    
    def _estimate_benefit(self, state_dict, agent_id):
        """
        Estimate expected benefit of intervention action.
        
        Returns:
            float: Expected military/strategic value
        """
        agent_pos = state_dict['agent_positions'][agent_id]
        target_positions = state_dict['target_positions']
        target_values = state_dict['target_values']
        
        if len(target_positions) == 0:
            return 0.0
        
        # Simple heuristic: value weighted by hit probability
        action_range = 3  # cells
        accuracy = 0.7  # base hit probability
        
        distances = np.linalg.norm(target_positions - agent_pos, axis=1)
        in_range = distances <= action_range
        
        # Hit probability decreases with distance
        hit_probs = accuracy * np.exp(-distances / action_range)
        hit_probs = np.clip(hit_probs, 0, 1)
        
        # Expected benefit = Σ value_i * P(hit_i)
        benefit = (target_values * hit_probs * in_range).sum()
        return float(benefit)
    
    def _estimate_cost(self, state_dict, agent_id):
        """
        Estimate potential negative impact on protected entities.
        
        THIS IS YOUR NOVEL EQUATION - explain carefully in the paper.
        
        Returns:
            float: Expected collateral damage
        """
        agent_pos = state_dict['agent_positions'][agent_id]
        protected_positions = state_dict['protected_positions']
        
        if len(protected_positions) == 0:
            return 0.0
        
        action_range = 3  # cells
        splash_radius = 1.5  # Collateral damage extends beyond target
        
        distances = np.linalg.norm(protected_positions - agent_pos, axis=1)
        
        # Protected entities within splash radius are at risk
        at_risk = distances <= (action_range + splash_radius)
        
        # Damage probability (decreases with distance)
        damage_probs = np.exp(-distances / splash_radius)
        damage_probs = np.clip(damage_probs, 0, 1)
        
        # Each protected entity casualty = -10 utility
        protected_value = 10
        cost = (protected_value * damage_probs * at_risk).sum()
        
        return float(cost)
    
    def compute_reward_penalty(self, proposed_action, state_dict, agent_id):
        """
        For use in reward shaping during training.
        
        This allows the policy to LEARN to avoid violations, not just be blocked.
        
        Returns:
            penalty: float (0 if action is safe, negative if violates proportionality)
        """
        _, violated, _ = self.verify_action(proposed_action, state_dict, agent_id)
        
        if violated:
            # Hard penalty for violation
            return -10.0
        
        # Even if not violated, add proportionality as soft constraint
        benefit = self._estimate_benefit(state_dict, agent_id)
        cost = self._estimate_cost(state_dict, agent_id)
        
        # Reward proportional actions
        proportionality = benefit - self.lambda_prop * cost
        
        # Normalize to [-1, 1]
        penalty =
-np.tanh(-proportionality) # Negative if cost > benefit
    return float(penalty)

### RESEARCH JUSTIFICATION

This module addresses the challenge of **verifiable safety** in learned policies.

**Key insight**: Combining symbolic rules (if-then logic) with neural cost estimation.

**Novel contribution**: The proportionality equation:
R_final = R_task - λ * max(0, Cost - Benefit)

This formalizes cost-benefit analysis as a differentiable constraint, enabling:
1. Hard guarantees (action blocking)
2. Soft guidance (reward shaping during training)
3. Interpretable safety metrics

**INTEGRATION NOTE**:
This reward penalty MODIFIES the environment reward:
```python
Final reward for training
total_reward = environment_reward + safety_module.compute_reward_penalty(...)

### TESTING REQUIREMENTS
```python
import numpy as np
def test_protected_entity_constraint(): """Verify that intervention is blocked near protected entities""" shield = SafetyConstraintModule(safe_distance=5)
state = {
    'agent_positions': np.array([[0, 0]]),
    'protected_positions': np.array([[3, 0]]),  # 3 cells away (< 5)
    'target_positions': np.array([[10, 10]]),
    'agent_resources': np.array([10]),
    'target_values': np.array([5])
}
action, violated, reason = shield.verify_action(4, state, 0)  # 4 = INTERVENTION
assert violated == True
assert reason == "Protected Entity Constraint"
assert action == 6  # HOLD
print("✓ Protected entity constraint test passed")
def test_proportionality(): """Verify that low-benefit high-cost actions are blocked""" shield = SafetyConstraintModule(safe_distance=5, proportionality_lambda=1.0)
state = {
    'agent_positions': np.array([[0, 0]]),
    'protected_positions': np.array([[8, 0]]),  # Far enough for Rule 1
    'target_positions': np.array([[2, 0]]),     # Close target
    'agent_resources': np.array([10]),
    'target_values': np.array([1])  # Low value
}
action, violated, reason = shield.verify_action(4, state, 0)
# Should violate proportionality (cost > benefit)
assert violated == True
assert reason == "Proportionality Constraint"
print("✓ Proportionality test passed")
def test_high_value_target(): """Verify that high-value targets ARE allowed even with some risk""" shield = SafetyConstraintModule(safe_distance=5, proportionality_lambda=1.0)
state = {
    'agent_positions': np.array([[0, 0]]),
    'protected_positions': np.array([[8, 0]]),
    'target_positions': np.array([[2, 0]]),
    'agent_resources': np.array([10]),
    'target_values': np.array([10])  # HIGH value
}
action, violated, reason = shield.verify_action(4, state, 0)
# Should NOT violate (benefit > cost)
assert violated == False
print("✓ High-value target test passed")
def test_action_indexing(): """Verify action indices match environment""" shield = SafetyConstraintModule()
# Non-intervention actions should pass through
for action_idx in [0, 1, 2, 3, 5, 6]:  # All except INTERVENTION=4
    state = {'agent_positions': np.array([[0, 0]]), 
             'protected_positions': np.array([[1, 0]]),  # Very close
             'target_positions': np.array([]),
             'agent_resources': np.array([10]),
             'target_values': np.array([])}
    final_action, violated, _ = shield.verify_action(action_idx, state, 0)
    assert final_action == action_idx  # Should not be modified
    assert violated == False
print("✓ Action indexing test passed")

### OUTPUT FORMAT

Provide:
1. Complete `shield.py` (renamed from `safety_module.py` for brevity)
2. Test file `test_shield.py`
3. Ethical justification document explaining:
   - Why these rules were chosen
   - How they map to safety principles
   - Parameter sensitivity analysis (what if λ = 0.5 vs 2.0?)
4. Ablation suggestion: Compare policy trained WITH shield vs WITHOUT shield

### ACADEMIC CONTEXT

This research addresses safety in:
- Multi-agent coordination
- High-stakes decision-making
- Learned policies with formal guarantees

Related work: Safe RL, Constrained MDPs, Neuro-symbolic AI

The proportionality constraint is inspired by principles from decision theory
and ethical frameworks for autonomous systems.




