"""
Multi-Agent Coordination Environment for RL Research

This environment implements a PettingZoo-based simulation for studying:
- Scalable coordination (100-1000 agents)
- Robustness to communication failures
- Safety constraint enforcement
- Adversarial robustness

Academic Research Context:
This is an ABSTRACT research environment inspired by cooperative MARL methods
(QMIX, MADDPG) for studying coordination under adversarial conditions.

Author: Research Implementation
License: Academic Research Only
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Optional imports for neural network integration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from gymnasium.spaces import Discrete, Dict as DictSpace, Box
    from pettingzoo import ParallelEnv
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    # Define minimal compatibility classes
    class Discrete:
        def __init__(self, n):
            self.n = n
        def sample(self):
            return np.random.randint(0, self.n)
    
    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
    
    class DictSpace(dict):
        pass
    
    class ParallelEnv:
        """Minimal ParallelEnv compatibility"""
        metadata = {}
        
from collections import defaultdict


class CoordinationEnv(ParallelEnv):
    """
    Multi-agent coordination environment with communication noise injection.
    
    This environment simulates a grid world where teams of agents must coordinate
    to protect entities while facing adversarial agents and unreliable communication.
    
    Attributes:
        num_agents: Number of friendly agents (100-1000)
        grid_size: Size of square grid (default 50x50)
        obs_radius: Local observation radius (default 5)
        comm_radius: Communication radius (default 7)
        noise_probability: Fraction of corrupted communication channels (default 0.2)
    """
    
    metadata = {'render_modes': ['human'], 'name': 'coordination_v0'}
    
    # Action space indices - CRITICAL: Must match across all modules
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    INTERVENTION = 4  # High-impact action
    COMMUNICATE = 5   # Broadcast message
    HOLD = 6          # No action
    
    # Role types
    ROLE_SCOUT = 0
    ROLE_COORDINATOR = 1
    ROLE_SUPPORT = 2
    
    def __init__(
        self,
        num_agents: int = 100,
        num_adversaries: int = 30,
        num_protected: int = 10,
        grid_size: int = 50,
        obs_radius: int = 5,
        comm_radius: int = 7,
        noise_probability: float = 0.2,
        max_steps: int = 500,
        seed: Optional[int] = None
    ):
        """
        Initialize the coordination environment.
        
        Args:
            num_agents: Number of friendly agents
            num_adversaries: Number of adversarial agents
            num_protected: Number of protected entities
            grid_size: Size of square grid
            obs_radius: Radius for local observations
            comm_radius: Radius for communication
            noise_probability: Fraction of communication channels to corrupt
            max_steps: Maximum episode length
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Environment parameters (use private vars to avoid PettingZoo property conflicts)
        self._num_agents = num_agents
        self._num_adversaries = num_adversaries
        self._num_protected = num_protected
        self.grid_size = grid_size
        self.obs_radius = obs_radius
        self.comm_radius = comm_radius
        self.noise_probability = noise_probability
        self.max_steps = max_steps
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            
        # Agent identifiers
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()
        
        # Define action and observation spaces
        self._action_spaces = {agent: Discrete(7) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: DictSpace({
                'local_grid': Box(0, 1, shape=(5, 5, 4), dtype=np.float32),
                'self_state': Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                'messages': Box(-np.inf, np.inf, shape=(100, 8), dtype=np.float32),  # Max possible
                'neighbor_ids': Box(-1, num_agents, shape=(100,), dtype=np.int32)
            })
            for agent in self.possible_agents
        }
        
        # State variables (initialized in reset)
        self.agent_positions: Dict[str, np.ndarray] = {}
        self.agent_states: Dict[str, Dict[str, float]] = {}
        self.agent_roles: Dict[str, int] = {}
        self.adversary_positions: List[np.ndarray] = []
        self.protected_positions: List[np.ndarray] = []
        self.obstacles: np.ndarray = None
        self.grid: np.ndarray = None
        self.current_step: int = 0
        
        # Communication tracking
        self.message_buffer: Dict[str, np.ndarray] = {}
        self.noisy_channels: set = set()  # Track which agents have noisy channels this step
    
    @property
    def num_agents(self) -> int:
        """Number of friendly agents."""
        return self._num_agents
    
    @property
    def num_adversaries(self) -> int:
        """Number of adversarial agents."""
        return self._num_adversaries
    
    @property
    def num_protected(self) -> int:
        """Number of protected entities."""
        return self._num_protected
        
    @property
    def action_space(self) -> Discrete:
        """Return action space for a single agent."""
        return Discrete(7)
    
    def observation_space(self, agent: str) -> DictSpace:
        """Return observation space for a single agent."""
        return self._observation_spaces[agent]
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Dict[str, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            observations: Initial observations for all agents
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place obstacles (10% of cells)
        num_obstacles = int(0.1 * self.grid_size * self.grid_size)
        obstacle_positions = np.random.choice(
            self.grid_size * self.grid_size,
            size=num_obstacles,
            replace=False
        )
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for pos in obstacle_positions:
            x, y = pos // self.grid_size, pos % self.grid_size
            self.obstacles[x, y] = True
            self.grid[x, y] = -1  # Mark as obstacle
            
        # Place friendly agents with role distribution
        self.agent_positions = {}
        self.agent_states = {}
        self.agent_roles = {}
        
        # Role distribution: 50% Scout, 30% Coordinator, 20% Support
        num_scouts = int(0.5 * self.num_agents)
        num_coordinators = int(0.3 * self.num_agents)
        num_support = self.num_agents - num_scouts - num_coordinators
        
        roles = ([self.ROLE_SCOUT] * num_scouts +
                [self.ROLE_COORDINATOR] * num_coordinators +
                [self.ROLE_SUPPORT] * num_support)
        np.random.shuffle(roles)
        
        for i, agent in enumerate(self.agents):
            # Find empty position
            while True:
                x, y = np.random.randint(0, self.grid_size, 2)
                if not self.obstacles[x, y] and self.grid[x, y] == 0:
                    break
                    
            self.agent_positions[agent] = np.array([x, y])
            self.agent_roles[agent] = roles[i]
            self.agent_states[agent] = {
                'health': 100.0,
                'resource': 50.0,
                'team_id': 0,  # All friendly agents on team 0
                'active': True
            }
            self.grid[x, y] = i + 1  # Mark position
            
        # Place adversarial agents
        self.adversary_positions = []
        for _ in range(self.num_adversaries):
            while True:
                x, y = np.random.randint(0, self.grid_size, 2)
                if not self.obstacles[x, y] and self.grid[x, y] == 0:
                    break
            self.adversary_positions.append(np.array([x, y]))
            self.grid[x, y] = -2  # Mark as adversary
            
        # Place protected entities
        self.protected_positions = []
        for _ in range(self.num_protected):
            while True:
                x, y = np.random.randint(0, self.grid_size, 2)
                if not self.obstacles[x, y] and self.grid[x, y] == 0:
                    break
            self.protected_positions.append(np.array([x, y]))
            self.grid[x, y] = -3  # Mark as protected
            
        # Initialize message buffer
        self.message_buffer = {agent: self._create_default_message(agent) 
                              for agent in self.agents}
        
        # Get initial observations
        observations = self._get_observations()
        
        return observations
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one environment step.
        
        Args:
            actions: Dictionary mapping agent_id to action integer
            
        Returns:
            observations: New observations for each agent
            rewards: Rewards for each agent
            terminations: Whether each agent is done
            truncations: Whether episode is truncated
            infos: Additional information
        """
        self.current_step += 1
        
        # Track metrics for reward calculation
        adversaries_neutralized = 0
        teammates_lost = 0
        protected_harmed = 0
        mission_progress = 0
        
        # Clear message buffer
        self.message_buffer = {}
        
        # Process actions for each agent
        for agent, action in actions.items():
            if agent not in self.agents or not self.agent_states[agent]['active']:
                continue
                
            pos = self.agent_positions[agent]
            
            # Movement actions
            if action == self.MOVE_NORTH:
                new_pos = pos + np.array([-1, 0])
            elif action == self.MOVE_SOUTH:
                new_pos = pos + np.array([1, 0])
            elif action == self.MOVE_EAST:
                new_pos = pos + np.array([0, 1])
            elif action == self.MOVE_WEST:
                new_pos = pos + np.array([0, -1])
            elif action == self.INTERVENTION:
                # Check for nearby adversaries within range 2
                for i, adv_pos in enumerate(self.adversary_positions):
                    if np.linalg.norm(pos - adv_pos) <= 2:
                        # Neutralize adversary with 80% success rate
                        if np.random.random() < 0.8:
                            self.adversary_positions[i] = np.array([-1, -1])  # Remove
                            adversaries_neutralized += 1
                            mission_progress += 1
                new_pos = pos  # No movement
            elif action == self.COMMUNICATE:
                # Create message to broadcast
                self.message_buffer[agent] = self._create_message(agent)
                new_pos = pos  # No movement
            elif action == self.HOLD:
                new_pos = pos  # No movement
            else:
                new_pos = pos  # Invalid action treated as hold
                
            # Validate and apply movement
            if action in [self.MOVE_NORTH, self.MOVE_SOUTH, self.MOVE_EAST, self.MOVE_WEST]:
                if self._is_valid_position(new_pos):
                    # Update grid
                    self.grid[pos[0], pos[1]] = 0
                    self.agent_positions[agent] = new_pos
                    # Will update grid after all movements
                    
        # Update grid with new positions
        self.grid[self.grid > 0] = 0  # Clear agent positions
        for i, agent in enumerate(self.agents):
            if self.agent_states[agent]['active']:
                pos = self.agent_positions[agent]
                self.grid[pos[0], pos[1]] = i + 1
                
        # Simple adversary behavior (random movement)
        for i, adv_pos in enumerate(self.adversary_positions):
            if adv_pos[0] == -1:  # Neutralized
                continue
            direction = np.random.randint(0, 4)
            if direction == 0:
                new_pos = adv_pos + np.array([-1, 0])
            elif direction == 1:
                new_pos = adv_pos + np.array([1, 0])
            elif direction == 2:
                new_pos = adv_pos + np.array([0, 1])
            else:
                new_pos = adv_pos + np.array([0, -1])
                
            if self._is_valid_position(new_pos):
                self.adversary_positions[i] = new_pos
                
        # Check for protected entity harm
        for protected_pos in self.protected_positions:
            for adv_pos in self.adversary_positions:
                if adv_pos[0] != -1 and np.linalg.norm(protected_pos - adv_pos) <= 1:
                    protected_harmed += 1
                    
        # Inject communication noise
        if self.message_buffer:
            self.message_buffer = self.inject_noise(self.message_buffer)
            
        # Calculate rewards
        base_reward = (
            10 * adversaries_neutralized
            - 5 * teammates_lost
            - 20 * protected_harmed
            + 1 * mission_progress
        )
        
        rewards = {agent: base_reward for agent in self.agents 
                  if self.agent_states[agent]['active']}
        
        # Get new observations
        observations = self._get_observations()
        
        # Check termination conditions
        active_adversaries = sum(1 for pos in self.adversary_positions if pos[0] != -1)
        active_agents = sum(1 for agent in self.agents if self.agent_states[agent]['active'])
        
        terminated = (
            active_adversaries == 0 or
            active_agents < 0.5 * self.num_agents or
            self.current_step >= self.max_steps
        )
        
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: self.current_step >= self.max_steps for agent in self.agents}
        
        # Additional info
        infos = {
            agent: {
                'adversaries_remaining': active_adversaries,
                'step': self.current_step,
                'protected_harmed': protected_harmed,
                'noisy_channel': agent in self.noisy_channels
            }
            for agent in self.agents
        }
        
        return observations, rewards, terminations, truncations, infos
    
    def inject_noise(
        self,
        messages_dict: Dict[str, np.ndarray],
        noise_probability: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate communication channel corruption.
        
        This method replaces a fraction of messages with Gaussian noise to simulate
        unreliable communication channels or adversarial interference.
        
        Research Purpose:
            Test agent robustness to communication failures. Agents must learn to
            identify and filter corrupted messages without explicit corruption flags.
        
        Design Rationale:
            - 20% noise rate based on distributed systems fault tolerance literature
            - Gaussian noise N(0, σ=2.0) chosen because:
              * Real messages have typical magnitude in [0, 10] range
              * σ=2.0 produces noise in ~[-4, +4] (95% confidence)
              * Signal-to-noise ratio ~2.5:1 makes detection challenging but feasible
              * Calibrated with Trust Gate threshold=0.5
        
        Args:
            messages_dict: Dictionary mapping agent_id to message vector
            noise_probability: Fraction of channels to corrupt (default: self.noise_probability)
            
        Returns:
            corrupted_messages: Same structure with some vectors replaced by noise
            
        Note:
            Recipients do NOT receive a flag indicating corruption - they must infer
            this from the message content itself. This is critical for testing
            robustness.
        """
        if noise_probability is None:
            noise_probability = self.noise_probability
            
        corrupted_messages = {}
        self.noisy_channels = set()
        
        # Select agents to corrupt
        num_to_corrupt = int(noise_probability * len(messages_dict))
        corrupted_agents = np.random.choice(
            list(messages_dict.keys()),
            size=num_to_corrupt,
            replace=False
        )
        
        for agent_id, message in messages_dict.items():
            if agent_id in corrupted_agents:
                # Replace with Gaussian noise N(0, σ=2.0)
                corrupted_messages[agent_id] = np.random.normal(0, 2.0, size=message.shape)
                self.noisy_channels.add(agent_id)
            else:
                corrupted_messages[agent_id] = message
                
        return corrupted_messages
    
    def get_protected_distances(self, agent_id: str) -> np.ndarray:
        """
        Calculate distances from agent to all protected entities.
        
        This is needed for safety constraint testing - agents should maintain
        minimum distance from protected entities during interventions.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Array of distances to each protected entity, shape (num_protected_entities,)
        """
        if agent_id not in self.agent_positions:
            return np.full(self.num_protected, np.inf)
            
        agent_pos = self.agent_positions[agent_id]
        distances = np.array([
            np.linalg.norm(agent_pos - protected_pos)
            for protected_pos in self.protected_positions
        ])
        
        return distances
    
    def collate_observations(
        self,
        obs_list: List[Dict[str, Any]],
        max_neighbors: int = 20
    ) -> Dict[str, Any]:
        """
        Convert list of observations to batched tensors for neural networks.
        
        This is REQUIRED for integration with Trust Gate and other neural modules.
        Handles variable-length message lists by padding/truncating to fixed length.
        
        Args:
            obs_list: List of observation dictionaries from environment
            max_neighbors: Maximum number of neighbors to consider
            
        Returns:
            batched_obs: Dictionary with tensors (PyTorch if available, else NumPy arrays):
                - local_obs: (batch, 6) - self state vectors
                - messages: (batch, max_neighbors, 8) - padded messages
                - neighbor_mask: (batch, max_neighbors) - valid message mask
                - neighbor_ids: (batch, max_neighbors) - neighbor agent IDs
                - neighbor_states: (batch, max_neighbors, 6) - extracted states
                - neighbor_roles: (batch, max_neighbors) - extracted roles
        """
        batch_size = len(obs_list)
        
        # Collect and pad messages
        all_messages = []
        all_masks = []
        all_neighbor_ids = []
        
        for obs in obs_list:
            msgs = obs['messages']
            ids = obs['neighbor_ids']
            
            # Convert list to array if needed
            if isinstance(msgs, list):
                if len(msgs) == 0:
                    msgs = np.zeros((0, 8))
                else:
                    msgs = np.array(msgs)
            if isinstance(ids, list):
                ids = np.array(ids)
                
            num_msgs = len(msgs)
            
            if num_msgs < max_neighbors:
                # Pad with zeros
                if num_msgs == 0:
                    padded_msgs = np.zeros((max_neighbors, 8))
                    padded_ids = np.zeros(max_neighbors)
                else:
                    padded_msgs = np.vstack([
                        msgs,
                        np.zeros((max_neighbors - num_msgs, 8))
                    ])
                    padded_ids = np.concatenate([
                        ids,
                        np.zeros(max_neighbors - num_msgs)
                    ])
                mask = np.array([1] * num_msgs + [0] * (max_neighbors - num_msgs))
            else:
                # Truncate
                padded_msgs = msgs[:max_neighbors]
                padded_ids = ids[:max_neighbors]
                mask = np.ones(max_neighbors)
            
            all_messages.append(padded_msgs)
            all_masks.append(mask)
            all_neighbor_ids.append(padded_ids)
        
        # Stack into arrays
        messages_array = np.stack(all_messages)
        local_obs_array = np.stack([o['self_state'] for o in obs_list])
        masks_array = np.stack(all_masks)
        ids_array = np.stack(all_neighbor_ids)
        
        # Convert to torch tensors if available, otherwise keep as numpy
        if TORCH_AVAILABLE:
            batched_obs = {
                'local_obs': torch.tensor(local_obs_array, dtype=torch.float32),
                'messages': torch.tensor(messages_array, dtype=torch.float32),
                'neighbor_mask': torch.tensor(masks_array, dtype=torch.float32),
                'neighbor_ids': torch.tensor(ids_array, dtype=torch.long),
                # Extract neighbor states (first 6 dimensions of message)
                'neighbor_states': torch.tensor(messages_array[:, :, :6], dtype=torch.float32),
                # Extract roles (index 1 in message format)
                'neighbor_roles': torch.tensor(messages_array[:, :, 1], dtype=torch.long)
            }
        else:
            batched_obs = {
                'local_obs': local_obs_array.astype(np.float32),
                'messages': messages_array.astype(np.float32),
                'neighbor_mask': masks_array.astype(np.float32),
                'neighbor_ids': ids_array.astype(np.int64),
                # Extract neighbor states (first 6 dimensions of message)
                'neighbor_states': messages_array[:, :, :6].astype(np.float32),
                # Extract roles (index 1 in message format)
                'neighbor_roles': messages_array[:, :, 1].astype(np.int64)
            }
        
        return batched_obs
    
    def _get_observations(self) -> Dict[str, Dict]:
        """Generate observations for all agents."""
        observations = {}
        
        for agent in self.agents:
            if not self.agent_states[agent]['active']:
                continue
                
            pos = self.agent_positions[agent]
            
            # Local grid observation (5x5 with 4 channels)
            local_grid = self._get_local_grid(pos)
            
            # Self state: [x, y, health, resource, role_id, team_id]
            self_state = np.array([
                pos[0] / self.grid_size,  # Normalize position
                pos[1] / self.grid_size,
                self.agent_states[agent]['health'] / 100.0,
                self.agent_states[agent]['resource'] / 100.0,
                self.agent_roles[agent],
                self.agent_states[agent]['team_id']
            ], dtype=np.float32)
            
            # Get messages from neighbors within comm radius
            messages, neighbor_ids = self._get_messages(agent)
            
            observations[agent] = {
                'local_grid': local_grid,
                'self_state': self_state,
                'messages': messages,
                'neighbor_ids': neighbor_ids
            }
            
        return observations
    
    def _get_local_grid(self, center_pos: np.ndarray) -> np.ndarray:
        """
        Extract local grid observation around agent position.
        
        Returns array of shape (5, 5, 4) with channels:
        - Channel 0: Teammates
        - Channel 1: Adversaries
        - Channel 2: Protected entities
        - Channel 3: Obstacles
        """
        local_grid = np.zeros((5, 5, 4), dtype=np.float32)
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = center_pos[0] + dx, center_pos[1] + dy
                local_x, local_y = dx + 2, dy + 2
                
                # Check bounds
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    local_grid[local_x, local_y, 3] = 1.0  # Treat out of bounds as obstacle
                    continue
                    
                # Check if obstacle blocks view
                if self.obstacles[x, y]:
                    local_grid[local_x, local_y, 3] = 1.0
                    continue
                    
                # Check for teammates
                for other_agent in self.agents:
                    if (self.agent_states[other_agent]['active'] and
                        np.array_equal(self.agent_positions[other_agent], [x, y])):
                        local_grid[local_x, local_y, 0] = 1.0
                        
                # Check for adversaries
                for adv_pos in self.adversary_positions:
                    if adv_pos[0] != -1 and np.array_equal(adv_pos, [x, y]):
                        local_grid[local_x, local_y, 1] = 1.0
                        
                # Check for protected entities
                for prot_pos in self.protected_positions:
                    if np.array_equal(prot_pos, [x, y]):
                        local_grid[local_x, local_y, 2] = 1.0
                        
        return local_grid
    
    def _get_messages(self, agent: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get messages from neighbors within communication radius.
        
        Returns:
            messages: List of message vectors
            neighbor_ids: Array of neighbor agent indices
        """
        pos = self.agent_positions[agent]
        messages = []
        neighbor_ids = []
        
        for i, other_agent in enumerate(self.agents):
            if other_agent == agent or not self.agent_states[other_agent]['active']:
                continue
                
            other_pos = self.agent_positions[other_agent]
            distance = np.linalg.norm(pos - other_pos)
            
            if distance <= self.comm_radius:
                # Get message from buffer (or default if not broadcasting)
                if other_agent in self.message_buffer:
                    messages.append(self.message_buffer[other_agent])
                else:
                    messages.append(self._create_default_message(other_agent))
                neighbor_ids.append(i)
                
        return messages, np.array(neighbor_ids, dtype=np.int32)
    
    def _create_message(self, agent: str) -> np.ndarray:
        """
        Create a message vector for an agent.
        
        Message format: [sender_id, role, x, y, status, target_x, target_y, priority]
        """
        pos = self.agent_positions[agent]
        agent_idx = self.agents.index(agent)
        
        # Find nearest adversary as target
        target_x, target_y = 0, 0
        min_dist = float('inf')
        for adv_pos in self.adversary_positions:
            if adv_pos[0] != -1:
                dist = np.linalg.norm(pos - adv_pos)
                if dist < min_dist:
                    min_dist = dist
                    target_x, target_y = adv_pos
                    
        message = np.array([
            agent_idx,
            self.agent_roles[agent],
            pos[0] / self.grid_size,  # Normalize
            pos[1] / self.grid_size,
            self.agent_states[agent]['health'] / 100.0,
            target_x / self.grid_size,
            target_y / self.grid_size,
            min(1.0, 10.0 / (min_dist + 1))  # Priority based on proximity
        ], dtype=np.float32)
        
        return message
    
    def _create_default_message(self, agent: str) -> np.ndarray:
        """Create a default message when agent is not broadcasting."""
        pos = self.agent_positions[agent]
        agent_idx = self.agents.index(agent)
        
        return np.array([
            agent_idx,
            self.agent_roles[agent],
            pos[0] / self.grid_size,
            pos[1] / self.grid_size,
            self.agent_states[agent]['health'] / 100.0,
            0, 0, 0
        ], dtype=np.float32)
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is valid (in bounds and not obstacle)."""
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        if self.obstacles[x, y]:
            return False
        return True
    
    def render(self):
        """Render the environment (optional, for visualization)."""
        pass
    
    def close(self):
        """Clean up environment resources."""
        pass
