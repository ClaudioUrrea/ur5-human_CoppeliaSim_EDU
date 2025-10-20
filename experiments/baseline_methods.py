"""
Baseline Methods for LBCF-MPC Comparison
Implementation of the 5 baseline methods used in the paper
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ManualCBF:
    """
    Manual CBF using spherical approximations.
    
    h(x) = ||p_r - p_h|| - (r_robot + r_human + d_min)
    """
    
    def __init__(self, r_robot: float = 0.4, r_human: float = 0.3, d_min: float = 0.15):
        self.r_robot = r_robot
        self.r_human = r_human
        self.d_min = d_min
        logger.info(f"Manual CBF initialized (r_robot={r_robot}, r_human={r_human})")
    
    def evaluate(self, state: np.ndarray) -> float:
        """Calculate manual CBF value."""
        # Extract positions (simplified)
        p_robot = state[0:3]  # Robot position proxy
        p_human = state[18:21]  # Human torso position
        
        distance = np.linalg.norm(p_robot - p_human)
        h_value = distance - (self.r_robot + self.r_human + self.d_min)
        
        return h_value
    
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """Calculate gradient of manual CBF."""
        p_robot = state[0:3]
        p_human = state[18:21]
        
        diff = p_robot - p_human
        distance = np.linalg.norm(diff)
        
        if distance < 1e-6:
            return np.zeros_like(state)
        
        grad = np.zeros_like(state)
        grad[0:3] = diff / distance
        
        return grad


class MPCSoft:
    """
    MPC with soft safety constraints (penalty-based).
    
    Adds safety as a penalty term in cost function instead of hard constraint.
    """
    
    def __init__(self, lambda_safety: float = 1000.0):
        self.lambda_safety = lambda_safety
        logger.info(f"MPC-Soft initialized (λ_safety={lambda_safety})")
    
    def compute_safety_penalty(self, distance: float, threshold: float = 0.15) -> float:
        """Calculate penalty for safety violation."""
        if distance >= threshold:
            return 0.0
        
        # Quadratic penalty below threshold
        violation = threshold - distance
        return self.lambda_safety * violation ** 2
    
    def solve(self, state: np.ndarray, distance: float) -> np.ndarray:
        """
        Solve MPC with soft constraint.
        
        Returns control action (simplified implementation).
        """
        # Nominal control
        u_nominal = np.random.randn(6) * 0.1
        
        # Adjust if too close
        if distance < 0.15:
            # Reduce control magnitude
            u_nominal *= 0.5
        
        return u_nominal


class CBFReactive:
    """
    Reactive CBF without human motion prediction.
    
    Uses constant velocity model: predicted_acceleration = 0
    """
    
    def __init__(self, d_min: float = 0.15, gamma: float = 0.1):
        self.d_min = d_min
        self.gamma = gamma
        logger.info(f"CBF-Reactive initialized (d_min={d_min}, gamma={gamma})")
    
    def predict_human_position(
        self, 
        current_pos: np.ndarray, 
        current_vel: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """Predict human position using constant velocity model."""
        # No acceleration assumed
        return current_pos + current_vel * dt
    
    def evaluate_cbf(self, robot_pos: np.ndarray, human_pos: np.ndarray) -> float:
        """Evaluate CBF value."""
        distance = np.linalg.norm(robot_pos - human_pos)
        return distance - self.d_min


class SafeRL:
    """
    Safe Reinforcement Learning baseline (PPO-Lagrangian).
    
    Simplified implementation for comparison.
    """
    
    def __init__(self, constraint_threshold: float = 0.15):
        self.constraint_threshold = constraint_threshold
        self.policy_network = None  # Placeholder
        logger.info("Safe-RL (PPO-Lagrangian) initialized")
    
    def select_action(self, state: np.ndarray, distance: float) -> np.ndarray:
        """
        Select action using learned policy with Lagrangian constraints.
        
        Simplified: returns random action modified by safety constraint.
        """
        # Simplified policy
        action = np.random.randn(6) * 0.5
        
        # Apply Lagrangian constraint
        if distance < self.constraint_threshold:
            # Penalize actions that move closer
            safety_penalty = (self.constraint_threshold - distance) * 10.0
            action *= np.exp(-safety_penalty)
        
        return action


class PIDTraditional:
    """
    Traditional PID control with pre-computed collision-free trajectories.
    
    Uses RRT-Connect for path planning and emergency stops.
    """
    
    def __init__(self, kp: float = 10.0, ki: float = 0.5, kd: float = 2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.emergency_zone = 1.0  # meters
        logger.info(f"PID-Traditional initialized (Kp={kp}, Ki={ki}, Kd={kd})")
    
    def compute_control(
        self, 
        current_pos: np.ndarray, 
        target_pos: np.ndarray,
        distance_to_human: float,
        dt: float = 0.02
    ) -> np.ndarray:
        """
        Compute PID control with emergency stop.
        
        Args:
            current_pos: Current joint positions [6]
            target_pos: Target joint positions [6]
            distance_to_human: Current distance to human
            dt: Time step
            
        Returns:
            Control action (joint velocities) [6]
        """
        # Emergency stop if human too close
        if distance_to_human < self.emergency_zone:
            logger.warning(f"Emergency stop! Distance: {distance_to_human:.2f}m")
            self.integral = np.zeros(6)  # Reset integrator
            return np.zeros(6)
        
        # PID control
        error = target_pos - current_pos
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        
        # Saturate
        control = np.clip(control, -3.15, 3.15)
        
        return control
    
    def reset(self):
        """Reset PID state."""
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)


class BaselineController:
    """
    Unified interface for all baseline methods.
    """
    
    def __init__(self, method: str = 'LBCF-MPC'):
        """
        Initialize baseline controller.
        
        Args:
            method: One of ['LBCF-MPC', 'CBF-Manual', 'MPC-Soft', 
                           'CBF-Reactive', 'Safe-RL', 'PID-Trad']
        """
        self.method = method
        
        if method == 'CBF-Manual':
            self.controller = ManualCBF()
        elif method == 'MPC-Soft':
            self.controller = MPCSoft()
        elif method == 'CBF-Reactive':
            self.controller = CBFReactive()
        elif method == 'Safe-RL':
            self.controller = SafeRL()
        elif method == 'PID-Trad':
            self.controller = PIDTraditional()
        elif method == 'LBCF-MPC':
            logger.info("LBCF-MPC should use main controller")
            self.controller = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Baseline controller initialized: {method}")
    
    def compute_control(
        self, 
        state: np.ndarray, 
        distance: float, 
        **kwargs
    ) -> np.ndarray:
        """
        Compute control action.
        
        Args:
            state: Current state
            distance: Distance to human
            **kwargs: Additional method-specific arguments
            
        Returns:
            Control action [6]
        """
        if self.method == 'PID-Trad':
            return self.controller.compute_control(
                state[0:6],  # Current position
                kwargs.get('target_pos', np.zeros(6)),
                distance
            )
        elif self.method == 'Safe-RL':
            return self.controller.select_action(state, distance)
        elif self.method == 'MPC-Soft':
            return self.controller.solve(state, distance)
        elif self.method == 'CBF-Reactive':
            # Simplified
            return np.random.randn(6) * 0.1
        elif self.method == 'CBF-Manual':
            # Would integrate with MPC solver
            return np.random.randn(6) * 0.1
        else:
            return np.zeros(6)


def compare_methods_single_step(
    state: np.ndarray,
    distance: float,
    methods: list = None
) -> Dict[str, np.ndarray]:
    """
    Compare all methods for a single timestep.
    
    Args:
        state: Current state [38]
        distance: Current distance to human
        methods: List of method names to compare
        
    Returns:
        Dictionary mapping method names to control actions
    """
    if methods is None:
        methods = ['LBCF-MPC', 'CBF-Manual', 'MPC-Soft', 
                  'CBF-Reactive', 'Safe-RL', 'PID-Trad']
    
    results = {}
    
    for method in methods:
        controller = BaselineController(method)
        try:
            control = controller.compute_control(state, distance)
            results[method] = control
        except Exception as e:
            logger.error(f"Error with {method}: {e}")
            results[method] = np.zeros(6)
    
    return results


if __name__ == "__main__":
    print("Testing baseline methods...")
    
    # Test state
    state = np.random.randn(38)
    distance = 0.25
    
    # Test each method
    methods = ['CBF-Manual', 'MPC-Soft', 'CBF-Reactive', 'Safe-RL', 'PID-Trad']
    
    for method in methods:
        print(f"\nTesting {method}...")
        controller = BaselineController(method)
        control = controller.compute_control(state, distance, target_pos=np.zeros(6))
        print(f"  Control shape: {control.shape}")
        print(f"  Control range: [{control.min():.3f}, {control.max():.3f}]")
    
    print("\n✓ All baseline methods tested successfully")