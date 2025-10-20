"""
CoppeliaSim simulation environment interface
"""

import numpy as np
from typing import Tuple, Optional
import time


class CoppeliaSimEnv:
    """
    Interface to CoppeliaSim simulation
    
    Note: This is a simplified mock interface.
    Real implementation would use ZeroMQ RemoteAPI.
    """
    
    def __init__(self, scene_file: str, scenario: int = 3, headless: bool = True):
        self.scene_file = scene_file
        self.scenario = scenario
        self.headless = headless
        
        # State
        self.state = None
        self.time = 0.0
        
        # Robot state (6 DOF)
        self.q_r = np.zeros(6)
        self.q_dot_r = np.zeros(6)
        
        # Human state (simplified)
        self.p_h = np.random.uniform(-1.0, 1.0, 18)
        self.v_h = np.random.uniform(-0.2, 0.2, 18)
        
        print(f"Initialized CoppeliaSim environment")
        print(f"  Scene: {scene_file}")
        print(f"  Scenario: {scenario}")
        print(f"  Headless: {headless}")
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.time = 0.0
        
        # Reset robot
        self.q_r = np.random.uniform(-0.5, 0.5, 6)
        self.q_dot_r = np.zeros(6)
        
        # Reset human
        self.p_h = np.random.uniform(-1.0, 1.0, 18)
        self.v_h = np.random.uniform(-0.2, 0.2, 18)
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        # Object states (on conveyor)
        s_obj = np.random.rand(12)
        
        # Conveyor state
        s_conv = np.array([0.4])  # Speed
        
        # Task state
        s_task = np.random.rand(5)
        
        state = np.concatenate([
            self.q_r,      # 6
            self.q_dot_r,  # 6
            self.p_h,      # 18
            self.v_h,      # 18
            s_obj,         # 12
            s_conv,        # 1
            s_task         # 5
        ])
        
        return state
    
    def step(self, u: np.ndarray, dt: float = 0.02):
        """
        Apply control and simulate one step
        
        Args:
            u: Joint velocity commands [6]
            dt: Time step
        """
        # Update robot
        self.q_r += dt * u
        self.q_dot_r = u
        
        # Update human (autonomous motion)
        a_h = -0.1 * self.v_h + np.random.randn(18) * 0.05
        self.p_h += dt * self.v_h
        self.v_h += dt * a_h
        
        self.time += dt
    
    def compute_distance(self) -> float:
        """
        Compute minimum distance between robot and human
        
        Simplified: distance between end-effector and closest human point
        """
        # Simple forward kinematics (approximation)
        p_ee = self.q_r[0:3]  # Use first 3 joints as position proxy
        
        # Human body parts
        p_h_parts = self.p_h.reshape(-1, 3)
        
        # Compute distances
        distances = np.linalg.norm(p_ee - p_h_parts, axis=1)
        
        return np.min(distances)
    
    def close(self):
        """Close simulation"""
        print("Closing simulation...")