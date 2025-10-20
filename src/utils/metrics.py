"""
Metrics computation utilities
"""

import numpy as np
from typing import Dict, List


class SafetyMetrics:
    """Compute safety-related metrics"""
    
    @staticmethod
    def violation_rate(distances: np.ndarray, threshold: float = 0.15) -> float:
        """Compute percentage of timesteps with violations"""
        return np.mean(distances < threshold)
    
    @staticmethod
    def minimum_distance(distances: np.ndarray) -> float:
        """Minimum distance over trajectory"""
        return np.min(distances)
    
    @staticmethod
    def safety_margin(distances: np.ndarray, threshold: float = 0.15) -> float:
        """Average safety margin"""
        return np.mean(distances - threshold)
    
    @staticmethod
    def time_to_collision(distances: np.ndarray, velocities: np.ndarray, 
                         threshold: float = 0.15) -> float:
        """Estimate time to collision"""
        # Find first time trending toward violation
        margins = distances - threshold
        rates = -np.gradient(margins)
        
        unsafe_idx = np.where((margins > 0) & (rates > 0))[0]
        
        if len(unsafe_idx) == 0:
            return np.inf
        
        ttc = margins[unsafe_idx] / rates[unsafe_idx]
        return np.min(ttc)


class EfficiencyMetrics:
    """Compute efficiency-related metrics"""
    
    @staticmethod
    def compute_throughput(trajectory: Dict) -> float:
        """
        Compute throughput in pieces/hour
        
        Simplified: count completed tasks
        """
        # Mock: random throughput based on trajectory length
        duration_hours = len(trajectory['states']) * 0.02 / 3600
        completed_pieces = np.random.randint(4, 6)
        
        return completed_pieces / duration_hours
    
    @staticmethod
    def compute_energy(trajectory: Dict) -> float:
        """
        Compute energy consumption
        
        E = ∫ ||u||² dt
        """
        controls = np.array(trajectory['controls'])
        energy = np.sum(np.linalg.norm(controls, axis=1)**2) * 0.02
        
        # Convert to kJ/hour
        duration_hours = len(controls) * 0.02 / 3600
        return energy / duration_hours / 1000  # kJ/hour
    
    @staticmethod
    def compute_cycle_time(trajectory: Dict) -> float:
        """Average cycle time per task"""
        # Mock implementation
        return 25.4 + np.random.randn() * 1.8
    
    @staticmethod
    def compute_reba_score(trajectory: Dict) -> float:
        """Ergonomic REBA score"""
        # Mock implementation
        return 6.8 + np.random.randn() * 0.9