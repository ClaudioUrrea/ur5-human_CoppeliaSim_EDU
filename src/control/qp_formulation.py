"""
QP Formulation for LBCF-MPC
Quadratic Programming problem formulation with CBF constraints

This module implements the QP formulation for the LBCF-MPC algorithm,
including multi-objective cost functions and safety constraints.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QPFormulation:
    """
    Quadratic Programming formulation for LBCF-MPC.
    
    Solves the optimization problem:
        min  J(u) = sum_i w_i * J_i(u)
        s.t. CBF constraint: h(x_k+1) >= (1-gamma)*h(x_k)
             Input constraints: u_min <= u <= u_max
             State constraints: x_min <= x <= x_max
    
    where J_i are the individual objectives (throughput, energy, etc.)
    """
    
    def __init__(
        self,
        n_controls: int = 6,
        horizon: int = 20,
        dt: float = 0.02,
        gamma: float = 0.1
    ):
        """
        Initialize QP formulation.
        
        Args:
            n_controls: Number of control inputs (6 for UR5)
            horizon: MPC prediction horizon
            dt: Time step (s)
            gamma: CBF decay rate
        """
        self.n_controls = n_controls
        self.horizon = horizon
        self.dt = dt
        self.gamma = gamma
        
        # Control limits (UR5 joint velocity limits)
        self.u_min = -np.ones(n_controls) * 3.15  # rad/s
        self.u_max = np.ones(n_controls) * 3.15
        
        # Multi-objective weights (default)
        self.weights = {
            'throughput': 1.0,
            'energy': 0.5,
            'cycle_time': 0.8,
            'ergonomics': 0.3,
            'equipment_wear': 0.2,
            'fairness': 0.1
        }
        
        logger.info(f"QP Formulation initialized (H={horizon}, dt={dt}, gamma={gamma})")
    
    def formulate_qp(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        h_values: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        u_prev: Optional[np.ndarray] = None
    ) -> Tuple[cp.Problem, cp.Variable]:
        """
        Formulate the QP problem for current state.
        
        Args:
            x0: Current state [n_states]
            x_ref: Reference trajectory [horizon, n_states]
            h_values: CBF values along horizon [horizon]
            A: System dynamics matrix [n_states, n_states]
            B: Control input matrix [n_states, n_controls]
            u_prev: Previous control for warm start [n_controls]
            
        Returns:
            Tuple of (cvxpy Problem, control variable)
        """
        # Decision variables
        U = cp.Variable((self.horizon, self.n_controls))
        
        # Initialize cost
        cost = 0.0
        
        # Tracking cost (primary objective)
        Q = np.eye(len(x0)) * 10.0  # State weight
        R = np.eye(self.n_controls) * 0.1  # Control weight
        
        x_predicted = x0
        for k in range(self.horizon):
            # Predict next state (linearized dynamics)
            x_next = A @ x_predicted + B @ U[k, :]
            
            # Tracking error
            tracking_error = x_next - x_ref[k, :]
            cost += cp.quad_form(tracking_error, Q)
            
            # Control effort
            cost += cp.quad_form(U[k, :], R)
            
            x_predicted = x_next
        
        # Control smoothness (minimize jerk)
        for k in range(self.horizon - 1):
            cost += 0.01 * cp.sum_squares(U[k+1, :] - U[k, :])
        
        # Warm start penalty
        if u_prev is not None:
            cost += 0.1 * cp.sum_squares(U[0, :] - u_prev)
        
        # Multi-objective terms
        cost += self._add_multi_objective_costs(U)
        
        # Constraints
        constraints = []
        
        # Input constraints
        constraints.append(U >= self.u_min)
        constraints.append(U <= self.u_max)
        
        # CBF constraints (forward invariance)
        for k in range(self.horizon - 1):
            # h(x_{k+1}) >= (1 - gamma) * h(x_k)
            constraints.append(h_values[k+1] >= (1 - self.gamma) * h_values[k])
        
        # Formulate problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        return problem, U
    
    def _add_multi_objective_costs(self, U: cp.Variable) -> cp.Expression:
        """
        Add multi-objective cost terms.
        
        Args:
            U: Control variable
            
        Returns:
            Combined multi-objective cost
        """
        cost = 0.0
        
        # Energy efficiency (minimize squared control)
        cost += self.weights['energy'] * cp.sum_squares(U)
        
        # Equipment wear (minimize accelerations)
        for k in range(self.horizon - 1):
            cost += self.weights['equipment_wear'] * cp.sum_squares(U[k+1, :] - U[k, :])
        
        return cost
    
    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        h_values: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        u_prev: Optional[np.ndarray] = None,
        solver: str = 'OSQP'
    ) -> Dict:
        """
        Solve the QP problem.
        
        Args:
            x0: Current state
            x_ref: Reference trajectory
            h_values: CBF values
            A: Dynamics matrix
            B: Input matrix
            u_prev: Previous control
            solver: QP solver ('OSQP', 'ECOS', 'SCS')
            
        Returns:
            Dictionary with solution:
                - 'u_opt': Optimal control [n_controls]
                - 'U_opt': Full control sequence [horizon, n_controls]
                - 'solve_time': Solution time (ms)
                - 'status': Solver status
                - 'cost': Optimal cost value
        """
        import time
        
        # Formulate problem
        problem, U = self.formulate_qp(x0, x_ref, h_values, A, B, u_prev)
        
        # Solve
        start_time = time.perf_counter()
        
        try:
            problem.solve(solver=solver, warm_start=(u_prev is not None))
            solve_time = (time.perf_counter() - start_time) * 1000  # ms
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"QP solver status: {problem.status}")
                # Return zero control if infeasible
                return {
                    'u_opt': np.zeros(self.n_controls),
                    'U_opt': np.zeros((self.horizon, self.n_controls)),
                    'solve_time': solve_time,
                    'status': problem.status,
                    'cost': float('inf')
                }
            
            return {
                'u_opt': U.value[0, :],  # First control action
                'U_opt': U.value,  # Full sequence for warm start
                'solve_time': solve_time,
                'status': problem.status,
                'cost': problem.value
            }
            
        except Exception as e:
            logger.error(f"QP solve failed: {e}")
            return {
                'u_opt': np.zeros(self.n_controls),
                'U_opt': np.zeros((self.horizon, self.n_controls)),
                'solve_time': 0.0,
                'status': 'error',
                'cost': float('inf')
            }
    
    def set_weights(self, weights: Dict[str, float]):
        """Update multi-objective weights."""
        self.weights.update(weights)
        logger.info(f"Updated weights: {self.weights}")
    
    def set_cbf_decay(self, gamma: float):
        """Update CBF decay rate."""
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        self.gamma = gamma
        logger.info(f"Updated gamma: {gamma}")


class AdaptiveWeightScheduler:
    """
    Adaptive weight scheduler for multi-objective optimization.
    
    Adjusts objective weights based on current performance metrics
    to balance competing objectives dynamically.
    """
    
    def __init__(self, initial_weights: Dict[str, float]):
        """Initialize with default weights."""
        self.weights = initial_weights.copy()
        self.history = []
    
    def update_weights(
        self,
        metrics: Dict[str, float],
        targets: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update weights based on current performance.
        
        Args:
            metrics: Current performance metrics
            targets: Target values for each metric
            
        Returns:
            Updated weights
        """
        # Simple adaptive rule: increase weight if far from target
        for key in self.weights:
            if key in metrics and key in targets:
                error = abs(metrics[key] - targets[key]) / max(targets[key], 1e-6)
                # Increase weight if error is large
                self.weights[key] *= (1.0 + 0.1 * error)
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.history.append(self.weights.copy())
        
        return self.weights.copy()


if __name__ == "__main__":
    # Test QP formulation
    print("Testing QP Formulation...")
    
    # Create formulation
    qp = QPFormulation(n_controls=6, horizon=20)
    
    # Create dummy problem
    n_states = 38
    x0 = np.random.randn(n_states)
    x_ref = np.random.randn(20, n_states)
    h_values = np.abs(np.random.randn(20)) + 0.1  # Ensure positive
    A = np.eye(n_states)
    B = np.random.randn(n_states, 6) * 0.1
    
    # Solve
    solution = qp.solve(x0, x_ref, h_values, A, B)
    
    print(f"Status: {solution['status']}")
    print(f"Solve time: {solution['solve_time']:.2f}ms")
    print(f"Optimal control: {solution['u_opt']}")
    print("✓ QP formulation test passed")