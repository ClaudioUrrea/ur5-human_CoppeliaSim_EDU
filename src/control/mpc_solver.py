"""
LBCF-MPC: Learning-Based CBF Model Predictive Control
"""

import numpy as np
import osqp
from scipy import sparse
from typing import Dict, Tuple, Optional
import time

from ..cbf.lipschitz_network import LipschitzCBFNetwork
from ..prediction.gp_predictor import GPHumanPredictor


class LBCFMPC:
    """
    Learning-Based Control Barrier Function Model Predictive Control
    
    Solves:
        min_u  Σ [w^T J_i(x,u) + ρ||u - u_prev||² + ρ_v||u||²]
        s.t.   x_{k+1} = f_d(x_k, u_k)
               h(x_k) ≥ ε_safe
               L_f h + L_g h · u ≥ -α(h)
               u_min ≤ u_k ≤ u_max
    """
    
    def __init__(self,
                 cbf_model: LipschitzCBFNetwork,
                 gp_predictor: GPHumanPredictor,
                 horizon: int = 20,
                 dt: float = 0.02,
                 n_states: int = 38,
                 n_controls: int = 6,
                 objective_weights: np.ndarray = None,
                 epsilon_safe: float = 0.02,
                 beta_uncertainty: float = 2.0,
                 rho_smoothness: float = 0.1,
                 rho_velocity: float = 0.01,
                 alpha_coeff: float = 0.5,
                 control_limits: Tuple[float, float] = (-np.pi/2, np.pi/2)):
        
        self.cbf = cbf_model
        self.gp = gp_predictor
        self.H = horizon
        self.dt = dt
        self.n = n_states
        self.m = n_controls
        
        # Objective weights [J1, ..., J6]
        if objective_weights is None:
            objective_weights = np.array([0.25, 0.20, 0.15, 0.20, 0.10, 0.10])
        self.w = objective_weights
        
        # Safety parameters
        self.eps_safe_base = epsilon_safe
        self.beta = beta_uncertainty
        self.alpha_coeff = alpha_coeff
        
        # Regularization
        self.rho = rho_smoothness
        self.rho_v = rho_velocity
        
        # Control limits
        self.u_min, self.u_max = control_limits
        
        # Previous solution (warm start)
        self.u_prev = np.zeros((self.H, self.m))
        
        # OSQP solver
        self.solver = osqp.OSQP()
        self.solver_initialized = False
        
        # Timing statistics
        self.solve_times = []
    
    def dynamics(self, x: np.ndarray, u: np.ndarray, a_h: np.ndarray) -> np.ndarray:
        """
        Discrete-time dynamics: x_{k+1} = f_d(x_k, u_k)
        
        Simplified Euler integration:
        x_{k+1} = x_k + dt * [f(x_k, a_h) + g(x_k) * u_k]
        """
        # Extract states
        q_r = x[0:6]  # Joint angles
        q_dot_r = x[6:12]  # Joint velocities
        p_h = x[12:30]  # Human positions
        v_h = x[30:48]  # Human velocities
        # ... other states
        
        # Robot dynamics: q_dot_{k+1} = u (velocity control)
        # q_{k+1} = q_k + dt * u
        
        # Human dynamics: p_h_{k+1} = p_h_k + dt * v_h_k
        # v_h_{k+1} = v_h_k + dt * a_h_k
        
        x_next = x.copy()
        x_next[0:6] = q_r + self.dt * u  # Joint angles
        x_next[6:12] = u  # Joint velocities
        x_next[12:30] = p_h + self.dt * v_h  # Human positions
        x_next[30:48] = v_h + self.dt * a_h  # Human velocities
        
        return x_next
    
    def linearize_dynamics(self, x_ref: np.ndarray, u_ref: np.ndarray, 
                          a_h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize dynamics around reference trajectory
        
        Returns:
            A: State transition matrix [n x n]
            B: Control input matrix [n x m]
        """
        # Simplified linearization (identity + dt * Jacobian)
        A = np.eye(self.n)
        B = np.zeros((self.n, self.m))
        
        # Robot kinematics
        A[0:6, 6:12] = self.dt * np.eye(6)  # q depends on q_dot
        B[0:6, 0:6] = self.dt * np.eye(6)  # q depends on u
        B[6:12, 0:6] = np.eye(6)  # q_dot = u
        
        # Human dynamics (autonomous)
        A[12:30, 30:48] = self.dt * np.eye(18)  # p_h depends on v_h
        
        return A, B
    
    def compute_cbf_constraint(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute CBF constraint: L_f h + L_g h · u ≥ -α(h)
        
        Returns:
            grad_h_u: Gradient w.r.t. control [m]
            rhs: Right-hand side of constraint
        """
        import torch
        
        x_torch = torch.FloatTensor(x).unsqueeze(0)
        x_torch.requires_grad_(True)
        
        # Barrier value
        h_val = self.cbf(x_torch).item()
        
        # Gradient
        grad_h = self.cbf.compute_gradient(x_torch).squeeze().detach().numpy()
        
        # Extract gradient w.r.t. joint velocities (control affects these)
        grad_h_u = grad_h[6:12]  # ∂h/∂q_dot
        
        # CBF condition: L_f h + L_g h · u ≥ -α(h)
        # L_g h · u represents: grad_h_u^T · u
        # We need: grad_h_u^T · u ≥ -α(h) - L_f h
        
        alpha_h = self.alpha_coeff * h_val
        L_f_h = 0.0  # Simplified (would require full dynamics)
        
        rhs = -alpha_h - L_f_h
        
        return grad_h_u, rhs
    
    def adaptive_safety_margin(self, sigma_pred: float) -> float:
        """
        Compute adaptive safety margin based on prediction uncertainty
        
        ε_safe = ε_base + β * σ_pred
        """
        return self.eps_safe_base + self.beta * sigma_pred
    
    def formulate_qp(self, 
                     x0: np.ndarray,
                     x_ref: np.ndarray,
                     u_ref: np.ndarray,
                     a_h_pred: np.ndarray,
                     sigma_pred: float) -> Tuple:
        """
        Formulate QP problem
        
        Returns:
            P, q, A, l, u for OSQP
        """
        # Decision variables: U = [u_0, ..., u_{H-1}] ∈ R^{Hm}
        n_vars = self.H * self.m
        
        # Objective: min 0.5 U^T P U + q^T U
        P_diag = []
        q_vec = []
        
        for k in range(self.H):
            # Smoothness: ρ||u_k - u_{k-1}||²
            P_diag.extend([self.rho] * self.m)
            if k == 0:
                q_vec.extend(-self.rho * self.u_prev[0])
            else:
                q_vec.extend(np.zeros(self.m))
            
            # Velocity penalty: ρ_v||u_k||²
            P_diag[-self.m:] = [p + self.rho_v for p in P_diag[-self.m:]]
        
        P = sparse.diags(P_diag, format='csc')
        q = np.array(q_vec)
        
        # Constraints
        constraints_A = []
        constraints_l = []
        constraints_u = []
        
        # 1. Dynamics constraints: x_{k+1} = A x_k + B u_k
        # (Not enforced in simplified version - could add as equality constraints)
        
        # 2. CBF constraints: h(x_k) ≥ ε_safe for all k
        eps_safe = self.adaptive_safety_margin(sigma_pred)
        
        # Simplified: Only enforce at current state
        import torch
        x0_torch = torch.FloatTensor(x0).unsqueeze(0)
        h0 = self.cbf(x0_torch).item()
        
        if h0 < eps_safe:
            # Add CBF derivative constraint
            grad_h_u, rhs_cbf = self.compute_cbf_constraint(x0)
            
            # Constraint: grad_h_u^T · u_0 ≥ rhs_cbf
            constraint_row = np.zeros(n_vars)
            constraint_row[0:self.m] = grad_h_u
            
            constraints_A.append(constraint_row)
            constraints_l.append(rhs_cbf)
            constraints_u.append(np.inf)
        
        # 3. Control bounds: u_min ≤ u_k ≤ u_max
        for k in range(self.H):
            for j in range(self.m):
                row = np.zeros(n_vars)
                row[k * self.m + j] = 1.0
                
                constraints_A.append(row)
                constraints_l.append(self.u_min)
                constraints_u.append(self.u_max)
        
        # Stack constraints
        if len(constraints_A) > 0:
            A = sparse.csc_matrix(np.array(constraints_A))
            l = np.array(constraints_l)
            u = np.array(constraints_u)
        else:
            A = sparse.csc_matrix((0, n_vars))
            l = np.array([])
            u = np.array([])
        
        return P, q, A, l, u
    
    def solve(self, x0: np.ndarray, 
              context: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve MPC problem
        
        Args:
            x0: Current state [n]
            context: Optional task context for adaptive weighting
        
        Returns:
            u_opt: Optimal control sequence [H x m]
            info: Solve information
        """
        t_start = time.time()
        
        # Human motion prediction
        p_h = x0[12:30]
        v_h = x0[30:48]
        a_h_pred, sigma_pred = self.gp.predict(p_h, v_h, horizon=self.H)
        
        # Reference trajectory (previous solution shifted)
        x_ref = np.tile(x0, (self.H + 1, 1))
        u_ref = self.u_prev
        
        # Formulate QP
        P, q, A, l, u = self.formulate_qp(x0, x_ref, u_ref, a_h_pred, sigma_pred)
        
        # Solve
        if not self.solver_initialized:
            self.solver.setup(P, q, A, l, u, 
                            verbose=False, 
                            eps_abs=1e-4, 
                            eps_rel=1e-4,
                            max_iter=4000)
            self.solver_initialized = True
        else:
            self.solver.update(q=q, l=l, u=u)
        
        result = self.solver.solve()
        
        t_solve = time.time() - t_start
        self.solve_times.append(t_solve * 1000)  # Convert to ms
        
        # Extract solution
        if result.info.status == 'solved':
            U_opt = result.x.reshape(self.H, self.m)
            u_opt = U_opt[0]  # Return first control
            
            # Update warm start
            self.u_prev = np.vstack([U_opt[1:], U_opt[-1:]])
        else:
            print(f"Warning: QP not solved. Status: {result.info.status}")
            u_opt = np.zeros(self.m)
            U_opt = np.zeros((self.H, self.m))
        
        info = {
            'solve_time_ms': t_solve * 1000,
            'status': result.info.status,
            'iterations': result.info.iter,
            'epsilon_safe': self.adaptive_safety_margin(sigma_pred),
            'sigma_pred': sigma_pred
        }
        
        return u_opt, info