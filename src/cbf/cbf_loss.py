"""
Multi-component loss function for CBF training
"""

import torch
import torch.nn as nn


class CBFLoss(nn.Module):
    """
    Multi-component loss for CBF training
    
    L = λ₁·L_safety + λ₂·L_validity + λ₃·L_smooth + λ₄·L_CBF
    """
    
    def __init__(self, 
                 safety: float = 10.0,
                 validity: float = 5.0,
                 smoothness: float = 0.1,
                 cbf_decrease: float = 2.0,
                 alpha_coeff: float = 0.5):
        super().__init__()
        
        self.lambda_safety = safety
        self.lambda_validity = validity
        self.lambda_smoothness = smoothness
        self.lambda_cbf = cbf_decrease
        self.alpha_coeff = alpha_coeff  # α(h) = alpha_coeff * h
    
    def safety_loss(self, h_pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        L_safety: Ensures correct classification of safe/unsafe states
        
        For safe states (label=1): penalize h < 0
        For unsafe states (label=0): penalize h > 0
        """
        safe_states = labels == 1
        unsafe_states = labels == 0
        
        loss = 0.0
        
        # Safe states should have h ≥ 0
        if safe_states.any():
            loss += torch.mean(torch.clamp(-h_pred[safe_states], min=0))
        
        # Unsafe states should have h < 0
        if unsafe_states.any():
            loss += torch.mean(torch.clamp(h_pred[unsafe_states], min=0))
        
        return loss
    
    def validity_loss(self, 
                      h_pred: torch.Tensor, 
                      states: torch.Tensor,
                      safe_mask: torch.Tensor,
                      model: nn.Module) -> torch.Tensor:
        """
        L_validity: Enforces CBF condition on safe states
        
        For safe states: sup_u [L_f h + L_g h · u + α(h)] ≥ 0
        
        Simplified: We check that ∇h is not pointing inward too aggressively
        """
        if not safe_mask.any():
            return torch.tensor(0.0, device=h_pred.device)
        
        safe_states = states[safe_mask]
        safe_h = h_pred[safe_mask]
        
        # Compute gradient
        grad_h = model.compute_gradient(safe_states)
        
        # α(h) = alpha_coeff * h (linear class-K function)
        alpha_h = self.alpha_coeff * safe_h
        
        # Simplified CBF condition: -α(h) should be achievable
        # (assumes there exists control u making L_f h + L_g h · u ≥ -α(h))
        # We penalize when this would require unrealistic control
        
        # Proxy: penalize large negative h with small gradient
        gradient_magnitude = torch.norm(grad_h, dim=1, keepdim=True)
        validity_violation = torch.clamp(-alpha_h - gradient_magnitude, min=0)
        
        return torch.mean(validity_violation ** 2)
    
    def smoothness_loss(self, states: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        L_smooth: Encourages smooth barrier gradients
        
        Penalizes large gradients to avoid overfitting
        """
        grad_h = model.compute_gradient(states)
        return torch.mean(torch.norm(grad_h, dim=1) ** 2)
    
    def cbf_decrease_loss(self,
                          h_current: torch.Tensor,
                          h_next: torch.Tensor,
                          epsilon_tol: float = 0.01) -> torch.Tensor:
        """
        L_CBF: Penalizes barrier decrease on trajectories
        
        For consecutive states (x_t, x_{t+1}): 
        penalize h(x_t) - h(x_{t+1}) > ε_tol
        
        Note: This requires trajectory data, so we apply it only when available
        """
        decrease = h_current - h_next
        violation = torch.clamp(decrease - epsilon_tol, min=0)
        return torch.mean(violation ** 2)
    
    def forward(self,
                h_pred: torch.Tensor,
                labels: torch.Tensor,
                states: torch.Tensor,
                safe_mask: torch.Tensor,
                model: nn.Module) -> dict:
        """
        Compute total loss
        
        Returns:
            Dictionary with all loss components
        """
        # Individual losses
        L_safety = self.safety_loss(h_pred, labels)
        L_validity = self.validity_loss(h_pred, states, safe_mask, model)
        L_smooth = self.smoothness_loss(states, model)
        
        # Total loss
        L_total = (
            self.lambda_safety * L_safety +
            self.lambda_validity * L_validity +
            self.lambda_smoothness * L_smooth
        )
        
        return {
            'total': L_total,
            'safety': L_safety,
            'validity': L_validity,
            'smoothness': L_smooth,
            'cbf_decrease': torch.tensor(0.0)  # Placeholder
        }