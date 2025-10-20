"""
Lipschitz-Constrained Neural Network for Control Barrier Functions
Implements spectral normalization to enforce global Lipschitz constant ≤ 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralNormLinear(nn.Module):
    """Linear layer with spectral normalization"""
    
    def __init__(self, in_features: int, out_features: int, L_max: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.L_max = L_max
        
        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Power iteration vector
        self.register_buffer('u', torch.randn(out_features))
        self.register_buffer('v', torch.randn(in_features))
        
    def _spectral_norm(self, n_iterations: int = 1):
        """Compute spectral norm via power iteration"""
        u = self.u
        v = self.v
        
        for _ in range(n_iterations):
            # Power iteration
            v = F.normalize(torch.mv(self.weight.t(), u), dim=0)
            u = F.normalize(torch.mv(self.weight, v), dim=0)
        
        # Update buffers
        self.u.copy_(u)
        self.v.copy_(v)
        
        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(self.weight, v))
        return sigma
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            sigma = self._spectral_norm(n_iterations=1)
        else:
            with torch.no_grad():
                sigma = self._spectral_norm(n_iterations=5)
        
        # Normalize weight if exceeds L_max
        weight = self.weight
        if sigma > self.L_max:
            weight = self.weight / (sigma / self.L_max)
        
        return F.linear(x, weight, self.bias)


class LipschitzCBFNetwork(nn.Module):
    """
    Lipschitz-constrained neural network for learning Control Barrier Functions
    
    Architecture:
        Input (38) -> Dense(128) -> tanh -> Dense(64) -> tanh -> Dense(32) -> tanh -> Output(1)
    
    Global Lipschitz constant: L_h ≤ 1.0
    """
    
    def __init__(self, 
                 input_dim: int = 38,
                 hidden_dims: list = [128, 64, 32],
                 L_max: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.L_max = L_max
        
        # Build layers with spectral normalization
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(SpectralNormLinear(prev_dim, hidden_dim, L_max))
            layers.append(nn.Tanh())  # Lipschitz constant = 1
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(SpectralNormLinear(prev_dim, 1, L_max))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better training"""
        for module in self.modules():
            if isinstance(module, SpectralNormLinear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: State tensor [batch_size, 38]
                [q_r(6), q_dot_r(6), p_h(18), v_h(18), s_obj(12), s_conv(1), s_task(5)]
        
        Returns:
            h: Barrier function value [batch_size, 1]
        """
        return self.network(x)
    
    def compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇h(x) for CBF condition
        
        Args:
            x: State tensor [batch_size, 38]
        
        Returns:
            grad_h: Gradient [batch_size, 38]
        """
        x.requires_grad_(True)
        h = self.forward(x)
        grad_h = torch.autograd.grad(
            outputs=h.sum(),
            inputs=x,
            create_graph=True
        )[0]
        return grad_h
    
    def lipschitz_constant(self) -> float:
        """
        Compute global Lipschitz constant (product of layer Lipschitz constants)
        
        Returns:
            L_global: Global Lipschitz constant
        """
        L_global = 1.0
        
        for module in self.modules():
            if isinstance(module, SpectralNormLinear):
                sigma = module._spectral_norm(n_iterations=10)
                L_global *= min(sigma.item(), self.L_max)
            elif isinstance(module, nn.Tanh):
                L_global *= 1.0  # Tanh has Lipschitz constant = 1
        
        return L_global


def test_lipschitz_network():
    """Test Lipschitz constraint"""
    model = LipschitzCBFNetwork(input_dim=38, L_max=1.0)
    
    # Random inputs
    x1 = torch.randn(100, 38)
    x2 = torch.randn(100, 38)
    
    # Forward pass
    h1 = model(x1)
    h2 = model(x2)
    
    # Check Lipschitz constraint: ||h(x1) - h(x2)|| ≤ L * ||x1 - x2||
    diff_h = torch.norm(h1 - h2, dim=1)
    diff_x = torch.norm(x1 - x2, dim=1)
    
    L_empirical = (diff_h / (diff_x + 1e-8)).max().item()
    L_network = model.lipschitz_constant()
    
    print(f"Empirical Lipschitz constant: {L_empirical:.4f}")
    print(f"Network Lipschitz constant: {L_network:.4f}")
    print(f"Constraint satisfied: {L_empirical <= L_network + 0.1}")


if __name__ == "__main__":
    test_lipschitz_network()