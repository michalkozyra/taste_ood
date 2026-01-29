"""
RobustBench attack wrappers for generating adversarial examples.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from .attack_config import AttackConfig


class PGDAttack:
    """Projected Gradient Descent attack (iterative FGSM with projection)."""
    
    def __init__(self, model: nn.Module, config: AttackConfig, device: torch.device,
                 mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        # Note: model is already wrapped to handle normalization
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Generate PGD adversarial examples.
        
        Args:
            x: Clean images (B, C, H, W) in [0, 1] range
            y: True labels (B,)
        
        Returns:
            Adversarial examples (B, C, H, W) in [0, 1] range
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Initialize adversarial examples
        x_adv = x.clone().detach()
        
        # Compute step size if not provided
        steps = self.config.steps or 50
        if self.config.step_size is None:
            if self.config.threat_model == 'linf':
                step_size = self.config.epsilon / steps * 2  # Conservative step size
            else:  # l2
                step_size = self.config.epsilon / steps * 2
        else:
            step_size = self.config.step_size
        
        # Iterative attack
        for _ in range(steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            # Forward pass
            logits = self.model(x_adv)
            loss = torch.nn.functional.cross_entropy(logits, y)
            
            # Backward pass
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
            
            # Update with gradient step
            if self.config.threat_model == 'linf':
                # L∞: sign of gradient
                perturbation = step_size * torch.sign(grad)
            elif self.config.threat_model == 'l2':
                # L2: normalized gradient
                grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
                grad_norm = grad_norm.view(-1, 1, 1, 1)
                perturbation = step_size * grad / (grad_norm + 1e-8)
            else:
                raise ValueError(f"PGD not supported for threat model: {self.config.threat_model}")
            
            x_adv = x_adv + perturbation
            
            # Project to epsilon ball around original image
            if self.config.threat_model == 'linf':
                # L∞: clip to [x - epsilon, x + epsilon]
                delta = x_adv - x
                delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
                x_adv = x + delta
            elif self.config.threat_model == 'l2':
                # L2: project to L2 ball
                delta = x_adv - x
                delta_flat = delta.view(delta.size(0), -1)
                delta_norm = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
                # Clip if norm exceeds epsilon
                scale = torch.clamp(self.config.epsilon / (delta_norm + 1e-8), max=1.0)
                delta_flat = delta_flat * scale
                delta = delta_flat.view(delta.shape)
                x_adv = x + delta
            
            # Clip to valid range [0, 1]
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()


def create_attack(
    config: AttackConfig,
    model: nn.Module,
    device: torch.device,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> object:
    """
    Create an attack instance based on configuration.
    
    Args:
        config: Attack configuration
        model: Target model to attack (expects normalized inputs)
        device: Device to run attack on
        mean: Normalization mean (for model wrapper, default: CIFAR-10)
        std: Normalization std (for model wrapper, default: CIFAR-10)
    
    Returns:
        Attack instance (AutoAttack, PGD, etc.)
    """
    model.eval()
    
    # Default normalization (CIFAR-10)
    if mean is None:
        mean = (0.4914, 0.4822, 0.4465)
    if std is None:
        std = (0.2023, 0.1994, 0.2010)
    
    # Create model wrapper that handles normalization
    # Attacks work on [0, 1] images, but our model expects normalized inputs
    class NormalizedModel(nn.Module):
        def __init__(self, base_model, mean, std, device):
            super().__init__()
            self.base_model = base_model
            self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        
        def forward(self, x):
            # x is in [0, 1] range, normalize it
            x_norm = (x - self.mean) / self.std
            return self.base_model(x_norm)
    
    wrapped_model = NormalizedModel(model, mean, std, device)
    wrapped_model.eval()
    
    if config.attack_type == 'autoattack':
        try:
            from autoattack import AutoAttack
        except ImportError:
            raise ImportError(
                "AutoAttack is not installed. Install it with: "
                "pip install git+https://github.com/fra31/auto-attack.git"
            )
        
        # AutoAttack expects norm to be 'Linf', 'L2', or 'L1' (specific casing)
        norm_map = {
            'linf': 'Linf',
            'l2': 'L2',
            'l1': 'L1',
        }
        norm = norm_map.get(config.threat_model.lower())
        if norm is None:
            raise ValueError(f"AutoAttack does not support threat model: {config.threat_model}")
        
        # AutoAttack expects model to return logits and work on [0, 1] images
        # Our wrapper handles the normalization
        attack = AutoAttack(
            wrapped_model,
            norm=norm,
            eps=config.epsilon,
            version='standard',  # or 'rand' or 'custom'
            device=device,
            verbose=config.verbose,
        )
        return attack
    
    elif config.attack_type == 'pgd':
        # Implement PGD directly (iterative FGSM with projection)
        # Use wrapped model that handles normalization
        return PGDAttack(wrapped_model, config, device, mean, std)
    
    elif config.attack_type == 'fgsm':
        # FGSM is a simple one-step attack
        # We'll implement it directly
        # FGSM works on [0, 1] images, model wrapper handles normalization
        return FGSMAttack(wrapped_model, config, device, mean, std)
    
    elif config.attack_type == 'apgd':
        try:
            from autoattack.autopgd_base import APGDAttack
        except ImportError:
            raise ImportError(
                "AutoAttack is not installed. Install it with: "
                "pip install git+https://github.com/fra31/auto-attack.git"
            )
        
        # APGDAttack expects norm as a string: 'Linf' or 'L2'
        if config.threat_model == 'linf':
            norm = 'Linf'
        elif config.threat_model == 'l2':
            norm = 'L2'
        else:
            raise ValueError(f"APGDAttack not supported for threat model: {config.threat_model}")
        
        attack = APGDAttack(
            wrapped_model,
            norm=norm,
            eps=config.epsilon,
            n_iter=config.steps or 100,
            n_restarts=config.restarts or 1,
            device=device,
            verbose=config.verbose,
        )
        return attack
    
    elif config.attack_type == 'square':
        try:
            from autoattack.square import SquareAttack
        except ImportError:
            raise ImportError(
                "AutoAttack is not installed. Install it with: "
                "pip install git+https://github.com/fra31/auto-attack.git"
            )
        
        attack = SquareAttack(
            wrapped_model,
            norm=config.threat_model.upper(),
            eps=config.epsilon,
            n_queries=config.n_queries or 5000,
            device=device,
            verbose=config.verbose,
        )
        return attack
    
    else:
        raise ValueError(f"Unsupported attack type: {config.attack_type}")


class FGSMAttack:
    """Fast Gradient Sign Method attack."""
    
    def __init__(self, model: nn.Module, config: AttackConfig, device: torch.device, 
                 mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        # Note: model is already wrapped to handle normalization
    
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Generate FGSM adversarial examples.
        
        Args:
            x: Clean images (B, C, H, W)
            y: True labels (B,)
        
        Returns:
            Adversarial examples (B, C, H, W)
        """
        x = x.to(self.device).requires_grad_(True)
        y = y.to(self.device)
        
        # Forward pass
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Backward pass
        grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
        
        # Generate perturbation
        if self.config.threat_model == 'linf':
            # L∞: sign of gradient
            perturbation = self.config.epsilon * torch.sign(grad)
        elif self.config.threat_model == 'l2':
            # L2: normalized gradient
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            perturbation = self.config.epsilon * grad / (grad_norm + 1e-8)
        else:
            raise ValueError(f"FGSM not supported for threat model: {self.config.threat_model}")
        
        # Clip to valid range [0, 1] (assuming images are normalized)
        # Note: We need to handle normalization - for now assume images are in [0, 1]
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()


def generate_adversarial_batch(
    attack: object,
    x: Tensor,
    y: Tensor,
    config: AttackConfig,
) -> Tuple[Tensor, Tensor]:
    """
    Generate adversarial examples for a batch.
    
    Args:
        attack: Attack instance
        x: Clean images (B, C, H, W) - should be in [0, 1] range (not normalized)
        y: True labels (B,)
        config: Attack configuration
    
    Returns:
        x_adv: Adversarial examples (B, C, H, W) - in [0, 1] range
        y: Original labels (B,) - unchanged
    """
    if config.attack_type == 'autoattack':
        # AutoAttack has a different API - it processes batches internally
        # We'll handle this in the dataset generation function
        # For now, use the perturb method if available
        if hasattr(attack, 'perturb'):
            x_adv = attack.perturb(x, y)
        else:
            # Fallback: use run_standard_evaluation (but this processes full dataset)
            # This is handled differently in generate_adversarial_dataset
            raise NotImplementedError(
                "AutoAttack batch generation should be handled in generate_adversarial_dataset"
            )
        return x_adv, y
    
    elif config.attack_type == 'pgd':
        # Our custom PGD
        x_adv = attack(x, y)
        return x_adv, y
    
    elif config.attack_type == 'apgd':
        # APGD from AutoAttack (if available)
        x_adv = attack.perturb(x, y)
        return x_adv, y
    
    elif config.attack_type == 'fgsm':
        # Our custom FGSM
        x_adv = attack(x, y)
        return x_adv, y
    
    elif config.attack_type == 'square':
        # Square attack
        x_adv = attack.perturb(x, y)
        return x_adv, y
    
    else:
        raise ValueError(f"Unsupported attack type: {config.attack_type}")
