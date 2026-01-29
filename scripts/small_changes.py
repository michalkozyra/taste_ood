"""
Simple 2D Gaussian example with Stein residual computation.

1. Simulate data from 2D standard Gaussian: X ~ N(0, I)
2. Create target: y = x2 - x1
3. Train a simple FC network to model X -> y
4. Calculate Stein residuals using Laplacian formulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------
# Data Generation
# ---------------------------

def generate_data(n_samples=1000, seed=42, sigma=0.1):
    """
    Generate data from 2D standard Gaussian.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        sigma: Noise scale parameter (default: 0.1). Noise is N(0, sigma * |x2 - x1|)

    Returns:
        X: (n_samples, 2) tensor of 2D Gaussian samples
        y: (n_samples,) tensor where y = x2 - x1 + N(0, sigma * |x2 - x1|)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate 2D standard Gaussian: X ~ N(0, I)
    X = torch.randn(n_samples, 2)

    # Create target: y = x2 - x1 + noise, where noise ~ N(0, sigma * |x2 - x1|)
    y_base = X[:, 1] - X[:, 0]  # (n_samples,)
    noise_std = sigma * torch.abs(y_base)  # (n_samples,)
    noise = torch.randn(n_samples) * noise_std  # (n_samples,)
    y = y_base + noise  # (n_samples,)

    return X, y


def rotation_matrix_2d(angle_degrees):
    """
    Create a 2D rotation matrix.
    
    Args:
        angle_degrees: Rotation angle in degrees
    
    Returns:
        2x2 rotation matrix as torch tensor
    """
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    R = torch.tensor([[cos_a, -sin_a],
                      [sin_a, cos_a]], dtype=torch.float32)
    return R


def generate_shifted_data(n_samples=1000, epsilon=1.0, angle_degrees=0.0, seed=43, sigma=0.1):
    """
    Generate data from 2D Gaussian with shifted mean in a rotated direction.

    Args:
        n_samples: Number of samples to generate
        epsilon: Shift magnitude (radius)
        angle_degrees: Rotation angle in degrees (0 degrees points in direction [1, 1])
        seed: Random seed for reproducibility
        sigma: Noise scale parameter (default: 0.1). Noise is N(0, sigma * |x2 - x1|)

    Returns:
        X: (n_samples, 2) tensor of 2D Gaussian samples with mean epsilon * rotated([1, 1])
        y: (n_samples,) tensor where y = x2 - x1 + N(0, sigma * |x2 - x1|)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Base direction [1, 1] normalized to unit vector
    base_direction = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)  # Unit vector in [1,1] direction
    
    # Apply rotation matrix
    R = rotation_matrix_2d(angle_degrees)
    rotated_direction = R @ base_direction  # (2,)
    
    # Generate 2D Gaussian with shifted mean: X ~ N(epsilon * rotated_direction, I)
    mean = epsilon * rotated_direction
    X = torch.randn(n_samples, 2) + mean

    # Create target: y = x2 - x1 + noise, where noise ~ N(0, sigma * |x2 - x1|)
    y_base = X[:, 1] - X[:, 0]  # (n_samples,)
    noise_std = sigma * torch.abs(y_base)  # (n_samples,)
    noise = torch.randn(n_samples) * noise_std  # (n_samples,)
    y = y_base + noise  # (n_samples,)

    return X, y


# ---------------------------
# Simple FC Network
# ---------------------------

class SimpleFCNet(nn.Module):
    """
    Simple fully connected network for regression: X (2D) -> y (scalar)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, 2)
        # returns: (B, 1)
        return self.net(x).squeeze(-1)  # (B,)


# ---------------------------
# Training
# ---------------------------

def train_model(model, X_train, y_train, epochs=100, lr=1e-3, batch_size=32, weight_decay=1e-5):
    """
    Train the FC network.

    Args:
        model: SimpleFCNet instance
        X_train: Training features (N, 2)
        y_train: Training targets (N,)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        weight_decay: L2 penalty (weight decay) for regularization (default: 1e-5)

    Returns:
        Trained model
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create dataloader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    return model


# ---------------------------
# Stein Residual Computation
# ---------------------------

def compute_grad_f_simple(x: torch.Tensor, f_model: nn.Module, device: torch.device):
    """
    Compute gradient of scalar output f(x) w.r.t. input x.
    For regression, f(x) is the model output directly.

    Args:
        x: Input tensor (B, 2)
        f_model: Trained model
        device: Device to use

    Returns:
        grads: Gradient of f w.r.t. x (B, 2)
        f_vals: Scalar output values (B,)
        x_req: Input tensor with requires_grad=True
    """
    x = x.clone().detach().to(device).requires_grad_(True)
    f_vals = f_model(x)  # (B,)
    grads = torch.autograd.grad(f_vals.sum(), x, create_graph=True)[0]  # (B, 2)
    return grads, f_vals, x


def hutchinson_laplacian_simple(x_req_grad: torch.Tensor, grads: torch.Tensor,
                                num_probes: int, device: torch.device, probe='rademacher'):
    """
    Hutchinson estimator for Laplacian (trace of Hessian) per sample.

    Args:
        x_req_grad: Input tensor with requires_grad=True (B, 2)
        grads: Gradient of f w.r.t. x (B, 2)
        num_probes: Number of random probes for estimation
        device: Device to use
        probe: 'rademacher' or 'gaussian'

    Returns:
        lap_est: (B,) tensor approximating Laplacian per sample
    """
    B = x_req_grad.shape[0]
    lap = torch.zeros(B, device=device)

    for _ in range(num_probes):
        if probe == 'rademacher':
            v = (torch.randint(0, 2, (B, 2), device=device).float() * 2.0 - 1.0).requires_grad_(False)
        else:
            v = torch.randn(B, 2, device=device).requires_grad_(False)

        # Inner product: v^T * grad f
        inner = (grads * v).sum(dim=1)  # (B,)

        # Hessian-vector product: grad(inner, x)
        Hv = torch.autograd.grad(inner.sum(), x_req_grad, retain_graph=True)[0]  # (B, 2)

        # v^T H v estimates trace(H) = Laplacian
        vtHv = (v * Hv).sum(dim=1)  # (B,)
        lap += vtHv

    lap = lap / float(num_probes)
    return lap


def compute_stein_residuals_simple(X: torch.Tensor, model: nn.Module,
                                    device: torch.device, num_probes=1):
    """
    Compute Stein residuals using Laplacian formulation.

    For 2D standard Gaussian, the score function is known: s(x) = -x
    (since log p(x) = -0.5 * ||x||^2 + const, so grad log p = -x)

    Stein residual: r(x) = Laplacian(f(x)) + s(x)^T grad f(x)

    Args:
        X: Input data (N, 2)
        model: Trained model
        device: Device to use
        num_probes: Number of Hutchinson probes for Laplacian
        use_score: If True, use known score function s(x) = -x. If False, only compute Laplacian.

    Returns:
        residuals: (N,) array of Stein residuals
        f_vals: (N,) array of function values
    """
    model.eval()
    residuals = []
    f_vals_list = []

    # Process in batches to avoid memory issues
    batch_size = 128
    dataloader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    with torch.enable_grad():
        for (X_batch,) in dataloader:
            X_batch = X_batch.to(device)

            # Compute gradient of f
            grads, f_scalar, X_req = compute_grad_f_simple(X_batch, model, device)

            # Compute Hutchinson Laplacian
            lap = hutchinson_laplacian_simple(X_req, grads, num_probes=num_probes, device=device)


            # For 2D standard Gaussian: s(x) = -x
            s = -X_req  # (B, 2)
            # Compute s^T grad f
            s_dot_grad = (s * grads).sum(dim=1)  # (B,)

            # Stein residual: r(x) = Laplacian(f(x)) + s(x)^T grad f(x)
            r = lap + s_dot_grad

            residuals.extend(r.detach().cpu().numpy().tolist())
            f_vals_list.extend(f_scalar.detach().cpu().numpy().tolist())

    return np.array(residuals), np.array(f_vals_list)


def compute_stein_residuals_alternative_simple(X: torch.Tensor, model: nn.Module,
                                               device: torch.device, num_probes=1):
    """
    Compute Stein residuals using alternative Stein operator.

    For 2D standard Gaussian, the score function is known: s(x) = -x
    (since log p(x) = -0.5 * ||x||^2 + const, so grad log p = -x)

    Stein residual: r(x) = ||f(x) * s(x) + grad_f(x)||_2 (L2 norm over spatial dimensions)

    Args:
        X: Input data (N, 2)
        model: Trained model
        device: Device to use
        num_probes: Not used (kept for compatibility)

    Returns:
        residuals: (N,) array of Stein residuals
        f_vals: (N,) array of function values
    """
    model.eval()
    residuals = []
    f_vals_list = []

    # Process in batches to avoid memory issues
    batch_size = 128
    dataloader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)

    with torch.enable_grad():
        for (X_batch,) in dataloader:
            X_batch = X_batch.to(device)

            # Compute gradient of f
            grads, f_scalar, X_req = compute_grad_f_simple(X_batch, model, device)

            # For 2D standard Gaussian: s(x) = -x
            s = -X_req  # (B, 2)
            
            # Alternative Stein operator: r(x) = ||f(x) * s(x) + grad_f(x)||_2
            # f_scalar is (B,), s is (B, 2), grads is (B, 2)
            # Broadcast f_scalar to (B, 1) then multiply with s
            f_expanded = f_scalar.unsqueeze(1)  # (B, 1)
            stein_term = f_expanded * s + grads  # (B, 2)
            r = torch.norm(stein_term, dim=1)  # (B,) - L2 norm over spatial dimensions

            residuals.extend(r.detach().cpu().numpy().tolist())
            f_vals_list.extend(f_scalar.detach().cpu().numpy().tolist())

    return np.array(residuals), np.array(f_vals_list)


# ---------------------------
# Log-Likelihood Computation
# ---------------------------

def compute_log_likelihood_standard_gaussian(X: torch.Tensor):
    """
    Compute log-likelihood of X under 2D standard Gaussian N(0, I).

    For 2D standard Gaussian: log p(x) = -0.5 * ||x||^2 - log(2*pi)

    Args:
        X: Input data (N, 2)

    Returns:
        log_likelihoods: (N,) array of log-likelihoods per sample
    """
    # Log-likelihood for 2D standard Gaussian: log p(x) = -0.5 * ||x||^2 - log(2*pi)
    # For 2D: log(2*pi) = log(2*pi) = constant
    log_2pi = np.log(2 * np.pi)

    # Compute squared norm per sample
    squared_norm = (X ** 2).sum(dim=1)  # (N,)

    # Log-likelihood: -0.5 * ||x||^2 - log(2*pi)
    log_likelihoods = -0.5 * squared_norm - log_2pi

    return log_likelihoods.cpu().numpy()


# ---------------------------
# Evaluation
# ---------------------------

def evaluate_shifted_data(model: nn.Module, X_shifted: torch.Tensor, y_shifted: torch.Tensor,
                          device: torch.device, epsilon: float, angle_degrees: float, num_probes=4,
                          use_alternative_stein=False):
    """
    Evaluate model on shifted test data.

    Computes:
    1. Test MSE
    2. Log-likelihood of X under original standard Gaussian
    3. Stein residuals

    Args:
        model: Trained model
        X_shifted: Shifted test data (N, 2)
        y_shifted: True targets for shifted data (N,)
        device: Device to use
        epsilon: Shift magnitude (radius) used
        angle_degrees: Rotation angle in degrees used
        num_probes: Number of Hutchinson probes for Stein residual computation

    Returns:
        results: Dictionary containing all computed metrics
    """
    model.eval()

    # 1. Test MSE
    with torch.no_grad():
        y_pred = model(X_shifted)
        test_mse = ((y_pred - y_shifted) ** 2).mean().item()

    # 2. Log-likelihood under original standard Gaussian
    log_likelihoods = compute_log_likelihood_standard_gaussian(X_shifted)
    mean_log_likelihood = log_likelihoods.mean()
    std_log_likelihood = log_likelihoods.std()

    # 3. Stein residuals
    if use_alternative_stein:
        residuals, f_vals = compute_stein_residuals_alternative_simple(
            X_shifted, model, device, num_probes=num_probes
        )
    else:
        residuals, f_vals = compute_stein_residuals_simple(
            X_shifted, model, device, num_probes=num_probes
        )

    results = {
        'epsilon': epsilon,
        'angle_degrees': angle_degrees,
        'n_samples': len(X_shifted),
        'test_mse': test_mse,
        'log_likelihood': {
            'mean': float(mean_log_likelihood),
            'std': float(std_log_likelihood),
            'min': float(log_likelihoods.min()),
            'max': float(log_likelihoods.max()),
        },
        'stein_residuals': {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'median': float(np.median(residuals)),
            'min': float(residuals.min()),
            'max': float(residuals.max()),
        },
        'f_values': {
            'mean': float(f_vals.mean()),
            'std': float(f_vals.std()),
            'min': float(f_vals.min()),
            'max': float(f_vals.max()),
        }
    }

    return results, residuals, log_likelihoods, f_vals


def save_results(results: dict, residuals: np.ndarray, log_likelihoods: np.ndarray,
                 f_vals: np.ndarray, X_shifted: torch.Tensor, output_dir='results/small_changes_results', angle_degrees=None):
    """
    Save all evaluation results to disk.

    Args:
        results: Dictionary of summary statistics
        residuals: Array of Stein residuals
        log_likelihoods: Array of log-likelihoods
        f_vals: Array of function values
        X_shifted: Shifted test data
        output_dir: Directory to save results
        angle_degrees: Angle in degrees (for per-angle saves)
    """
    os.makedirs(output_dir, exist_ok=True)

    # If angle is provided, save in angle-specific subdirectory
    if angle_degrees is not None:
        # Use integer representation if angle is an integer, otherwise use formatted float
        if angle_degrees == int(angle_degrees):
            angle_str = f'angle_{int(angle_degrees)}'
        else:
            angle_str = f'angle_{angle_degrees:.1f}'.rstrip('0').rstrip('.')
        angle_dir = os.path.join(output_dir, angle_str)
        os.makedirs(angle_dir, exist_ok=True)
        save_path = angle_dir
    else:
        save_path = output_dir

    # Save summary statistics as JSON
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save arrays as numpy files
    np.save(os.path.join(save_path, 'residuals.npy'), residuals)
    np.save(os.path.join(save_path, 'log_likelihoods.npy'), log_likelihoods)
    np.save(os.path.join(save_path, 'f_values.npy'), f_vals)
    np.save(os.path.join(save_path, 'X_shifted.npy'), X_shifted.cpu().numpy())

    if angle_degrees is not None:
        print(f'  Results for angle={angle_degrees}° saved to {save_path}/')


def save_all_results(all_results: list, output_dir='results/small_changes_results'):
    """
    Save aggregated results across all angle values.

    Args:
        all_results: List of tuples (epsilon, angle_degrees, results_dict, residuals, log_likelihoods, f_vals, X_shifted)
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create summary dictionary with all angles
    summary = {
        'epsilon': all_results[0][0] if all_results else None,  # Single epsilon value
        'angles': [],
        'test_mse': [],
        'log_likelihood_mean': [],
        'log_likelihood_std': [],
        'stein_residual_mean': [],
        'stein_residual_std': [],
        'stein_residual_median': [],
    }

    for epsilon, angle_degrees, results, residuals, log_likelihoods, f_vals, X_shifted in all_results:
        summary['angles'].append(float(angle_degrees))
        summary['test_mse'].append(results['test_mse'])
        summary['log_likelihood_mean'].append(results['log_likelihood']['mean'])
        summary['log_likelihood_std'].append(results['log_likelihood']['std'])
        summary['stein_residual_mean'].append(results['stein_residuals']['mean'])
        summary['stein_residual_std'].append(results['stein_residuals']['std'])
        summary['stein_residual_median'].append(results['stein_residuals']['median'])

    # Save summary
    with open(os.path.join(output_dir, 'summary_all_angles.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save arrays for all angles
    angles_array = np.array([angle for _, angle, _, _, _, _, _ in all_results])
    test_mse_array = np.array([r['test_mse'] for _, _, r, _, _, _, _ in all_results])
    log_likelihood_mean_array = np.array([r['log_likelihood']['mean'] for _, _, r, _, _, _, _ in all_results])
    stein_residual_mean_array = np.array([r['stein_residuals']['mean'] for _, _, r, _, _, _, _ in all_results])
    stein_residual_median_array = np.array([r['stein_residuals']['median'] for _, _, r, _, _, _, _ in all_results])

    np.save(os.path.join(output_dir, 'angles.npy'), angles_array)
    np.save(os.path.join(output_dir, 'test_mse_all.npy'), test_mse_array)
    np.save(os.path.join(output_dir, 'log_likelihood_mean_all.npy'), log_likelihood_mean_array)
    np.save(os.path.join(output_dir, 'stein_residual_mean_all.npy'), stein_residual_mean_array)
    np.save(os.path.join(output_dir, 'stein_residual_median_all.npy'), stein_residual_median_array)

    print(f'\nAggregated results saved to {output_dir}/')
    print(f'  - summary_all_angles.json')
    print(f'  - angles.npy')
    print(f'  - test_mse_all.npy')
    print(f'  - log_likelihood_mean_all.npy')
    print(f'  - stein_residual_mean_all.npy')
    print(f'  - stein_residual_median_all.npy')


# ---------------------------
# Visualization
# ---------------------------

def plot_results(all_results: list, output_dir='results/small_changes_results'):
    """
    Create visualization plots showing relationships between test MSE, Stein residuals, and log-likelihood.

    Args:
        all_results: List of tuples (epsilon, angle_degrees, results_dict, residuals, log_likelihoods, f_vals, X_shifted)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    epsilon = all_results[0][0] if all_results else None  # Single epsilon value
    angles = np.array([angle for _, angle, _, _, _, _, _ in all_results])
    test_mse = np.array([r['test_mse'] for _, _, r, _, _, _, _ in all_results])
    log_likelihood_mean = np.array([r['log_likelihood']['mean'] for _, _, r, _, _, _, _ in all_results])
    log_likelihood_std = np.array([r['log_likelihood']['std'] for _, _, r, _, _, _, _ in all_results])
    stein_residual_mean = np.array([r['stein_residuals']['mean'] for _, _, r, _, _, _, _ in all_results])
    stein_residual_std = np.array([r['stein_residuals']['std'] for _, _, r, _, _, _, _ in all_results])
    stein_residual_median = np.array([r['stein_residuals']['median'] for _, _, r, _, _, _, _ in all_results])
    
    # Compute mean of absolute residuals for each angle
    stein_residual_mean_abs = np.array([np.abs(residuals).mean() for _, _, _, residuals, _, _, _ in all_results])

    # Sort by angle for better visualization
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    test_mse = test_mse[sort_idx]
    log_likelihood_mean = log_likelihood_mean[sort_idx]
    log_likelihood_std = log_likelihood_std[sort_idx]
    stein_residual_mean = stein_residual_mean[sort_idx]
    stein_residual_std = stein_residual_std[sort_idx]
    stein_residual_median = stein_residual_median[sort_idx]
    stein_residual_mean_abs = stein_residual_mean_abs[sort_idx]

    # Plot 1: Individual metrics vs angle
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test MSE vs angle
    axes[0, 0].plot(angles, test_mse, 'o-', linewidth=2, markersize=6, color='blue')
    axes[0, 0].set_xlabel('Angle (degrees)', fontsize=12)
    axes[0, 0].set_ylabel('Test MSE', fontsize=12)
    axes[0, 0].set_title(f'Test MSE vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Log-likelihood vs angle
    axes[0, 1].plot(angles, log_likelihood_mean, 'o-', linewidth=2, markersize=6, color='green')
    axes[0, 1].fill_between(angles,
                            log_likelihood_mean - log_likelihood_std,
                            log_likelihood_mean + log_likelihood_std,
                            alpha=0.3, color='green')
    axes[0, 1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[0, 1].set_ylabel('Log-Likelihood (mean ± std)', fontsize=12)
    axes[0, 1].set_title(f'Log-Likelihood vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Stein residual mean vs angle
    axes[1, 0].plot(angles, stein_residual_mean, 'o-', linewidth=2, markersize=6, color='red', label='Mean residual')
    axes[1, 0].plot(angles, stein_residual_median, 's-', linewidth=2, markersize=6, color='orange', label='Median residual')
    axes[1, 0].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1, 0].set_ylabel('Stein Residual', fontsize=12)
    axes[1, 0].set_title(f'Stein Residuals vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Mean of absolute residuals vs angle
    axes[1, 1].plot(angles, stein_residual_mean_abs, 'o-', linewidth=2, markersize=6, color='purple')
    axes[1, 1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1, 1].set_ylabel('Mean |Stein Residual|', fontsize=12)
    axes[1, 1].set_title(f'Mean |Stein Residual| vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_angle.png'), dpi=150, bbox_inches='tight')
    print(f'  Saved: metrics_vs_angle.png')
    plt.close()

    # Also save each subplot as a standalone figure (paper-friendly / easy inclusion).
    # Keep filenames stable so downstream LaTeX can reference them.
    # 1) Test MSE vs angle
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(angles, test_mse, 'o-', linewidth=2, markersize=6, color='blue')
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Test MSE vs Angle', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_angle__test_mse.png'), dpi=200, bbox_inches='tight')
    print(f'  Saved: metrics_vs_angle__test_mse.png')
    plt.close(fig)

    # 2) Log-likelihood vs angle (mean ± std)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(angles, log_likelihood_mean, 'o-', linewidth=2, markersize=6, color='green')
    ax.fill_between(
        angles,
        log_likelihood_mean - log_likelihood_std,
        log_likelihood_mean + log_likelihood_std,
        alpha=0.3,
        color='green',
    )
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Log-Likelihood (mean ± std)', fontsize=12)
    ax.set_title(f'Log-Likelihood vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_angle__loglik.png'), dpi=200, bbox_inches='tight')
    print(f'  Saved: metrics_vs_angle__loglik.png')
    plt.close(fig)

    # 3) Stein residual mean/median vs angle
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(angles, stein_residual_mean, 'o-', linewidth=2, markersize=6, color='red', label='Mean residual')
    ax.plot(angles, stein_residual_median, 's-', linewidth=2, markersize=6, color='orange', label='Median residual')
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Stein Residual', fontsize=12)
    ax.set_title(f'Stein Residuals vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_angle__stein_residuals.png'), dpi=200, bbox_inches='tight')
    print(f'  Saved: metrics_vs_angle__stein_residuals.png')
    plt.close(fig)

    # 4) Mean absolute Stein residual vs angle
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(angles, stein_residual_mean_abs, 'o-', linewidth=2, markersize=6, color='purple')
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Mean |Stein Residual|', fontsize=12)
    ax.set_title(f'Mean |Stein Residual| vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_angle__mean_abs_stein.png'), dpi=200, bbox_inches='tight')
    print(f'  Saved: metrics_vs_angle__mean_abs_stein.png')
    plt.close(fig)

    # 5) Overlay: Stein residuals (mean/median) + log-likelihood mean (no CI)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax2 = ax1.twinx()

    # Stein residuals on left axis (mean only)
    l1 = ax1.plot(angles, stein_residual_mean, 'o-', linewidth=2, markersize=6, color='orange', label='TASTE')
    ax1.set_xlabel('Angle (degrees)', fontsize=12)
    ax1.set_ylabel('TASTE', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Log-likelihood mean on right axis (no CI band)
    l3 = ax2.plot(angles, log_likelihood_mean, 'o-', linewidth=2, markersize=6, color='green', label='Log-likelihood (mean)')
    ax2.set_ylabel('Log-Likelihood (mean)', fontsize=12)
    # Make the log-likelihood axis visually "flat" (fluctuations are estimation noise):
    # widen the y-range to ~10× the observed peak-to-peak variation.
    ll_mean = float(np.mean(log_likelihood_mean))
    ll_ptp = float(np.max(log_likelihood_mean) - np.min(log_likelihood_mean))
    ll_half_span = max(0.5, 5.0 * ll_ptp)  # half-span = 10×ptp / 2
    ax2.set_ylim(ll_mean - ll_half_span, ll_mean + ll_half_span)

    ax1.set_title('TASTE and Log-Likelihood vs Angle', fontsize=14, fontweight='bold')

    # Combined legend
    lines = list(l1) + list(l3)
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_angle__stein_vs_loglik_overlay.png'), dpi=200, bbox_inches='tight')
    print(f'  Saved: metrics_vs_angle__stein_vs_loglik_overlay.png')
    plt.close(fig)

    # Plot 2: Scatter plots showing relationships
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test MSE vs Log-likelihood
    axes[0, 0].scatter(log_likelihood_mean, test_mse, c=angles, cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    axes[0, 0].set_xlabel('Log-Likelihood (mean)', fontsize=12)
    axes[0, 0].set_ylabel('Test MSE', fontsize=12)
    axes[0, 0].set_title('Test MSE vs Log-Likelihood', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar.set_label('Angle (degrees)', fontsize=10)

    # Test MSE vs |Stein residual|
    axes[0, 1].scatter(stein_residual_mean_abs, test_mse, c=angles, cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    axes[0, 1].set_xlabel('Mean |Stein Residual|', fontsize=12)
    axes[0, 1].set_ylabel('Test MSE', fontsize=12)
    axes[0, 1].set_title('Test MSE vs |Stein Residual|', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Angle (degrees)', fontsize=10)

    # Log-likelihood vs Stein residual (median)
    axes[1, 0].scatter(stein_residual_median, log_likelihood_mean, c=angles, cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    axes[1, 0].set_xlabel('Stein Residual (median)', fontsize=12)
    axes[1, 0].set_ylabel('Log-Likelihood (mean)', fontsize=12)
    axes[1, 0].set_title('Log-Likelihood vs Stein Residual', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('Angle (degrees)', fontsize=10)

    # Test MSE vs Stein residual (mean)
    axes[1, 1].scatter(stein_residual_mean, test_mse, c=angles, cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=1)
    axes[1, 1].set_xlabel('Stein Residual (mean)', fontsize=12)
    axes[1, 1].set_ylabel('Test MSE', fontsize=12)
    axes[1, 1].set_title('Test MSE vs Stein Residual (mean)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Angle (degrees)', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relationships_scatter.png'), dpi=150, bbox_inches='tight')
    print(f'  Saved: relationships_scatter.png')
    plt.close()

    # Plot 3: Combined plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Test MSE', color=color1, fontsize=12)
    line1, = ax1.plot(angles, test_mse, 'o-', color=color1, linewidth=2, markersize=8, label='Test MSE')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Stein Residual (median)', color=color2, fontsize=12)
    line2, = ax2.plot(angles, stein_residual_median, 's-', color=color2, linewidth=2, markersize=8, label='Stein Residual')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'tab:green'
    ax3.set_ylabel('Log-Likelihood (mean)', color=color3, fontsize=12)
    line3, = ax3.plot(angles, log_likelihood_mean, '^-', color=color3, linewidth=2, markersize=8, label='Log-Likelihood')
    ax3.tick_params(axis='y', labelcolor=color3)

    # Combine legends
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Test MSE, Stein Residual, and Log-Likelihood vs Angle (ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=150, bbox_inches='tight')
    print(f'  Saved: combined_metrics.png')
    plt.close()

    # Plot 4: Correlation heatmap-style visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create correlation matrix
    metrics = np.column_stack([
        test_mse,
        log_likelihood_mean,
        stein_residual_mean,
        stein_residual_median,
        stein_residual_std
    ])

    # Normalize each metric to [0, 1] for visualization
    metrics_norm = (metrics - metrics.min(axis=0)) / (metrics.max(axis=0) - metrics.min(axis=0) + 1e-10)

    # Create a line plot showing normalized trends
    metric_names = ['Test MSE', 'Log-Likelihood', 'Stein Res Mean', 'Stein Res Median', 'Stein Res Std']
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, (name, color) in enumerate(zip(metric_names, colors)):
        ax.plot(angles, metrics_norm[:, i], 'o-', label=name, linewidth=2, markersize=6, color=color, alpha=0.8)

    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Value [0, 1]', fontsize=12)
    ax.set_title(f'Normalized Metrics vs Angle (All on Same Scale, ε={epsilon:.2f})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_metrics.png'), dpi=150, bbox_inches='tight')
    print(f'  Saved: normalized_metrics.png')
    plt.close()

    print(f'\nAll plots saved to {output_dir}/')


# ---------------------------
# Main Example
# ---------------------------

def main():
    """Main example: generate data, train model, compute Stein residuals."""
    # Config: choose Stein operator ('laplacian' or 'alternative')
    use_alternative_stein = False  # Set to False to use Laplacian-based operator
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Generate data
    print('\nGenerating data...')
    X, y = generate_data(n_samples=5000, seed=42, sigma=0.0)
    print(f'Generated {len(X)} samples from 2D standard Gaussian')
    print(f'X shape: {X.shape}, y shape: {y.shape}')
    print(f'X mean: {X.mean(dim=0)}, X std: {X.std(dim=0)}')

    # Split into train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Move to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Create and train model
    print('\nTraining model...')
    model = SimpleFCNet(hidden_dim=64).to(device)
    model = train_model(model, X_train, y_train, epochs=100, lr=1e-3, batch_size=32, weight_decay=1e-3)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_mse = ((y_pred_test - y_test) ** 2).mean().item()
        print(f'\nTest MSE: {test_mse:.4f}')

    # Compute Stein residuals on original test set
    print('\nComputing Stein residuals on original test set...')
    if use_alternative_stein:
        residuals, f_vals = compute_stein_residuals_alternative_simple(
            X_test, model, device, num_probes=4
        )
    else:
        residuals, f_vals = compute_stein_residuals_simple(
            X_test, model, device, num_probes=4
        )

    # Print statistics
    print('\n=== Stein Residual Statistics (Original Test Set) ===')
    print(f'Mean residual: {residuals.mean():.4e}')
    print(f'Std residual: {residuals.std():.4e}')
    print(f'Median residual: {np.median(residuals):.4e}')
    print(f'Min residual: {residuals.min():.4e}')
    print(f'Max residual: {residuals.max():.4e}')

    # Evaluate on shifted test data for multiple rotation angles
    print('\n' + '='*60)
    print('Evaluating on shifted test data for multiple rotation angles')
    print('='*60)

    epsilon = 10.0  # Single epsilon (radius) value
    angles = list(range(0, 360, 15))  # [0, 15, 30, 45, ..., 330, 345]
    print(f'\nEvaluating for angles: {angles} degrees (epsilon={epsilon})')

    all_results = []

    for idx, angle in enumerate(angles):
        print(f'\n--- Angle {idx+1}/{len(angles)}: {angle}° ---')
        print(f'Generating shifted test data with epsilon={epsilon}, angle={angle}°...')

        X_shifted, y_shifted = generate_shifted_data(n_samples=1000, epsilon=epsilon, angle_degrees=angle, seed=44+idx, sigma=0.0)
        X_shifted = X_shifted.to(device)
        y_shifted = y_shifted.to(device)

        print(f'Shifted data mean: {X_shifted.mean(dim=0).cpu().numpy()}')
        print(f'Shifted data std: {X_shifted.std(dim=0).cpu().numpy()}')

        # Evaluate shifted data
        results, residuals_shifted, log_likelihoods, f_vals_shifted = evaluate_shifted_data(
            model, X_shifted, y_shifted, device, epsilon=epsilon, angle_degrees=angle, num_probes=4,
            use_alternative_stein=use_alternative_stein
        )

        # Print evaluation results
        print(f'\nEvaluation Results for angle={angle}°:')
        print(f'  Test MSE: {results["test_mse"]:.4e}')
        print(f'  Log-likelihood mean: {results["log_likelihood"]["mean"]:.4e}')
        print(f'  Stein residual mean: {results["stein_residuals"]["mean"]:.4e}')
        print(f'  Stein residual median: {results["stein_residuals"]["median"]:.4e}')

        # Save results for this angle
        save_results(results, residuals_shifted, log_likelihoods, f_vals_shifted, X_shifted,
                    output_dir='results/small_changes_results', angle_degrees=angle)

        # Store for aggregated save
        all_results.append((epsilon, angle, results, residuals_shifted, log_likelihoods, f_vals_shifted, X_shifted))

    # Save aggregated results across all angles
    print('\n' + '='*60)
    print('Saving aggregated results...')
    print('='*60)
    save_all_results(all_results, output_dir='results/small_changes_results')

    # Print summary table
    print('\n=== Summary Across All Angles ===')
    print(f'{"Angle":<12} {"Test MSE":<15} {"Log-Lik Mean":<18} {"Stein Res Mean":<18} {"Stein Res Median":<18}')
    print('-' * 85)
    for epsilon, angle, results, _, _, _, _ in all_results:
        print(f'{angle:<12.0f}° {results["test_mse"]:<15.4e} {results["log_likelihood"]["mean"]:<18.4e} '
              f'{results["stein_residuals"]["mean"]:<18.4e} {results["stein_residuals"]["median"]:<18.4e}')

    # Generate plots
    print('\n' + '='*60)
    print('Generating visualization plots...')
    print('='*60)
    plot_results(all_results, output_dir='results/small_changes_results')

    return model, X_test, residuals, f_vals, all_results


if __name__ == '__main__':
    model, X_test, residuals, f_vals, all_results = main()

