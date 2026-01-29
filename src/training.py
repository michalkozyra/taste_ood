"""
Training functions for classifier and score models.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


def train_classifier(model, train_loader, val_loader, device, epochs=5, lr=1e-3, checkpoint_path='checkpoints/classifier.pth'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val = 1e9
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        # val
        model.eval()
        total = 0; correct = 0; vloss = 0.0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                vloss += criterion(logits,y).item()*x.size(0)
                preds = logits.argmax(1)
                total += x.size(0); correct += (preds==y).sum().item()
        val_loss = vloss/total; val_acc = correct/total
        print(f'Epoch {epoch+1} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    return model


def train_score_model(score_model, plain_dataset, device, epochs=5, batch_size=128, lr=1e-3, noise_sigma=0.2, ckpt='checkpoints/score.pth'):
    """
    Very small-scale denoising score matching training on MNIST images.
    This is *not* a full diffusion training schedule; it's a lightweight DSM to get a usable score for MNIST.
    For best results, train a proper multi-noise schedule (see Song & Ermon).
    """
    score_model.to(device)
    opt = torch.optim.Adam(score_model.parameters(), lr=lr)
    # Use num_workers=0 for small datasets to avoid overhead, and disable pin_memory for MPS
    pin_mem = (device.type == 'cuda')
    loader = DataLoader(plain_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    
    for epoch in range(epochs):
        score_model.train()
        pbar = tqdm(loader, desc=f'Score Epoch {epoch+1}/{epochs}')
        for x, y in pbar:
            x = x.to(device)
            # add i.i.d. gaussian noise
            noise = torch.randn_like(x, device=device) * noise_sigma
            x_noisy = x + noise
            # target score: grad_x log p_noisy(x_noisy) approximated by -noise / sigma^2 (for Gaussian corruption)
            target = - noise / (noise_sigma**2)
            pred = score_model(x_noisy)
            loss = ((pred - target)**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
    torch.save(score_model.state_dict(), ckpt)
    print('Saved score model to', ckpt)
    return score_model


def train_score_model_annealed(score_model: nn.Module, dataset: Dataset, device: torch.device,
                               epochs=50, batch_size=128, lr=2e-4,
                               n_levels=10, sigma_min=0.01, sigma_max=0.5, ckpt='checkpoints/score_unet.pth',
                               val_dataset: Optional[Dataset] = None, eval_every: int = 5):
    """
    Train score_model using annealed (multi-sigma) denoising score matching (song & ermon style).
    For each batch:
      - sample a sigma for each example from a geometric schedule
      - x_noisy = x + sigma * z
      - target score = - (x_noisy - x) / sigma^2
      - minimize MSE between predicted score(x_noisy, sigma) and target
    
    Args:
        val_dataset: Optional validation dataset for evaluation
        eval_every: Evaluate on validation set every N epochs
    """
    score_model.to(device)
    opt = torch.optim.Adam(score_model.parameters(), lr=lr)
    # Use cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    # Use num_workers=0 for MPS (Apple Silicon) and when testing to avoid multiprocessing issues
    # For CUDA, use 2 workers; for others, use 0
    pin_mem = (device.type == 'cuda')
    num_workers = 2 if device.type == 'cuda' else 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem) if val_dataset else None
    # geometric schedule of noise levels
    sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), steps=n_levels)).to(device)

    train_losses = []
    val_losses = []

    for ep in range(epochs):
        score_model.train()
        losses = []
        pbar = tqdm(loader, desc=f'Score epoch {ep+1}/{epochs}')
        for x, _ in pbar:
            x = x.to(device)
            B = x.shape[0]
            # sample a sigma index per sample
            idx = torch.randint(0, len(sigmas), (B,), device=device)
            sigma_batch = sigmas[idx]  # shape (B,)
            noise = torch.randn_like(x) * sigma_batch.view(B,1,1,1)
            x_noisy = x + noise
            # target: score of noisy image = grad_x log p_sigma(x_noisy) = - (x_noisy - x) / sigma^2
            target = - noise / (sigma_batch.view(B,1,1,1)**2)
            pred = score_model(x_noisy, sigma_batch)
            
            # Weighted loss: weight by sigma^2 to normalize across noise levels
            # Without weighting, small sigma (large target values) dominates the loss.
            # Weighting by σ² balances learning across all noise levels.
            # Alternative: weight by 1/σ² to emphasize small sigma (fine details).
            # Handle shape mismatch: pred is (B, 1, H, W), target is (B, C, H, W)
            if target.shape[1] != pred.shape[1]:
                # Average target across channels to match pred shape
                target_for_loss = target.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            else:
                target_for_loss = target
            
            # Compute per-sample loss, then weight by sigma^2 to normalize
            per_sample_loss = ((pred - target_for_loss)**2).mean(dim=(1, 2, 3))  # (B,)
            weights = (sigma_batch ** 2).to(device)  # (B,) - weight by sigma^2 to balance scales
            loss = (per_sample_loss * weights).mean()
            
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))
        
        epoch_train_loss = np.mean(losses)
        train_losses.append(epoch_train_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Validation evaluation
        if val_loader is not None and (ep + 1) % eval_every == 0:
            val_loss = _evaluate_score_model(score_model, val_loader, sigmas, device)
            val_losses.append((ep + 1, val_loss))
            print(f'  Epoch {ep+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    # Final evaluation and diagnostics
    print('\n' + '='*60)
    print('Score Model Training Summary')
    print('='*60)
    print(f'Final training loss: {train_losses[-1]:.4f}')
    if train_losses[0] > 0:
        improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f'Loss improvement: {improvement:.1f}% (from {train_losses[0]:.4f} to {train_losses[-1]:.4f})')
    
    # Run comprehensive diagnostics
    _diagnose_score_model(score_model, loader, sigmas, device)
    
    torch.save(score_model.state_dict(), ckpt)
    print(f'\nSaved UNet score model to {ckpt}')
    return score_model, sigmas


@torch.no_grad()
def _evaluate_score_model(score_model: nn.Module, val_loader: DataLoader, sigmas: torch.Tensor, device: torch.device) -> float:
    """Evaluate score model on validation set."""
    score_model.eval()
    losses = []
    for x, _ in val_loader:
        x = x.to(device)
        B = x.shape[0]
        idx = torch.randint(0, len(sigmas), (B,), device=device)
        sigma_batch = sigmas[idx]
        noise = torch.randn_like(x) * sigma_batch.view(B,1,1,1)
        x_noisy = x + noise
        target = - noise / (sigma_batch.view(B,1,1,1)**2)
        pred = score_model(x_noisy, sigma_batch)
        
        # Weighted loss: weight by sigma^2 to normalize across noise levels
        # Handle shape mismatch: pred is (B, 1, H, W), target is (B, C, H, W)
        if target.shape[1] != pred.shape[1]:
            target_for_loss = target.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            target_for_loss = target
        
        # Compute per-sample loss, then weight by sigma^2 to normalize
        per_sample_loss = ((pred - target_for_loss)**2).mean(dim=(1, 2, 3))  # (B,)
        weights = (sigma_batch ** 2).to(device)  # (B,) - weight by sigma^2 to balance scales
        loss = (per_sample_loss * weights).mean()
        
        losses.append(loss.item())
    return np.mean(losses)


@torch.no_grad()
def _diagnose_score_model(score_model: nn.Module, loader: DataLoader, sigmas: torch.Tensor, device: torch.device, num_batches: int = 2):
    """Run diagnostic checks on trained score model."""
    score_model.eval()
    
    print('\nRunning diagnostics...')
    all_preds_flat = []
    all_targets_flat = []
    all_sigmas_used = []
    per_sigma_losses = {sigma.item(): [] for sigma in sigmas}
    
    batch_count = 0
    for x, _ in loader:
        if batch_count >= num_batches:
            break
        x = x.to(device)
        B = x.shape[0]
        idx = torch.randint(0, len(sigmas), (B,), device=device)
        sigma_batch = sigmas[idx]
        noise = torch.randn_like(x) * sigma_batch.view(B,1,1,1)
        x_noisy = x + noise
        target = - noise / (sigma_batch.view(B,1,1,1)**2)
        pred = score_model(x_noisy, sigma_batch)
        
        # Handle shape mismatch: pred is (B, 1, H, W), target is (B, C, H, W) where C=in_channels
        # For diagnostics, we'll compute per-channel statistics or average across channels
        # For overall loss, we need to match shapes - take mean across channels for target
        if target.shape[1] != pred.shape[1]:
            # Target has multiple channels (RGB), pred has 1 channel
            # Average target across channels to match pred shape
            target_for_stats = target.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            target_for_stats = target
        
        # Flatten for statistics
        pred_flat = pred.cpu().flatten().numpy()
        target_flat = target_for_stats.cpu().flatten().numpy()
        
        # Store flattened arrays
        all_preds_flat.append(pred_flat)
        all_targets_flat.append(target_flat)
        
        # Store sigma values (one per sample, but we'll use them for per-sigma analysis)
        for i in range(B):
            all_sigmas_used.append(sigma_batch[i].cpu().item())
            # Per-sigma losses - handle shape mismatch
            sigma_val = sigma_batch[i].item()
            if target.shape[1] != pred.shape[1]:
                # Average target across channels for loss computation
                target_i = target[i].mean(dim=0, keepdim=True)  # (1, H, W)
            else:
                target_i = target[i]
            loss_val = ((pred[i] - target_i)**2).mean().item()
            per_sigma_losses[sigma_val].append(loss_val)
        
        batch_count += 1
    
    # Concatenate all flattened predictions and targets
    all_preds_flat = np.concatenate(all_preds_flat)
    all_targets_flat = np.concatenate(all_targets_flat)
    all_sigmas_used = np.array(all_sigmas_used)
    
    # Diagnostic 1: Prediction vs Target statistics
    print(f'\n1. Prediction Statistics:')
    print(f'   Mean: {all_preds_flat.mean():.4f}, Std: {all_preds_flat.std():.4f}')
    print(f'   Min: {all_preds_flat.min():.4f}, Max: {all_preds_flat.max():.4f}')
    print(f'\n2. Target Statistics:')
    print(f'   Mean: {all_targets_flat.mean():.4f}, Std: {all_targets_flat.std():.4f}')
    print(f'   Min: {all_targets_flat.min():.4f}, Max: {all_targets_flat.max():.4f}')
    
    # Diagnostic 2: Correlation (sample a subset to avoid memory issues)
    sample_size = min(10000, len(all_preds_flat))
    indices = np.random.choice(len(all_preds_flat), sample_size, replace=False)
    correlation = np.corrcoef(all_preds_flat[indices], all_targets_flat[indices])[0, 1]
    print(f'\n3. Prediction-Target Correlation: {correlation:.4f} (sampled {sample_size} values)')
    if correlation < 0.5:
        print('   ⚠️  WARNING: Low correlation suggests model may not be learning correctly')
    elif correlation > 0.8:
        print('   ✓ Good correlation - model is learning the score function')
    
    # Diagnostic 3: Per-sigma loss breakdown
    print(f'\n4. Per-Sigma Loss Breakdown:')
    for sigma_val in sorted(per_sigma_losses.keys()):
        losses = per_sigma_losses[sigma_val]
        if losses:
            mean_loss = np.mean(losses)
            print(f'   σ={sigma_val:.4f}: Mean Loss = {mean_loss:.4f} (n={len(losses)})')
    
    # Diagnostic 4: Scale check
    pred_scale = np.abs(all_preds_flat).mean()
    target_scale = np.abs(all_targets_flat).mean()
    scale_ratio = pred_scale / target_scale if target_scale > 0 else float('inf')
    print(f'\n5. Scale Check:')
    print(f'   Pred scale: {pred_scale:.4f}, Target scale: {target_scale:.4f}')
    print(f'   Ratio: {scale_ratio:.4f} (should be close to 1.0)')
    if scale_ratio < 0.1 or scale_ratio > 10:
        print('   ⚠️  WARNING: Scale mismatch suggests output normalization issue')
    else:
        print('   ✓ Scale ratio is reasonable')
    
    # Diagnostic 5: Overall loss
    overall_loss = np.mean((all_preds_flat - all_targets_flat)**2)
    print(f'\n6. Overall Diagnostic Loss: {overall_loss:.4f}')
    if overall_loss > 100:
        print('   ⚠️  WARNING: Very high loss - model may need more training or different hyperparameters')
    elif overall_loss < 1.0:
        print('   ✓ Loss is reasonable')
    else:
        print('   ⚠️  Loss is moderately high - consider more training or hyperparameter tuning')
    
    print('='*60)

