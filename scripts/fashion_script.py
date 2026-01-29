"""
fashion_script.py

Pipeline for evaluating Stein shift detection on mixed MNIST/Fashion-MNIST test sets:
 - Train a CNN classifier on MNIST (32x32 resized)
 - Train a score model on MNIST (same procedure as script.py)
 - Generate mixed test sets parameterized by alpha (fraction of MNIST)
 - For each test set, evaluate loss, accuracy, and Stein metrics (corrected by training baseline)
"""

import os
import argparse
import math
import json
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from src.models import ClassifierNet32x32, SmallScoreNet32x32, UNetScore
from src.utils import get_device
from src.data import get_mnist_dataloaders_32x32, get_fashion_mnist_dataloaders_32x32, create_mixed_test_set
from src.eval_functions import compute_stein_residuals, compute_stein_residuals_alternative
from src.training import train_classifier, train_score_model, train_score_model_annealed

# ---------------------------
# Evaluation driver & plotting
# ---------------------------
def evaluate_pipeline(args):
    device = get_device()
    print(f'Using device: {device}')
    
    # Load MNIST and Fashion-MNIST datasets (32x32)
    print('\nLoading MNIST and Fashion-MNIST datasets (32x32)...')
    mnist_train_loader, mnist_test_loader, mnist_train_ds, mnist_test_set = get_mnist_dataloaders_32x32(
        batch_size=args.batch_size, data_dir=args.data_dir, device=device)
    fashion_train_loader, fashion_test_loader, fashion_train_ds, fashion_test_set = get_fashion_mnist_dataloaders_32x32(
        batch_size=args.batch_size, data_dir=args.data_dir, device=device)
    
    print(f'MNIST: {len(mnist_train_ds)} train, {len(mnist_test_set)} test')
    print(f'Fashion-MNIST: {len(fashion_train_ds)} train, {len(fashion_test_set)} test')

    # Classifier: train on MNIST or load from cache
    clf = ClassifierNet32x32()
    classifier_ckpt_path = args.classifier_ckpt or 'checkpoints/classifier_32x32.pth'
    
    if args.cached_classifier:
        # Load from cache - raise error if checkpoint doesn't exist
        if not os.path.exists(classifier_ckpt_path):
            raise FileNotFoundError(f'Classifier checkpoint not found: {classifier_ckpt_path}. Use --cached-classifier only when checkpoint exists.')
        clf.load_state_dict(torch.load(classifier_ckpt_path, map_location=device))
        clf.to(device)
        print('Loaded classifier from', classifier_ckpt_path)
    else:
        # Train and save (overwrite if exists)
        # For validation, we'll use a subset of training data
        n_train = len(mnist_train_ds)
        val_size = int(0.1 * n_train)
        train_indices = list(range(val_size, n_train))
        val_indices = list(range(val_size))
        train_subset = Subset(mnist_train_ds, train_indices)
        val_subset = Subset(mnist_train_ds, val_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        print('Training classifier on MNIST (this will take some time).')
        clf = train_classifier(clf, train_loader, val_loader, device, epochs=args.classifier_epochs, checkpoint_path=classifier_ckpt_path)
        print('Classifier trained and saved to', classifier_ckpt_path)

    # Verify image sizes
    sample_mnist, _ = next(iter(mnist_test_loader))
    print(f'\nImage size verification:')
    print(f'  MNIST test set: {sample_mnist.shape[2]}x{sample_mnist.shape[3]} (expected: 32x32)')
    assert sample_mnist.shape[2] == 32 and sample_mnist.shape[3] == 32, 'Images should be 32x32'
    
    # Score model: either load from cache or train on MNIST
    score_model = None
    sigmas = None
    use_unet = (args.score_model_type == 'unet')
    
    # Set default checkpoint path based on model type
    if args.score_ckpt is None:
        score_ckpt_path = 'checkpoints/score_unet_32x32.pth' if use_unet else 'checkpoints/score_32x32.pth'
    else:
        score_ckpt_path = args.score_ckpt
    
    if args.cached_score:
        # Load from cache - raise error if checkpoint doesn't exist
        if not os.path.exists(score_ckpt_path):
            raise FileNotFoundError(f'Score model checkpoint not found: {score_ckpt_path}. Use --cached-score only when checkpoint exists.')
        
        if use_unet:
            score_model = UNetScore(base_ch=64)
            score_model.load_state_dict(torch.load(score_ckpt_path, map_location=device))
            score_model.to(device)
            # Reconstruct sigmas schedule (same as training)
            n_levels = 10
            sigma_min = 0.01
            sigma_max = 0.5
            sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), steps=n_levels)).to(device)
        else:
            score_model = SmallScoreNet32x32()
            score_model.load_state_dict(torch.load(score_ckpt_path, map_location=device))
            score_model.to(device)
            sigmas = None  # SmallScoreNet does not use sigmas
        
        print(f'Loaded {args.score_model_type} score model from {score_ckpt_path}')
    else:
        # Train and save (overwrite if exists)
        if use_unet:
            print('Training UNet score model on MNIST (annealed DSM)...')
            score_model = UNetScore(base_ch=64)
            score_model, sigmas = train_score_model_annealed(score_model, mnist_train_ds, device, epochs=args.score_epochs, 
                                                              batch_size=args.batch_size, lr=2e-4, ckpt=score_ckpt_path)
        else:
            print('Training Small score model on MNIST (simple DSM)...')
            score_model = SmallScoreNet32x32()
            score_model = train_score_model(score_model, mnist_train_ds, device, epochs=args.score_epochs,
                                            batch_size=args.batch_size, ckpt=score_ckpt_path)
            sigmas = None  # SmallScoreNet does not use sigmas
        
        print(f'Score model trained and saved to {score_ckpt_path}')

    # Helper function to compute loss and accuracy
    def compute_loss_and_accuracy(model, dataloader, device):
        """Compute cross-entropy loss and accuracy for a dataloader."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    x, y, _ = batch  # (x, y, datatype)
                else:
                    x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    # Generate mixed test sets with different alpha values
    print('\nGenerating mixed test sets...')
    alphas = np.linspace(0.8, 1.0, 11)  # [0.0, 0.111..., 0.222..., ..., 1.0]
    mixed_test_sets = {}
    mixed_test_loaders = {}
    
    for alpha in alphas:
        print(f'  Creating mixed test set with alpha={alpha:.3f} ({alpha*100:.1f}% MNIST, {(1-alpha)*100:.1f}% Fashion-MNIST)')
        mixed_ds = create_mixed_test_set(mnist_test_set, fashion_test_set, alpha)
        mixed_test_sets[alpha] = mixed_ds
        mixed_test_loaders[alpha] = DataLoader(mixed_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f'    Mixed set size: {len(mixed_ds)} samples')
    
    # Compute loss and accuracy for all mixed test sets
    print('\nComputing loss and accuracy for all mixed test sets...')
    mixed_losses = {}
    mixed_accuracies = {}
    for alpha in alphas:
        loss, acc = compute_loss_and_accuracy(clf, mixed_test_loaders[alpha], device)
        mixed_losses[alpha] = loss
        mixed_accuracies[alpha] = acc
        print(f'Alpha {alpha:.3f} - Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    
    # Compute Stein residuals for all mixed test sets
    # Also track which samples come from MNIST vs Fashion-MNIST
    print('\nComputing Stein residuals for all mixed test sets...')
    r_mixed_dict = {}
    f_mixed_dict = {}
    y_mixed_dict = {}
    datatype_dict = {}  # Track datatype: 0 = MNIST, 1 = Fashion-MNIST
    
    for alpha in alphas:
        print(f'Computing Stein residuals for alpha={alpha:.3f}...')
        
        # First, extract datatype information before computing Stein residuals
        # We need to iterate through the dataset to get datatypes
        datatypes = []
        for i in range(len(mixed_test_sets[alpha])):
            _, _, dt = mixed_test_sets[alpha][i]
            datatypes.append(dt)
        datatype_dict[alpha] = np.array(datatypes)
        
        # Create a wrapper dataloader that only returns (x, y) for compute_stein_residuals
        class XYOnlyDataset(Dataset):
            def __init__(self, mixed_ds):
                self.mixed_ds = mixed_ds
            def __len__(self):
                return len(self.mixed_ds)
            def __getitem__(self, idx):
                x, y, _ = self.mixed_ds[idx]
                return x, y
        
        xy_only_ds = XYOnlyDataset(mixed_test_sets[alpha])
        xy_only_loader = DataLoader(xy_only_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Choose Stein operator based on argument
        if args.stein_operator == 'alternative':
            r_mixed_dict[alpha], f_mixed_dict[alpha], y_mixed_dict[alpha] = compute_stein_residuals_alternative(
                xy_only_loader, clf, score_model, device, sigmas=sigmas)
        else:
            r_mixed_dict[alpha], f_mixed_dict[alpha], y_mixed_dict[alpha] = compute_stein_residuals(
                xy_only_loader, clf, score_model, device, num_probes=args.hutchinson_probes, sigmas=sigmas)
    
    # Compute Stein residuals on a subset of training data for baseline
    print('\nComputing Stein residuals on training data subset (5K samples) for baseline...')
    train_size = len(mnist_train_ds)
    train_subset_size = min(5000, train_size)
    train_subset_indices = random.sample(range(train_size), train_subset_size)
    train_subset = Subset(mnist_train_ds, train_subset_indices)
    train_subset_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Choose Stein operator based on argument (same as for test sets)
    if args.stein_operator == 'alternative':
        r_train, _, _ = compute_stein_residuals_alternative(
            train_subset_loader, clf, score_model, device, sigmas=sigmas)
    else:
        r_train, _, _ = compute_stein_residuals(
            train_subset_loader, clf, score_model, device, num_probes=args.hutchinson_probes, sigmas=sigmas)
    
    mean_stein_train = r_train.mean()
    print(f'Training data baseline - Mean Stein Score: {mean_stein_train:.4e}')
    
    # Prepare null distribution for hypothesis testing
    # Demean the training Stein scores
    null_distribution = r_train - mean_stein_train
    # For one-sided test: null distribution of absolute values
    null_distribution_abs = np.abs(null_distribution)
    
    # Compute critical values for hypothesis testing (α = 0.05)
    alpha_significance = 0.05
    null_2_5th = np.percentile(null_distribution, 2.5)
    null_97_5th = np.percentile(null_distribution, 97.5)
    null_95th_abs = np.percentile(null_distribution_abs, 95)
    
    print(f'\nHypothesis testing setup:')
    print(f'  Null distribution size: {len(null_distribution)}')
    print(f'  Two-sided critical values: [{null_2_5th:.4e}, {null_97_5th:.4e}]')
    print(f'  One-sided (abs) critical value: {null_95th_abs:.4e}')

    # Basic diagnostics
    def summarize(name, r, fvals, labels):
        print(f'\n---- {name} ----')
        print('count:', len(r))
        print('r mean {:.4e}, std {:.4e}, median {:.4e}'.format(r.mean(), r.std(), np.median(np.abs(r))))
        # correlation between |r| and model confidence (fvals)
        if fvals.ndim == 1:
            confidence = fvals
        else:
            confidence = fvals.max(axis=1)
        print('corr(|r|, 1 - confidence):', np.corrcoef(np.abs(r), (1.0 - confidence))[0,1])
    
    for alpha in alphas:
        summarize(f'Mixed test set alpha={alpha:.3f}', r_mixed_dict[alpha], f_mixed_dict[alpha], y_mixed_dict[alpha])

    # Visualization
    output_dir = 'results/fashion_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot: Accuracy, Loss, and Stein Score vs Alpha
    z_score = 1.96
    alphas_sorted = sorted(alphas)
    mixed_accs = [mixed_accuracies[a] for a in alphas_sorted]
    mixed_losses_list = [mixed_losses[a] for a in alphas_sorted]
    mixed_stein_mean = [r_mixed_dict[a].mean() for a in alphas_sorted]
    mixed_stein_std = [r_mixed_dict[a].std() for a in alphas_sorted]
    mixed_stein_sem = [std / np.sqrt(len(r_mixed_dict[a])) for std, a in zip(mixed_stein_std, alphas_sorted)]
    mixed_stein_mean_relative = [m - mean_stein_train for m in mixed_stein_mean]
    
    # Accuracy confidence intervals
    mixed_acc_cis = []
    for alpha in alphas_sorted:
        n = len(r_mixed_dict[alpha])
        p = mixed_accuracies[alpha]
        se = np.sqrt(p * (1 - p) / n)
        mixed_acc_cis.append(z_score * se)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy vs Alpha
    axes[0].plot(alphas_sorted, mixed_accs, marker='o', linewidth=2, markersize=8, color='green', label='Mean')
    axes[0].fill_between(alphas_sorted,
                         [acc - ci for acc, ci in zip(mixed_accs, mixed_acc_cis)],
                         [acc + ci for acc, ci in zip(mixed_accs, mixed_acc_cis)],
                         alpha=0.3, color='green', label='95% CI')
    axes[0].set_xlabel('Alpha (MNIST fraction)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Alpha')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Mean Stein Score (relative to training) vs Alpha
    axes[1].plot(alphas_sorted, mixed_stein_mean_relative, marker='^', linewidth=2, markersize=8, color='blue', label='Mean (relative to training)')
    axes[1].fill_between(alphas_sorted,
                         [m - z_score * sem for m, sem in zip(mixed_stein_mean_relative, mixed_stein_sem)],
                         [m + z_score * sem for m, sem in zip(mixed_stein_mean_relative, mixed_stein_sem)],
                         alpha=0.3, color='blue', label='95% CI')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Training baseline')
    axes[1].set_xlabel('Alpha (MNIST fraction)')
    axes[1].set_ylabel('Mean Stein Score (relative to training)')
    axes[1].set_title('Stein Score vs Alpha')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_vs_alpha.png', dpi=150)
    print(f'Saved metrics plot to {output_dir}/metrics_vs_alpha.png')
    plt.close()
    
    # Scatter plot: Accuracy vs Stein Score
    plt.figure(figsize=(10, 6))
    plt.scatter(mixed_stein_mean_relative, mixed_accs, s=150, alpha=0.7, c=alphas_sorted, cmap='viridis', edgecolors='black', linewidth=1.5)
    plt.xlabel('Mean Stein Score (relative to training)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Stein Score (colored by Alpha)')
    plt.colorbar(label='Alpha (MNIST fraction)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_stein.png', dpi=150)
    print(f'Saved scatter plot to {output_dir}/accuracy_vs_stein.png')
    plt.close()
    
    # Stein score histograms per test set (with training mean removed)
    # Generate separate plots for each alpha with MNIST and Fashion-MNIST in side-by-side subplots
    print('\nCreating Stein score histograms per test set...')
    
    for alpha in alphas_sorted:
        r_relative = r_mixed_dict[alpha] - mean_stein_train
        datatype = datatype_dict[alpha]
        
        # Separate by datatype
        mnist_mask = (datatype == 0)
        fashion_mask = (datatype == 1)
        
        r_mnist = r_relative[mnist_mask]
        r_fashion = r_relative[fashion_mask]
        
        # Determine shared x-axis limits for fair comparison
        all_values = np.concatenate([r_mnist, r_fashion])
        x_min = np.percentile(all_values, 1)  # Use 1st and 99th percentile to avoid outliers
        x_max = np.percentile(all_values, 99)
        x_range = x_max - x_min
        x_min = x_min - 0.05 * x_range
        x_max = x_max + 0.05 * x_range
        
        # Create figure with side-by-side subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Common bins for both histograms
        bins = np.linspace(x_min, x_max, 50)
        
        # Plot 1: MNIST histogram
        axes[0].hist(r_mnist, bins=bins, alpha=0.7, density=True, edgecolor='black', linewidth=0.5, 
                color='blue', label=f'MNIST (n={len(r_mnist)})')
        axes[0].set_xlabel('Stein Score (relative to training)')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'MNIST - Alpha={alpha:.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(x_min, x_max)
        
        # Plot 2: Fashion-MNIST histogram
        axes[1].hist(r_fashion, bins=bins, alpha=0.7, density=True, edgecolor='black', linewidth=0.5, 
                color='orange', label=f'Fashion-MNIST (n={len(r_fashion)})')
        axes[1].set_xlabel('Stein Score (relative to training)')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'Fashion-MNIST - Alpha={alpha:.3f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(x_min, x_max)
        
        # Add overall title
        fig.suptitle(f'Stein Score Distributions - Alpha={alpha:.3f} (Total n={len(r_mixed_dict[alpha])})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = f'{output_dir}/stein_histogram_alpha_{alpha:.3f}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f'  Saved histogram for alpha={alpha:.3f} to {filename}')
    
    print(f'Saved {len(alphas_sorted)} individual histogram files to {output_dir}/')
    
    # Mean Stein score conditioned on datatype (MNIST vs Fashion-MNIST)
    print('\nComputing mean Stein scores by datatype...')
    stein_mean_mnist = []
    stein_mean_fashion = []
    stein_std_mnist = []
    stein_std_fashion = []
    stein_abs_mean_mnist = []
    stein_abs_mean_fashion = []
    correlations = []
    
    for alpha in alphas_sorted:
        r_relative = r_mixed_dict[alpha] - mean_stein_train
        r_abs = np.abs(r_relative)
        datatype = datatype_dict[alpha]
        
        # Separate by datatype
        mnist_mask = (datatype == 0)
        fashion_mask = (datatype == 1)
        
        r_mnist = r_relative[mnist_mask]
        r_fashion = r_relative[fashion_mask]
        r_abs_mnist = r_abs[mnist_mask]
        r_abs_fashion = r_abs[fashion_mask]
        
        stein_mean_mnist.append(r_mnist.mean() if len(r_mnist) > 0 else np.nan)
        stein_mean_fashion.append(r_fashion.mean() if len(r_fashion) > 0 else np.nan)
        stein_std_mnist.append(r_mnist.std() if len(r_mnist) > 0 else np.nan)
        stein_std_fashion.append(r_fashion.std() if len(r_fashion) > 0 else np.nan)
        stein_abs_mean_mnist.append(r_abs_mnist.mean() if len(r_abs_mnist) > 0 else np.nan)
        stein_abs_mean_fashion.append(r_abs_fashion.mean() if len(r_abs_fashion) > 0 else np.nan)
        
        # Correlation between Stein score and datatype
        if len(r_relative) > 1:
            corr = np.corrcoef(r_relative, datatype)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    
    # Plot mean Stein scores by datatype
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean Stein Score by datatype vs Alpha
    axes[0, 0].plot(alphas_sorted, stein_mean_mnist, marker='o', linewidth=2, markersize=8, 
                 color='blue', label='MNIST', linestyle='-')
    axes[0, 0].plot(alphas_sorted, stein_mean_fashion, marker='s', linewidth=2, markersize=8, 
                 color='orange', label='Fashion-MNIST', linestyle='-')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Training baseline')
    axes[0, 0].set_xlabel('Alpha (MNIST fraction)')
    axes[0, 0].set_ylabel('Mean Stein Score (relative to training)')
    axes[0, 0].set_title('Mean Stein Score by Datatype vs Alpha')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean Absolute Stein Score by datatype vs Alpha
    axes[0, 1].plot(alphas_sorted, stein_abs_mean_mnist, marker='o', linewidth=2, markersize=8, 
                 color='blue', label='MNIST', linestyle='-')
    axes[0, 1].plot(alphas_sorted, stein_abs_mean_fashion, marker='s', linewidth=2, markersize=8, 
                 color='orange', label='Fashion-MNIST', linestyle='-')
    axes[0, 1].set_xlabel('Alpha (MNIST fraction)')
    axes[0, 1].set_ylabel('Mean |Stein Score| (relative to training)')
    axes[0, 1].set_title('Mean Absolute Stein Score by Datatype vs Alpha')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation between Stein Score and Datatype vs Alpha
    axes[1, 0].plot(alphas_sorted, correlations, marker='^', linewidth=2, markersize=8, 
                 color='purple', label='Correlation')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Alpha (MNIST fraction)')
    axes[1, 0].set_ylabel('Correlation (Stein Score, Datatype)')
    axes[1, 0].set_title('Stein-Datatype Correlation vs Alpha')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation between Absolute Stein Score and Datatype vs Alpha
    correlations_abs = []
    for alpha in alphas_sorted:
        r_relative = r_mixed_dict[alpha] - mean_stein_train
        r_abs = np.abs(r_relative)
        datatype = datatype_dict[alpha]
        if len(r_abs) > 1:
            corr_abs = np.corrcoef(r_abs, datatype)[0, 1]
            correlations_abs.append(corr_abs)
        else:
            correlations_abs.append(np.nan)
    
    axes[1, 1].plot(alphas_sorted, correlations_abs, marker='^', linewidth=2, markersize=8, 
                 color='purple', label='Correlation (|Stein|, Datatype)')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Alpha (MNIST fraction)')
    axes[1, 1].set_ylabel('Correlation (|Stein Score|, Datatype)')
    axes[1, 1].set_title('Absolute Stein-Datatype Correlation vs Alpha')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stein_by_datatype.png', dpi=150)
    print(f'Saved datatype analysis plot to {output_dir}/stein_by_datatype.png')
    plt.close()
    
    # Print correlation statistics
    print('\n--- Stein Score and Datatype Correlation ---')
    for alpha, corr in zip(alphas_sorted, correlations):
        print(f'Alpha {alpha:.3f}: Correlation = {corr:.4f}')
    
    # Hypothesis testing for each test set
    print('\n--- Hypothesis Testing ---')
    rejection_two_sided_dict = {}  # Per-point rejection flags
    rejection_one_sided_dict = {}  # Per-point rejection flags
    rejection_rate_two_sided = {}
    rejection_rate_one_sided = {}
    rejection_rate_two_sided_mnist = {}
    rejection_rate_one_sided_mnist = {}
    rejection_rate_two_sided_fashion = {}
    rejection_rate_one_sided_fashion = {}
    
    for alpha in alphas_sorted:
        r_relative = r_mixed_dict[alpha] - mean_stein_train
        datatype = datatype_dict[alpha]
        
        # Separate by datatype
        mnist_mask = (datatype == 0)
        fashion_mask = (datatype == 1)
        
        # Two-sided test: reject if outside [2.5th, 97.5th] percentile
        rejected_two_sided = (r_relative < null_2_5th) | (r_relative > null_97_5th)
        rejection_two_sided_dict[alpha] = rejected_two_sided
        rejection_rate_two_sided[alpha] = rejected_two_sided.mean()
        
        # One-sided test: reject if |test| > 95th percentile of |null|
        r_abs = np.abs(r_relative)
        rejected_one_sided = r_abs > null_95th_abs
        rejection_one_sided_dict[alpha] = rejected_one_sided
        rejection_rate_one_sided[alpha] = rejected_one_sided.mean()
        
        # Separate statistics for MNIST and Fashion-MNIST
        rejection_rate_two_sided_mnist[alpha] = rejected_two_sided[mnist_mask].mean() if mnist_mask.sum() > 0 else np.nan
        rejection_rate_one_sided_mnist[alpha] = rejected_one_sided[mnist_mask].mean() if mnist_mask.sum() > 0 else np.nan
        rejection_rate_two_sided_fashion[alpha] = rejected_two_sided[fashion_mask].mean() if fashion_mask.sum() > 0 else np.nan
        rejection_rate_one_sided_fashion[alpha] = rejected_one_sided[fashion_mask].mean() if fashion_mask.sum() > 0 else np.nan
        
        print(f'Alpha {alpha:.3f}:')
        print(f'  Two-sided rejection rate: {rejection_rate_two_sided[alpha]:.4f} (MNIST: {rejection_rate_two_sided_mnist[alpha]:.4f}, Fashion: {rejection_rate_two_sided_fashion[alpha]:.4f})')
        print(f'  One-sided rejection rate: {rejection_rate_one_sided[alpha]:.4f} (MNIST: {rejection_rate_one_sided_mnist[alpha]:.4f}, Fashion: {rejection_rate_one_sided_fashion[alpha]:.4f})')
    
    # Plot rejection rates (only bottom two subplots: MNIST vs Fashion-MNIST comparisons)
    print('\nCreating hypothesis testing plots...')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Two-sided rejection rate comparison (MNIST vs Fashion-MNIST)
    axes[0].plot(alphas_sorted, [rejection_rate_two_sided_mnist[a] for a in alphas_sorted], 
                    marker='s', linewidth=2, markersize=8, color='green', label='MNIST', linestyle='-')
    axes[0].plot(alphas_sorted, [rejection_rate_two_sided_fashion[a] for a in alphas_sorted], 
                    marker='^', linewidth=2, markersize=8, color='orange', label='Fashion-MNIST', linestyle='-')
    axes[0].axhline(y=alpha_significance, color='r', linestyle=':', alpha=0.5, label=f'Significance level ({alpha_significance})')
    axes[0].set_xlabel('Alpha (MNIST fraction)')
    axes[0].set_ylabel('Rejection Rate')
    axes[0].set_title('Two-Sided Test: MNIST vs Fashion-MNIST')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Plot 2: One-sided rejection rate comparison (MNIST vs Fashion-MNIST)
    axes[1].plot(alphas_sorted, [rejection_rate_one_sided_mnist[a] for a in alphas_sorted], 
                    marker='s', linewidth=2, markersize=8, color='green', label='MNIST', linestyle='-')
    axes[1].plot(alphas_sorted, [rejection_rate_one_sided_fashion[a] for a in alphas_sorted], 
                    marker='^', linewidth=2, markersize=8, color='orange', label='Fashion-MNIST', linestyle='-')
    axes[1].axhline(y=alpha_significance, color='r', linestyle=':', alpha=0.5, label=f'Significance level ({alpha_significance})')
    axes[1].set_xlabel('Alpha (MNIST fraction)')
    axes[1].set_ylabel('Rejection Rate')
    axes[1].set_title('One-Sided Test (|Stein|): MNIST vs Fashion-MNIST')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hypothesis_testing.png', dpi=150)
    print(f'Saved hypothesis testing plot to {output_dir}/hypothesis_testing.png')
    plt.close()
    
    # Save top-k and bottom-k Fashion-MNIST images based on |stein_score|
    print('\nSaving top-k and bottom-k Fashion-MNIST images...')
    top_images_dir = f'{output_dir}/top_images'
    os.makedirs(top_images_dir, exist_ok=True)
    
    def save_topk_fashion_images(mixed_ds, r_relative, datatype, k=10, prefix='top', alpha=None):
        """
        Save top-k or bottom-k Fashion-MNIST images based on |stein_score|.
        
        Args:
            mixed_ds: Mixed dataset (returns (x, y, datatype))
            r_relative: Stein scores relative to training (demeaned)
            datatype: Array indicating datatype (0=MNIST, 1=Fashion-MNIST)
            k: Number of images to save
            prefix: 'top' for highest |stein|, 'bottom' for lowest |stein|
            alpha: Alpha value for filename
        """
        # Filter to only Fashion-MNIST samples
        fashion_mask = (datatype == 1)
        if fashion_mask.sum() == 0:
            print(f'  No Fashion-MNIST samples for alpha={alpha:.3f}, skipping...')
            return
        
        r_fashion = r_relative[fashion_mask]
        r_abs_fashion = np.abs(r_fashion)
        
        # Get indices of Fashion-MNIST samples in the original dataset
        fashion_indices = np.where(fashion_mask)[0]
        
        # Sort by absolute Stein score
        if prefix == 'top':
            # Top-k: highest |stein_score|
            sorted_idx = np.argsort(-r_abs_fashion)[:k]
        else:
            # Bottom-k: lowest |stein_score|
            sorted_idx = np.argsort(r_abs_fashion)[:k]
        
        # Save images
        for i, idx in enumerate(sorted_idx):
            original_idx = fashion_indices[idx]
            x, y, dt = mixed_ds[original_idx]
            
            # x is tensor (1, H, W) - should be (1, 32, 32) after resize
            assert x.shape[1] == 32 and x.shape[2] == 32, f'Expected 32x32 image, got {x.shape[1]}x{x.shape[2]}'
            img_data = x.squeeze().numpy()  # (H, W) shape, should be (32, 32)
            H, W = img_data.shape
            assert H == 32 and W == 32, f'Expected 32x32 after squeeze, got {H}x{W}'
            
            # Convert to uint8 in range [0, 255] for PIL
            # img_data is in range [0, 1] from ToTensor()
            img_uint8 = (img_data * 255).astype(np.uint8)
            # Save directly using PIL to guarantee exact pixel dimensions
            img_pil = Image.fromarray(img_uint8, mode='L')  # 'L' mode for grayscale
            assert img_pil.size == (32, 32), f'PIL image size is {img_pil.size}, expected (32, 32)'
            
            # Metadata in filename: {prefix}_alpha{alpha}_rank{i}_idx{original_idx}_r{stein_abs}_label{y}.png
            stein_abs = r_abs_fashion[idx]
            alpha_str = f'{alpha:.3f}'.replace('.', '_')
            img_pil.save(f'{top_images_dir}/{prefix}_fashion_alpha{alpha_str}_rank{i}_idx{original_idx}_r{stein_abs:.4f}_label{y}.png')
    
    # Save top-k and bottom-k Fashion-MNIST images for each alpha
    for alpha in alphas_sorted:
        r_relative = r_mixed_dict[alpha] - mean_stein_train
        datatype = datatype_dict[alpha]
        
        print(f'  Saving Fashion-MNIST images for alpha={alpha:.3f}...')
        save_topk_fashion_images(mixed_test_sets[alpha], r_relative, datatype, k=10, prefix='top', alpha=alpha)
        save_topk_fashion_images(mixed_test_sets[alpha], r_relative, datatype, k=10, prefix='bottom', alpha=alpha)
    
    print(f'Saved top-k and bottom-k Fashion-MNIST images to {top_images_dir}/')
    
    # Save all results
    print(f'\nSaving results to {output_dir}/...')
    for alpha in alphas:
        np.save(f'{output_dir}/resid_alpha_{alpha:.3f}.npy', r_mixed_dict[alpha])
        np.save(f'{output_dir}/fvals_alpha_{alpha:.3f}.npy', f_mixed_dict[alpha])
        np.save(f'{output_dir}/labels_alpha_{alpha:.3f}.npy', y_mixed_dict[alpha])
        np.save(f'{output_dir}/datatype_alpha_{alpha:.3f}.npy', datatype_dict[alpha])
        np.save(f'{output_dir}/rejection_two_sided_alpha_{alpha:.3f}.npy', rejection_two_sided_dict[alpha])
        np.save(f'{output_dir}/rejection_one_sided_alpha_{alpha:.3f}.npy', rejection_one_sided_dict[alpha])
    
    # Save null distribution for reference
    np.save(f'{output_dir}/null_distribution.npy', null_distribution)
    np.save(f'{output_dir}/null_distribution_abs.npy', null_distribution_abs)
    
    # Save comprehensive statistics
    def get_confidence(fvals):
        if fvals.ndim == 1:
            return fvals
        else:
            return fvals.max(axis=1)
    
    stats = {
        'training_baseline': {
            'mean_stein_score': float(mean_stein_train),
            'count': len(r_train)
        },
        'mixed_test_sets': {
            'alphas': [float(a) for a in alphas],
            'per_alpha': {
                str(alpha): {
                    'count': len(r_mixed_dict[alpha]),
                    'loss': float(mixed_losses[alpha]),
                    'accuracy': float(mixed_accuracies[alpha]),
                    'residual_mean': float(r_mixed_dict[alpha].mean()),
                    'residual_std': float(r_mixed_dict[alpha].std()),
                    'residual_median': float(np.median(np.abs(r_mixed_dict[alpha]))),
                    'residual_mean_relative': float(r_mixed_dict[alpha].mean() - mean_stein_train),
                    'fval_mean': float(f_mixed_dict[alpha].mean()),
                    'fval_std': float(f_mixed_dict[alpha].std()),
                    'corr_abs_residual_confidence': float(np.corrcoef(np.abs(r_mixed_dict[alpha]), (1.0 - get_confidence(f_mixed_dict[alpha])))[0,1]),
                    'stein_mean_mnist': float(stein_mean_mnist[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'stein_mean_fashion': float(stein_mean_fashion[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'stein_std_mnist': float(stein_std_mnist[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'stein_std_fashion': float(stein_std_fashion[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'stein_abs_mean_mnist': float(stein_abs_mean_mnist[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'stein_abs_mean_fashion': float(stein_abs_mean_fashion[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'corr_stein_datatype': float(correlations[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'corr_abs_stein_datatype': float(correlations_abs[alphas_sorted.index(alpha)]) if alpha in alphas_sorted else None,
                    'rejection_rate_two_sided': float(rejection_rate_two_sided[alpha]) if alpha in rejection_rate_two_sided else None,
                    'rejection_rate_one_sided': float(rejection_rate_one_sided[alpha]) if alpha in rejection_rate_one_sided else None,
                    'rejection_rate_two_sided_mnist': float(rejection_rate_two_sided_mnist[alpha]) if alpha in rejection_rate_two_sided_mnist else None,
                    'rejection_rate_one_sided_mnist': float(rejection_rate_one_sided_mnist[alpha]) if alpha in rejection_rate_one_sided_mnist else None,
                    'rejection_rate_two_sided_fashion': float(rejection_rate_two_sided_fashion[alpha]) if alpha in rejection_rate_two_sided_fashion else None,
                    'rejection_rate_one_sided_fashion': float(rejection_rate_one_sided_fashion[alpha]) if alpha in rejection_rate_one_sided_fashion else None
                } for alpha in alphas
            }
        },
        'hypothesis_testing': {
            'significance_level': alpha_significance,
            'null_distribution_size': len(null_distribution),
            'null_2_5th_percentile': float(null_2_5th),
            'null_97_5th_percentile': float(null_97_5th),
            'null_95th_percentile_abs': float(null_95th_abs)
        },
        'parameters': {
            'hutchinson_probes': args.hutchinson_probes,
            'alphas': [float(a) for a in alphas],
            'batch_size': args.batch_size,
            'classifier_ckpt': classifier_ckpt_path,
            'score_ckpt': score_ckpt_path if score_model is not None else None,
            'device': str(device),
            'score_model_type': args.score_model_type
        }
    }
    
    with open(f'{output_dir}/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f'\nSaved all results to {output_dir}/')
    print(f'  - Residuals: resid_alpha_*.npy')
    print(f'  - Function values: fvals_alpha_*.npy')
    print(f'  - Labels: labels_alpha_*.npy')
    print(f'  - Datatypes: datatype_alpha_*.npy')
    print(f'  - Rejection flags: rejection_two_sided_alpha_*.npy, rejection_one_sided_alpha_*.npy')
    print(f'  - Null distributions: null_distribution.npy, null_distribution_abs.npy')
    print(f'  - Statistics: statistics.json')
    print(f'  - Plots: metrics_vs_alpha.png, accuracy_vs_stein.png, stein_histogram_alpha_*.png, stein_by_datatype.png, hypothesis_testing.png')

# ---------------------------
# Argument parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Stein shift detection pipeline for mixed MNIST/Fashion-MNIST')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST/Fashion-MNIST data (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training and evaluation (default: 128)')
    parser.add_argument('--classifier-epochs', type=int, default=5,
                       help='Number of epochs to train classifier (default: 5)')
    parser.add_argument('--classifier-ckpt', type=str, default=None,
                       help='Path to classifier checkpoint file. Default: checkpoints/classifier_32x32.pth')
    parser.add_argument('--cached-classifier', action='store_true',
                       help='Load classifier from checkpoint. Raises error if checkpoint does not exist.')
    parser.add_argument('--score-model-type', type=str, default='unet', choices=['small', 'unet'],
                       help='Type of score model to use: "small" for SmallScoreNet32x32 (simple DSM), "unet" for UNetScore (annealed DSM). Default: unet')
    parser.add_argument('--score-epochs', type=int, default=5,
                       help='Number of epochs to train score model (default: 5)')
    parser.add_argument('--score-ckpt', type=str, default=None,
                       help='Path to score model checkpoint file. Default: checkpoints/score_unet_32x32.pth or checkpoints/score_32x32.pth based on model type')
    parser.add_argument('--cached-score', action='store_true',
                       help='Load score model from checkpoint. Raises error if checkpoint does not exist.')
    parser.add_argument('--hutchinson-probes', type=int, default=1,
                       help='Number of Hutchinson probes for Laplacian estimate (1-4 typical, default: 1)')
    parser.add_argument('--stein-operator', type=str, default='laplacian', choices=['laplacian', 'alternative'],
                       help='Stein operator to use: "laplacian" (default) or "alternative" (f(x)*grad log p(x) + grad f(x))')
    
    args = parser.parse_args()
    
    # Validation
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if args.classifier_epochs <= 0:
        parser.error('--classifier-epochs must be positive')
    if args.score_epochs <= 0:
        parser.error('--score-epochs must be positive')
    if args.hutchinson_probes <= 0:
        parser.error('--hutchinson-probes must be positive')
    
    return args

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == '__main__':
    args = parse_args()
    evaluate_pipeline(args)

