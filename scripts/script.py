"""
stein_mnist_pipeline.py

Complete pipeline:
 - Train an AlexNet-like classifier on MNIST (or load checkpoint).
 - Create two test variants from held-out test set:
     * translated (small translations)  --> we expect classifier to be *robust* and predictions similar
     * rotated 90 degrees               --> we expect classifier predictions to *change*
 - Compute per-sample Stein residuals:
     r(x) = Laplacian_x f(x) + s(x)^T grad_x f(x)
   using:
     - autodiff for grad_x f
     - Hutchinson estimator for Laplacian (Hessian-vector products)
     - a score model s(x) (optional; can be trained by denoising score matching)
 - Aggregate residuals and output simple diagnostics & visualizations.

Notes:
 - This script is intended to be run locally (GPU recommended).
 - Training a score model (denoising score matching) is optional but recommended for faithful Stein residuals.
 - If you prefer to avoid training a score model, use the kernelized fallback (MMD witness) included as a simple baseline.
"""

import os
import argparse
import math
import json
import shutil
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

from src.models import ClassifierNet, SmallScoreNet, UNetScore
from src.utils import get_device
from src.data import get_mnist_dataloaders, create_translated_dataset, create_rotated_dataset, create_translated_dataset_fixed_size, create_rotated_dataset_fixed_angle
from src.eval_functions import compute_score_outputs, compute_stein_residuals, compute_stein_residuals_classic, compute_stein_residuals_per_dim, compute_stein_residuals_alternative, compute_log_likelihood_score_model
from src.training import train_classifier, train_score_model, train_score_model_annealed

# ---------------------------
# Evaluation driver & plotting
# ---------------------------
def evaluate_pipeline(args):
    device = get_device()
    print(f'Using device: {device}')
    train_loader, val_loader, test_loader, train_ds, val_ds, test_set = get_mnist_dataloaders(batch_size=args.batch_size, data_dir=args.data_dir, device=device)

    # classifier
    clf = ClassifierNet()
    classifier_ckpt_path = args.classifier_ckpt or 'checkpoints/classifier.pth'
    
    if args.cached_classifier:
        # Load from cache - raise error if checkpoint doesn't exist
        if not os.path.exists(classifier_ckpt_path):
            raise FileNotFoundError(f'Classifier checkpoint not found: {classifier_ckpt_path}. Use --cached-classifier only when checkpoint exists.')
        clf.load_state_dict(torch.load(classifier_ckpt_path, map_location=device))
        clf.to(device)
        print('Loaded classifier from', classifier_ckpt_path)
    else:
        # Train and save (overwrite if exists)
        print('Training classifier (this will take some time).')
        clf = train_classifier(clf, train_loader, val_loader, device, epochs=args.classifier_epochs, checkpoint_path=classifier_ckpt_path)
        print('Classifier trained and saved to', classifier_ckpt_path)

    # Prepare datasets: original test, 10 translated datasets, and 10 rotated datasets
    # Use single-process loading for broad compatibility (some environments disallow torch shared-memory helpers).
    test_loader_orig = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Split test set into 10 groups of 1K samples each
    test_size = len(test_set)
    samples_per_group = test_size // 10
    test_indices = list(range(test_size))
    random.shuffle(test_indices)  # Shuffle for random assignment
    
    # Create 10 translated datasets with fixed translation sizes: [0, 2, 4, ..., 18]
    translation_sizes = [i * 2 for i in range(10)]  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    translated_datasets = {}
    translated_loaders = {}
    for i, t_size in enumerate(translation_sizes):
        start_idx = i * samples_per_group
        end_idx = start_idx + samples_per_group if i < 9 else test_size
        subset_indices = test_indices[start_idx:end_idx]
        subset = Subset(test_set, subset_indices)
        translated_datasets[t_size] = create_translated_dataset_fixed_size(subset, translation_size=t_size)
        translated_loaders[t_size] = DataLoader(translated_datasets[t_size], batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create 10 rotated datasets with fixed angles: [-90, -72, -54, -36, -18, 0, 18, 36, 54, 72] (steps of 18)
    rotation_angles = [-90, -72, -54, -36, -18, 0, 18, 36, 54, 72]
    rotated_datasets = {}
    rotated_loaders = {}
    for i, angle in enumerate(rotation_angles):
        start_idx = i * samples_per_group
        end_idx = start_idx + samples_per_group if i < 9 else test_size
        subset_indices = test_indices[start_idx:end_idx]
        subset = Subset(test_set, subset_indices)
        rotated_datasets[angle] = create_rotated_dataset_fixed_angle(subset, angle=angle)
        rotated_loaders[angle] = DataLoader(rotated_datasets[angle], batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Verify image sizes
    sample_orig, _ = next(iter(test_loader_orig))
    print(f'\nImage size verification:')
    print(f'  Original test set: {sample_orig.shape[2]}x{sample_orig.shape[3]} (expected: 64x64)')
    assert sample_orig.shape[2] == 64 and sample_orig.shape[3] == 64, 'Original images should be 64x64'
    
    # Verify a sample from translated and rotated datasets
    sample_trans, _ = next(iter(translated_loaders[translation_sizes[0]]))
    sample_rot, _ = next(iter(rotated_loaders[rotation_angles[0]]))
    assert sample_trans.shape[2] == 64 and sample_trans.shape[3] == 64, 'Translated images should be 64x64'
    assert sample_rot.shape[2] == 64 and sample_rot.shape[3] == 64, 'Rotated images should be 64x64'

    # Score model: either load from cache or train
    score_model = None
    sigmas = None
    use_unet = (args.score_model_type == 'unet')
    
    # Set default checkpoint path based on model type
    if args.score_ckpt is None:
        score_ckpt_path = 'checkpoints/score_unet.pth' if use_unet else 'checkpoints/score.pth'
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
            score_model = SmallScoreNet()
            score_model.load_state_dict(torch.load(score_ckpt_path, map_location=device))
            score_model.to(device)
            sigmas = None  # SmallScoreNet does not use sigmas
        
        print(f'Loaded {args.score_model_type} score model from {score_ckpt_path}')
    else:
        # Train and save (overwrite if exists)
        if use_unet:
            print('Training UNet score model (annealed DSM)...')
            score_model = UNetScore(base_ch=64)
            score_model, sigmas = train_score_model_annealed(score_model, train_ds, device, epochs=args.score_epochs, 
                                                              batch_size=args.batch_size, lr=2e-4, ckpt=score_ckpt_path)
        else:
            print('Training Small score model (simple DSM)...')
            score_model = SmallScoreNet()
            score_model = train_score_model(score_model, train_ds, device, epochs=args.score_epochs,
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
            for x, y in dataloader:
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
    
    # Compute loss and accuracy for original test set
    print('\nComputing loss and accuracy...')
    loss_orig, acc_orig = compute_loss_and_accuracy(clf, test_loader_orig, device)
    print(f'Original test set - Loss: {loss_orig:.4f}, Accuracy: {acc_orig:.4f}')
    
    # Compute loss and accuracy for all translated datasets
    translation_losses = {}
    translation_accuracies = {}
    for t_size in translation_sizes:
        loss, acc = compute_loss_and_accuracy(clf, translated_loaders[t_size], device)
        translation_losses[t_size] = loss
        translation_accuracies[t_size] = acc
        print(f'Translation size {t_size} - Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    
    # Compute loss and accuracy for all rotated datasets
    rotation_losses = {}
    rotation_accuracies = {}
    for angle in rotation_angles:
        loss, acc = compute_loss_and_accuracy(clf, rotated_loaders[angle], device)
        rotation_losses[angle] = loss
        rotation_accuracies[angle] = acc
        print(f'Rotation angle {angle}° - Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    
    # Compute Stein residuals for all test sets
    if args.use_per_dim_stein:
        print(f'\nComputing per-dimension Stein residuals (aggregation: {args.stein_aggregation}) on original test set...')
        r_orig, f_orig, y_orig, r_orig_per_dim = compute_stein_residuals_per_dim(
            test_loader_orig, clf, score_model, device, num_probes=args.hutchinson_probes, 
            sigmas=sigmas, aggregation=args.stein_aggregation)
        
        # Compute for all translated datasets
        r_trans_dict = {}
        f_trans_dict = {}
        y_trans_dict = {}
        r_trans_per_dim_dict = {}
        for t_size in translation_sizes:
            print(f'Computing per-dimension Stein residuals for translation size {t_size}...')
            r_trans_dict[t_size], f_trans_dict[t_size], y_trans_dict[t_size], r_trans_per_dim_dict[t_size] = compute_stein_residuals_per_dim(
                translated_loaders[t_size], clf, score_model, device, num_probes=args.hutchinson_probes, 
                sigmas=sigmas, aggregation=args.stein_aggregation)
        
        # Compute for all rotated datasets
        r_rot_dict = {}
        f_rot_dict = {}
        y_rot_dict = {}
        r_rot_per_dim_dict = {}
        for angle in rotation_angles:
            print(f'Computing per-dimension Stein residuals for rotation angle {angle}°...')
            r_rot_dict[angle], f_rot_dict[angle], y_rot_dict[angle], r_rot_per_dim_dict[angle] = compute_stein_residuals_per_dim(
                rotated_loaders[angle], clf, score_model, device, num_probes=args.hutchinson_probes, 
                sigmas=sigmas, aggregation=args.stein_aggregation)
    else:
        print('\nComputing Stein residuals on original test set...')
        if args.stein_operator == 'alternative':
            r_orig, f_orig, y_orig = compute_stein_residuals_alternative(test_loader_orig, clf, score_model, device, sigmas=sigmas)
        else:
            r_orig, f_orig, y_orig = compute_stein_residuals(test_loader_orig, clf, score_model, device, num_probes=args.hutchinson_probes, sigmas=sigmas)
        
        # Compute for all translated datasets
        r_trans_dict = {}
        f_trans_dict = {}
        y_trans_dict = {}
        for t_size in translation_sizes:
            print(f'Computing Stein residuals for translation size {t_size}...')
            if args.stein_operator == 'alternative':
                r_trans_dict[t_size], f_trans_dict[t_size], y_trans_dict[t_size] = compute_stein_residuals_alternative(
                    translated_loaders[t_size], clf, score_model, device, sigmas=sigmas)
            else:
                r_trans_dict[t_size], f_trans_dict[t_size], y_trans_dict[t_size] = compute_stein_residuals(
                    translated_loaders[t_size], clf, score_model, device, num_probes=args.hutchinson_probes, sigmas=sigmas)
        
        # Compute for all rotated datasets
        r_rot_dict = {}
        f_rot_dict = {}
        y_rot_dict = {}
        for angle in rotation_angles:
            print(f'Computing Stein residuals for rotation angle {angle}°...')
            if args.stein_operator == 'alternative':
                r_rot_dict[angle], f_rot_dict[angle], y_rot_dict[angle] = compute_stein_residuals_alternative(
                    rotated_loaders[angle], clf, score_model, device, sigmas=sigmas)
            else:
                r_rot_dict[angle], f_rot_dict[angle], y_rot_dict[angle] = compute_stein_residuals(
                    rotated_loaders[angle], clf, score_model, device, num_probes=args.hutchinson_probes, sigmas=sigmas)

    # Compute Stein residuals on a subset of training data for baseline
    print('\nComputing Stein residuals on training data subset (5K samples) for baseline...')
    train_size = len(train_ds)
    train_subset_size = min(5000, train_size)
    train_subset_indices = random.sample(range(train_size), train_subset_size)
    train_subset = Subset(train_ds, train_subset_indices)
    train_subset_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.use_per_dim_stein:
        r_train, _, _, _ = compute_stein_residuals_per_dim(
            train_subset_loader, clf, score_model, device, num_probes=args.hutchinson_probes,
            sigmas=sigmas, aggregation=args.stein_aggregation)
    else:
        if args.stein_operator == 'alternative':
            r_train, _, _ = compute_stein_residuals_alternative(
                train_subset_loader, clf, score_model, device, sigmas=sigmas)
        else:
            r_train, _, _ = compute_stein_residuals(
                train_subset_loader, clf, score_model, device, num_probes=args.hutchinson_probes, sigmas=sigmas)

    mean_stein_train = r_train.mean()
    print(f'Training data baseline - Mean Stein Score: {mean_stein_train:.4e}')

    # Compute score model outputs for all test sets
    print('\nComputing score model outputs...')
    score_norm_orig, score_out_orig, _ = compute_score_outputs(test_loader_orig, score_model, device, sigmas=sigmas)
    
    score_norm_trans_dict = {}
    score_out_trans_dict = {}
    for t_size in translation_sizes:
        score_norm_trans_dict[t_size], score_out_trans_dict[t_size], _ = compute_score_outputs(
            translated_loaders[t_size], score_model, device, sigmas=sigmas)
    
    score_norm_rot_dict = {}
    score_out_rot_dict = {}
    for angle in rotation_angles:
        score_norm_rot_dict[angle], score_out_rot_dict[angle], _ = compute_score_outputs(
            rotated_loaders[angle], score_model, device, sigmas=sigmas)
    
    # Compute log-likelihoods under score model for all test sets
    print('\nComputing log-likelihoods under score model...')
    log_likelihood_orig, _ = compute_log_likelihood_score_model(test_loader_orig, score_model, device, sigmas=sigmas)
    
    log_likelihood_trans_dict = {}
    for t_size in translation_sizes:
        log_likelihood_trans_dict[t_size], _ = compute_log_likelihood_score_model(
            translated_loaders[t_size], score_model, device, sigmas=sigmas)
    
    log_likelihood_rot_dict = {}
    for angle in rotation_angles:
        log_likelihood_rot_dict[angle], _ = compute_log_likelihood_score_model(
            rotated_loaders[angle], score_model, device, sigmas=sigmas)

    # Basic diagnostics
    def summarize(name, r, fvals, labels):
        print(f'\n---- {name} ----')
        print('count:', len(r))
        print('r mean {:.4e}, std {:.4e}, median {:.4e}'.format(r.mean(), r.std(), np.median(np.abs(r))))
        # correlation between |r| and model confidence (fvals)
        # Handle both scalar fvals (1D) and per-dimension fvals (2D: samples x classes)
        if fvals.ndim == 1:
            # Scalar case: use max probability as confidence
            confidence = fvals
        else:
            # Per-dimension case: use max probability across classes as confidence
            confidence = fvals.max(axis=1)
        print('corr(|r|, 1 - confidence):', np.corrcoef(np.abs(r), (1.0 - confidence))[0,1])
        # fraction of large residuals
        thr = np.percentile(np.abs(r_orig), 95)  # threshold from orig distribution
        print('frac above orig 95th pct:', (np.abs(r) > thr).mean())
    
    summarize('Original test', r_orig, f_orig, y_orig)
    # Summarize for all translation sizes
    for t_size in translation_sizes:
        summarize(f'Translation size {t_size}', r_trans_dict[t_size], f_trans_dict[t_size], y_trans_dict[t_size])
    # Summarize for all rotation angles
    for angle in rotation_angles:
        summarize(f'Rotation angle {angle}°', r_rot_dict[angle], f_rot_dict[angle], y_rot_dict[angle])

    # Visualization: histograms
    output_dir = 'results/stein_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histogram with original and selected translated/rotated datasets
    plt.figure(figsize=(12, 6))
    plt.hist(r_orig, bins=100, alpha=0.5, label='Original', density=True)
    # Plot a few key translation sizes
    for t_size in [0, 6, 12, 18]:
        plt.hist(r_trans_dict[t_size], bins=100, alpha=0.5, label=f'Translation {t_size}', density=True)
    # Plot a few key rotation angles
    for angle in [-90, -36, 0, 36, 90]:
        if angle in r_rot_dict:
            plt.hist(r_rot_dict[angle], bins=100, alpha=0.5, label=f'Rotation {angle}°', density=True)
    plt.legend()
    plt.title('Stein residual histograms')
    plt.xlabel('Residual value')
    plt.ylabel('Density')
    hist_path = f'{output_dir}/stein_residuals_hist.png'
    plt.savefig(hist_path)
    print(f'Saved histogram to {hist_path}')
    
    # Plot translation size vs mean residual and accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    trans_sizes_sorted = sorted(translation_sizes)
    trans_means = [np.abs(r_trans_dict[t]).mean() for t in trans_sizes_sorted]
    plt.plot(trans_sizes_sorted, trans_means, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Translation Size (L∞ norm)')
    plt.ylabel('Mean |Stein Residual|')
    plt.title('Stein Residual vs Translation Size')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    trans_accs = [translation_accuracies[t] for t in trans_sizes_sorted]
    plt.plot(trans_sizes_sorted, trans_accs, marker='o', linewidth=2, markersize=8, color='green')
    plt.xlabel('Translation Size (L∞ norm)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Translation Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/translation_analysis.png')
    print(f'Saved translation analysis to {output_dir}/translation_analysis.png')
    
    # Plot rotation angle vs mean residual and accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    rot_angles_sorted = sorted(rotation_angles)
    rot_means = [np.abs(r_rot_dict[a]).mean() for a in rot_angles_sorted]
    plt.plot(rot_angles_sorted, rot_means, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Mean |Stein Residual|')
    plt.title('Stein Residual vs Rotation Angle')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    rot_accs = [rotation_accuracies[a] for a in rot_angles_sorted]
    plt.plot(rot_angles_sorted, rot_accs, marker='o', linewidth=2, markersize=8, color='green')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Rotation Angle')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rotation_analysis.png')
    print(f'Saved rotation analysis to {output_dir}/rotation_analysis.png')
    
    # Comprehensive plots showing relationships between accuracy, shift size, stein scores, and abs(stein scores)
    # First, compute all data and determine shared y-axis limits for accuracy plots
    z_score = 1.96
    
    # Translation data
    trans_sizes_sorted = sorted(translation_sizes)
    trans_stein_mean = [r_trans_dict[t].mean() for t in trans_sizes_sorted]
    trans_stein_std = [r_trans_dict[t].std() for t in trans_sizes_sorted]
    trans_stein_sem = [std / np.sqrt(len(r_trans_dict[t])) for std, t in zip(trans_stein_std, trans_sizes_sorted)]
    trans_stein_abs_mean = [np.abs(r_trans_dict[t]).mean() for t in trans_sizes_sorted]
    trans_stein_abs_std = [np.abs(r_trans_dict[t]).std() for t in trans_sizes_sorted]
    trans_stein_abs_sem = [std / np.sqrt(len(r_trans_dict[t])) for std, t in zip(trans_stein_abs_std, trans_sizes_sorted)]
    trans_accs = [translation_accuracies[t] for t in trans_sizes_sorted]
    trans_acc_cis = []
    for t_size in trans_sizes_sorted:
        n = len(r_trans_dict[t_size])
        p = translation_accuracies[t_size]
        se = np.sqrt(p * (1 - p) / n)
        trans_acc_cis.append(z_score * se)
    
    # Rotation data
    rot_angles_sorted = sorted(rotation_angles)
    rot_accs = [rotation_accuracies[a] for a in rot_angles_sorted]
    rot_acc_cis = []
    for angle in rot_angles_sorted:
        n = len(r_rot_dict[angle])
        p = rotation_accuracies[angle]
        se = np.sqrt(p * (1 - p) / n)
        rot_acc_cis.append(z_score * se)
    
    # Determine shared y-axis limits for accuracy plots (used in both translation and rotation plots)
    all_accs = trans_accs + rot_accs
    all_acc_cis = trans_acc_cis + rot_acc_cis
    acc_y_min = min([acc - ci for acc, ci in zip(all_accs, all_acc_cis)])
    acc_y_max = max([acc + ci for acc, ci in zip(all_accs, all_acc_cis)])
    # Add some padding
    acc_y_range = acc_y_max - acc_y_min
    acc_y_min = acc_y_min - 0.05 * acc_y_range
    acc_y_max = acc_y_max + 0.05 * acc_y_range
    
    # For translations
    plt.figure(figsize=(12, 8))
    # Plot 1: Accuracy vs Translation Size
    plt.subplot(2, 2, 1)
    plt.plot(trans_sizes_sorted, trans_accs, marker='o', linewidth=2, markersize=8, color='green', label='Mean')
    plt.fill_between(trans_sizes_sorted,
                     [acc - ci for acc, ci in zip(trans_accs, trans_acc_cis)],
                     [acc + ci for acc, ci in zip(trans_accs, trans_acc_cis)],
                     alpha=0.3, color='green', label='95% CI')
    plt.xlabel('Translation Size (L∞ norm)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Translation Size')
    plt.ylim(acc_y_min, acc_y_max)  # Use shared y-axis limits
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean Stein Score (relative to training) vs Translation Size
    plt.subplot(2, 2, 2)
    trans_stein_mean_relative = [m - mean_stein_train for m in trans_stein_mean]
    plt.plot(trans_sizes_sorted, trans_stein_mean_relative, marker='o', linewidth=2, markersize=8, color='blue', label='Mean (relative to training)')
    plt.fill_between(trans_sizes_sorted,
                     [m - z_score * sem for m, sem in zip(trans_stein_mean_relative, trans_stein_sem)],
                     [m + z_score * sem for m, sem in zip(trans_stein_mean_relative, trans_stein_sem)],
                     alpha=0.3, color='blue', label='95% CI')
    plt.xlabel('Translation Size (L∞ norm)')
    plt.ylabel('Mean Stein Score (relative to training)')
    plt.title('Mean Stein Score vs Translation Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Accuracy vs Mean Stein Score (scatter)
    plt.subplot(2, 2, 3)
    plt.scatter(trans_stein_mean, trans_accs, s=100, alpha=0.6, c=trans_sizes_sorted, cmap='viridis')
    plt.xlabel('Mean Stein Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Mean Stein Score')
    plt.colorbar(label='Translation Size')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Mean Log-Likelihood (scatter)
    plt.subplot(2, 2, 4)
    trans_log_likelihood_mean = [log_likelihood_trans_dict[t].mean() for t in trans_sizes_sorted]
    plt.scatter(trans_log_likelihood_mean, trans_accs, s=100, alpha=0.6, c=trans_sizes_sorted, cmap='viridis')
    plt.xlabel('Mean Log-Likelihood (Score Model)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Mean Log-Likelihood')
    plt.colorbar(label='Translation Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/translation_relationships.png', dpi=150)
    print(f'Saved translation relationships plot to {output_dir}/translation_relationships.png')
    
    # For rotations
    plt.figure(figsize=(12, 8))
    rot_stein_mean = [r_rot_dict[a].mean() for a in rot_angles_sorted]
    rot_stein_std = [r_rot_dict[a].std() for a in rot_angles_sorted]
    rot_stein_sem = [std / np.sqrt(len(r_rot_dict[a])) for std, a in zip(rot_stein_std, rot_angles_sorted)]
    rot_stein_abs_mean = [np.abs(r_rot_dict[a]).mean() for a in rot_angles_sorted]
    rot_stein_abs_std = [np.abs(r_rot_dict[a]).std() for a in rot_angles_sorted]
    rot_stein_abs_sem = [std / np.sqrt(len(r_rot_dict[a])) for std, a in zip(rot_stein_abs_std, rot_angles_sorted)]
    
    # Plot 1: Accuracy vs Rotation Angle (using shared y-axis limits)
    plt.subplot(2, 2, 1)
    plt.plot(rot_angles_sorted, rot_accs, marker='o', linewidth=2, markersize=8, color='green', label='Mean')
    plt.fill_between(rot_angles_sorted,
                     [acc - ci for acc, ci in zip(rot_accs, rot_acc_cis)],
                     [acc + ci for acc, ci in zip(rot_accs, rot_acc_cis)],
                     alpha=0.3, color='green', label='95% CI')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Rotation Angle')
    plt.ylim(acc_y_min, acc_y_max)  # Use shared y-axis limits
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean Stein Score (relative to training) vs Rotation Angle
    plt.subplot(2, 2, 2)
    # Debug: print values to verify baseline subtraction
    print(f'\nDebug: Training baseline = {mean_stein_train:.6e}')
    print(f'Debug: Rotation means before subtraction: {[f"{m:.6e}" for m in rot_stein_mean]}')
    rot_stein_mean_relative = [m - mean_stein_train for m in rot_stein_mean]
    print(f'Debug: Rotation means after subtraction: {[f"{m:.6e}" for m in rot_stein_mean_relative]}')
    plt.plot(rot_angles_sorted, rot_stein_mean_relative, marker='o', linewidth=2, markersize=8, color='blue', label='Mean (relative to training)')
    plt.fill_between(rot_angles_sorted,
                     [m - z_score * sem for m, sem in zip(rot_stein_mean_relative, rot_stein_sem)],
                     [m + z_score * sem for m, sem in zip(rot_stein_mean_relative, rot_stein_sem)],
                     alpha=0.3, color='blue', label='95% CI')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Mean Stein Score (relative to training)')
    plt.title('Mean Stein Score vs Rotation Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Accuracy vs Mean Stein Score (scatter)
    plt.subplot(2, 2, 3)
    plt.scatter(rot_stein_mean, rot_accs, s=100, alpha=0.6, c=rot_angles_sorted, cmap='viridis')
    plt.xlabel('Mean Stein Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Mean Stein Score')
    plt.colorbar(label='Rotation Angle (degrees)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Mean Log-Likelihood (scatter)
    plt.subplot(2, 2, 4)
    rot_log_likelihood_mean = [log_likelihood_rot_dict[a].mean() for a in rot_angles_sorted]
    plt.scatter(rot_log_likelihood_mean, rot_accs, s=100, alpha=0.6, c=rot_angles_sorted, cmap='viridis')
    plt.xlabel('Mean Log-Likelihood (Score Model)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Mean Log-Likelihood')
    plt.colorbar(label='Rotation Angle (degrees)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rotation_relationships.png', dpi=150)
    print(f'Saved rotation relationships plot to {output_dir}/rotation_relationships.png')
    
    # Combined plot: Translation and Rotation relationships side by side
    print('\nCreating combined translation and rotation relationships plot...')
    
    # Calculate shared y-axis limits for bottom plots (Stein scores)
    trans_stein_mean_relative = [m - mean_stein_train for m in trans_stein_mean]
    rot_stein_mean_relative = [m - mean_stein_train for m in rot_stein_mean]
    
    # Get all values including confidence intervals for both translation and rotation
    all_stein_values = []
    # Translation values with CIs
    for m, sem in zip(trans_stein_mean_relative, trans_stein_sem):
        all_stein_values.extend([m - z_score * sem, m + z_score * sem])
    # Rotation values with CIs
    for m, sem in zip(rot_stein_mean_relative, rot_stein_sem):
        all_stein_values.extend([m - z_score * sem, m + z_score * sem])
    
    stein_y_min = min(all_stein_values)
    stein_y_max = max(all_stein_values)
    # Add some padding
    stein_y_range = stein_y_max - stein_y_min
    stein_y_min = stein_y_min - 0.05 * stein_y_range
    stein_y_max = stein_y_max + 0.05 * stein_y_range
    
    plt.figure(figsize=(16, 10))
    
    # Top-left: Accuracy vs Translation Size
    plt.subplot(2, 2, 1)
    plt.plot(trans_sizes_sorted, trans_accs, marker='o', linewidth=2, markersize=8, color='green', label='Mean')
    plt.fill_between(trans_sizes_sorted,
                     [acc - ci for acc, ci in zip(trans_accs, trans_acc_cis)],
                     [acc + ci for acc, ci in zip(trans_accs, trans_acc_cis)],
                     alpha=0.3, color='green', label='95% CI')
    plt.xlabel('Translation Size (L∞ norm)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Translation Size')
    plt.ylim(acc_y_min, acc_y_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Top-right: Accuracy vs Rotation Angle
    plt.subplot(2, 2, 2)
    plt.plot(rot_angles_sorted, rot_accs, marker='o', linewidth=2, markersize=8, color='green', label='Mean')
    plt.fill_between(rot_angles_sorted,
                     [acc - ci for acc, ci in zip(rot_accs, rot_acc_cis)],
                     [acc + ci for acc, ci in zip(rot_accs, rot_acc_cis)],
                     alpha=0.3, color='green', label='95% CI')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Rotation Angle')
    plt.ylim(acc_y_min, acc_y_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom-left: Mean Stein Score vs Translation Size
    plt.subplot(2, 2, 3)
    plt.plot(trans_sizes_sorted, trans_stein_mean_relative, marker='o', linewidth=2, markersize=8, color='blue', label='Mean (relative to training)')
    plt.fill_between(trans_sizes_sorted,
                     [m - z_score * sem for m, sem in zip(trans_stein_mean_relative, trans_stein_sem)],
                     [m + z_score * sem for m, sem in zip(trans_stein_mean_relative, trans_stein_sem)],
                     alpha=0.3, color='blue', label='95% CI')
    plt.xlabel('Translation Size (L∞ norm)')
    plt.ylabel('Mean Stein Score (relative to training)')
    plt.title('Mean Stein Score vs Translation Size')
    plt.ylim(stein_y_min, stein_y_max)  # Use shared y-axis limits
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Bottom-right: Mean Stein Score vs Rotation Angle
    plt.subplot(2, 2, 4)
    plt.plot(rot_angles_sorted, rot_stein_mean_relative, marker='o', linewidth=2, markersize=8, color='blue', label='Mean (relative to training)')
    plt.fill_between(rot_angles_sorted,
                     [m - z_score * sem for m, sem in zip(rot_stein_mean_relative, rot_stein_sem)],
                     [m + z_score * sem for m, sem in zip(rot_stein_mean_relative, rot_stein_sem)],
                     alpha=0.3, color='blue', label='95% CI')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Mean Stein Score (relative to training)')
    plt.title('Mean Stein Score vs Rotation Angle')
    plt.ylim(stein_y_min, stein_y_max)  # Use shared y-axis limits
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_translation_rotation_relationships.png', dpi=150)
    print(f'Saved combined translation and rotation relationships plot to {output_dir}/combined_translation_rotation_relationships.png')

    # --------------------------------------------
    # ICML-friendly overlay layout (two panels)
    # --------------------------------------------
    # Each panel overlays accuracy (left y-axis) and Stein score (right y-axis)
    # for a single x-axis (translation / rotation). This is easier to read in
    # two-column format than a 2×2 grid.
    # Save as two separate single-panel figures (more flexible for two-column layouts).
    def _save_overlay_panel(*, kind: str) -> None:
        if kind == "translation":
            x = trans_sizes_sorted
            acc = trans_accs
            acc_ci = trans_acc_cis
            stein = trans_stein_mean_relative
            stein_sem = trans_stein_sem
            title = "Translation"
            xlabel = "Translation size (L∞)"
            out_name = "combined_translation_rotation_relationships_overlaid__translation.png"
        elif kind == "rotation":
            x = rot_angles_sorted
            acc = rot_accs
            acc_ci = rot_acc_cis
            stein = rot_stein_mean_relative
            stein_sem = rot_stein_sem
            title = "Rotation"
            xlabel = "Rotation angle (deg)"
            out_name = "combined_translation_rotation_relationships_overlaid__rotation.png"
        else:
            raise ValueError(kind)

        fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.8))
        ax2 = ax.twinx()

        l1 = ax.plot(x, acc, marker='o', linewidth=2, markersize=6, color='green', label='Accuracy')
        ax.fill_between(
            x,
            [a - ci for a, ci in zip(acc, acc_ci)],
            [a + ci for a, ci in zip(acc, acc_ci)],
            alpha=0.20,
            color='green',
            linewidth=0,
        )
        l2 = ax2.plot(
            x,
            stein,
            marker='o',
            linewidth=2,
            markersize=6,
            color='blue',
            label='TASTE',
        )
        ax2.fill_between(
            x,
            [m - z_score * sem for m, sem in zip(stein, stein_sem)],
            [m + z_score * sem for m, sem in zip(stein, stein_sem)],
            alpha=0.18,
            color='blue',
            linewidth=0,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('TASTE')
        # "Lift" the accuracy curve visually by ensuring the y-axis starts near 0.
        # This avoids the curve hugging the bottom when accuracy is low.
        ax.set_ylim(-0.10, acc_y_max)
        ax2.set_ylim(stein_y_min, stein_y_max)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.35)
        ax.grid(True, alpha=0.3)

        lines = list(l1) + list(l2)
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, loc='best', frameon=False)

        fig.tight_layout()
        plt.savefig(f'{output_dir}/{out_name}', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved overlaid {kind} layout to {output_dir}/{out_name}')

    _save_overlay_panel(kind="translation")
    _save_overlay_panel(kind="rotation")
    
    # Comparison plots: Accuracy, Confidence, and Stein Score by test set
    print('\nCreating comparison plots: Accuracy, Confidence, and Stein Score by test set...')
    
    # Helper function to compute mean confidence from fvals
    def get_mean_confidence(fvals):
        """Compute mean confidence (max softmax probability) from fvals."""
        if fvals.ndim == 1:
            # Scalar case: fvals is already confidence
            return fvals.mean()
        else:
            # Per-dimension case: use max probability across classes
            return fvals.max(axis=1).mean()
    
    # For translations: collect all metrics
    trans_comparison_data = {
        'size': [],
        'accuracy': [],
        'confidence': [],
        'stein_score': []
    }
    
    # Original test set
    trans_comparison_data['size'].append(0)
    trans_comparison_data['accuracy'].append(acc_orig)
    trans_comparison_data['confidence'].append(get_mean_confidence(f_orig))
    trans_comparison_data['stein_score'].append(r_orig.mean() - mean_stein_train)
    
    # Translated test sets
    for t_size in sorted(translation_sizes):
        trans_comparison_data['size'].append(t_size)
        trans_comparison_data['accuracy'].append(translation_accuracies[t_size])
        trans_comparison_data['confidence'].append(get_mean_confidence(f_trans_dict[t_size]))
        trans_comparison_data['stein_score'].append(r_trans_dict[t_size].mean() - mean_stein_train)
    
    # For rotations: collect all metrics
    rot_comparison_data = {
        'angle': [],
        'accuracy': [],
        'confidence': [],
        'stein_score': []
    }
    
    # Original test set
    rot_comparison_data['angle'].append(0)
    rot_comparison_data['accuracy'].append(acc_orig)
    rot_comparison_data['confidence'].append(get_mean_confidence(f_orig))
    rot_comparison_data['stein_score'].append(r_orig.mean() - mean_stein_train)
    
    # Rotated test sets
    for angle in sorted(rotation_angles):
        rot_comparison_data['angle'].append(angle)
        rot_comparison_data['accuracy'].append(rotation_accuracies[angle])
        rot_comparison_data['confidence'].append(get_mean_confidence(f_rot_dict[angle]))
        rot_comparison_data['stein_score'].append(r_rot_dict[angle].mean() - mean_stein_train)
    
    # Plot for translations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy vs Translation Size
    axes[0].plot(trans_comparison_data['size'], trans_comparison_data['accuracy'], 
                 marker='o', linewidth=2, markersize=8, color='green', label='Accuracy')
    axes[0].set_xlabel('Translation Size (L∞ norm)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Translation Size')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Confidence vs Translation Size
    axes[1].plot(trans_comparison_data['size'], trans_comparison_data['confidence'], 
                 marker='s', linewidth=2, markersize=8, color='orange', label='Mean Confidence')
    axes[1].set_xlabel('Translation Size (L∞ norm)')
    axes[1].set_ylabel('Mean Confidence (Max Softmax)')
    axes[1].set_title('Confidence vs Translation Size')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Stein Score (relative) vs Translation Size
    axes[2].plot(trans_comparison_data['size'], trans_comparison_data['stein_score'], 
                 marker='^', linewidth=2, markersize=8, color='blue', label='Mean Stein Score (relative)')
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Training baseline')
    axes[2].set_xlabel('Translation Size (L∞ norm)')
    axes[2].set_ylabel('Mean Stein Score (relative to training)')
    axes[2].set_title('Stein Score vs Translation Size')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/translation_comparison.png', dpi=150)
    print(f'Saved translation comparison plot to {output_dir}/translation_comparison.png')
    plt.close()
    
    # Plot for rotations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy vs Rotation Angle
    axes[0].plot(rot_comparison_data['angle'], rot_comparison_data['accuracy'], 
                 marker='o', linewidth=2, markersize=8, color='green', label='Accuracy')
    axes[0].set_xlabel('Rotation Angle (degrees)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Rotation Angle')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Confidence vs Rotation Angle
    axes[1].plot(rot_comparison_data['angle'], rot_comparison_data['confidence'], 
                 marker='s', linewidth=2, markersize=8, color='orange', label='Mean Confidence')
    axes[1].set_xlabel('Rotation Angle (degrees)')
    axes[1].set_ylabel('Mean Confidence (Max Softmax)')
    axes[1].set_title('Confidence vs Rotation Angle')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Stein Score (relative) vs Rotation Angle
    axes[2].plot(rot_comparison_data['angle'], rot_comparison_data['stein_score'], 
                 marker='^', linewidth=2, markersize=8, color='blue', label='Mean Stein Score (relative)')
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Training baseline')
    axes[2].set_xlabel('Rotation Angle (degrees)')
    axes[2].set_ylabel('Mean Stein Score (relative to training)')
    axes[2].set_title('Stein Score vs Rotation Angle')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rotation_comparison.png', dpi=150)
    print(f'Saved rotation comparison plot to {output_dir}/rotation_comparison.png')
    plt.close()
    
    # Diagnostic: Visualize translated images to understand accuracy drop
    print('\nCreating diagnostic visualization of translated images...')
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Sample a few images from the original test set
    sample_batch = next(iter(test_loader_orig))
    sample_images = sample_batch[0][:5]  # First 5 images
    sample_labels = sample_batch[1][:5]
    
    for img_idx in range(5):
        # Original image
        orig_img = sample_images[img_idx].squeeze().numpy()
        axes[img_idx].imshow(orig_img, cmap='gray')
        # Handle both tensor and int labels
        orig_label_val = sample_labels[img_idx].item() if hasattr(sample_labels[img_idx], 'item') else sample_labels[img_idx]
        axes[img_idx].set_title(f'Original (label={orig_label_val})')
        axes[img_idx].axis('off')
        
        # Translated image with size 18
        if 18 in translated_datasets:
            trans_idx = img_idx % len(translated_datasets[18])
            trans_img, trans_label = translated_datasets[18][trans_idx]
            trans_img_np = trans_img.squeeze().numpy()
            axes[img_idx + 5].imshow(trans_img_np, cmap='gray')
            # Handle both tensor and int labels
            label_val = trans_label.item() if hasattr(trans_label, 'item') else trans_label
            axes[img_idx + 5].set_title(f'Translated 18 (label={label_val})')
            axes[img_idx + 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/translation_diagnostic.png', dpi=150)
    print(f'Saved translation diagnostic to {output_dir}/translation_diagnostic.png')
    
    # Print diagnostic information
    print('\n=== Translation Diagnostic ===')
    print('Note: CNNs are only approximately translation-invariant.')
    print('The classifier was trained WITHOUT translation augmentation,')
    print('so it may not generalize well to translated images.')
    print(f'\nTranslation sizes tested: {translation_sizes}')
    print('With L∞ norm, translation_size=18 means max(|tx|, |ty|) = 18,')
    print('which can push digits to the very edge of the 64x64 image.')
    print('\nAccuracy by translation size:')
    for t_size in sorted(translation_sizes):
        print(f'  Size {t_size:2d}: {translation_accuracies[t_size]:.4f}')

    # Save top-k flagged images (highest |r|)
    top_images_dir = f'{output_dir}/top_images'
    os.makedirs(top_images_dir, exist_ok=True)
    
    def save_topk_images(ds, residuals, k=10, prefix='top'):
        # ds: dataset (indexable)
        idx_sorted = np.argsort(-np.abs(residuals))[:k]
        for i, idx in enumerate(idx_sorted):
            x,y = ds[idx]
            # x is tensor (1,H,W) - should be (1, 64, 64) after padding
            assert x.shape[1] == 64 and x.shape[2] == 64, f'Expected 64x64 image, got {x.shape[1]}x{x.shape[2]}'
            img_data = x.squeeze().numpy()  # (H, W) shape, should be (64, 64)
            H, W = img_data.shape
            assert H == 64 and W == 64, f'Expected 64x64 after squeeze, got {H}x{W}'
            # Convert to uint8 in range [0, 255] for PIL
            # img_data is in range [0, 1] from ToTensor()
            img_uint8 = (img_data * 255).astype(np.uint8)
            # Save directly using PIL to guarantee exact pixel dimensions
            img_pil = Image.fromarray(img_uint8, mode='L')  # 'L' mode for grayscale
            assert img_pil.size == (64, 64), f'PIL image size is {img_pil.size}, expected (64, 64)'
            # Metadata is in filename: {prefix}_{i}_idx{idx}_r{residual}_label{y}.png
            img_pil.save(f'{top_images_dir}/{prefix}_{i}_idx{idx}_r{residuals[idx]:.4f}_label{y}.png')

    # Save top-k images for original, and selected translated/rotated datasets
    save_topk_images(test_set, r_orig, k=10, prefix='orig')
    for t_size in [0, 6, 12, 18]:
        save_topk_images(translated_datasets[t_size], r_trans_dict[t_size], k=10, prefix=f'trans_{t_size}')
    for angle in [-90, -36, 0, 36, 90]:
        if angle in rotated_datasets:
            save_topk_images(rotated_datasets[angle], r_rot_dict[angle], k=10, prefix=f'rot_{angle}')
    print(f'Saved top-k images to {top_images_dir}/')

    # Save all Stein-related data to disk
    # Save residual arrays
    np.save(f'{output_dir}/resid_orig.npy', r_orig)
    for t_size in translation_sizes:
        np.save(f'{output_dir}/resid_trans_{t_size}.npy', r_trans_dict[t_size])
    for angle in rotation_angles:
        np.save(f'{output_dir}/resid_rot_{angle}.npy', r_rot_dict[angle])
    
    # Save per-dimension residuals if computed
    if args.use_per_dim_stein:
        np.save(f'{output_dir}/resid_per_dim_orig.npy', r_orig_per_dim)
        for t_size in translation_sizes:
            np.save(f'{output_dir}/resid_per_dim_trans_{t_size}.npy', r_trans_per_dim_dict[t_size])
        for angle in rotation_angles:
            np.save(f'{output_dir}/resid_per_dim_rot_{angle}.npy', r_rot_per_dim_dict[angle])
    
    # Save function values (classifier confidence)
    np.save(f'{output_dir}/fvals_orig.npy', f_orig)
    for t_size in translation_sizes:
        np.save(f'{output_dir}/fvals_trans_{t_size}.npy', f_trans_dict[t_size])
    for angle in rotation_angles:
        np.save(f'{output_dir}/fvals_rot_{angle}.npy', f_rot_dict[angle])
    
    # Save labels
    np.save(f'{output_dir}/labels_orig.npy', y_orig)
    for t_size in translation_sizes:
        np.save(f'{output_dir}/labels_trans_{t_size}.npy', y_trans_dict[t_size])
    for angle in rotation_angles:
        np.save(f'{output_dir}/labels_rot_{angle}.npy', y_rot_dict[angle])
    
    # Save score model outputs (||s(x)||^2 and full score vectors)
    if score_model is not None:
        np.save(f'{output_dir}/score_norm_orig.npy', score_norm_orig)
        np.save(f'{output_dir}/score_output_orig.npy', score_out_orig)
        for t_size in translation_sizes:
            np.save(f'{output_dir}/score_norm_trans_{t_size}.npy', score_norm_trans_dict[t_size])
            np.save(f'{output_dir}/score_output_trans_{t_size}.npy', score_out_trans_dict[t_size])
        for angle in rotation_angles:
            np.save(f'{output_dir}/score_norm_rot_{angle}.npy', score_norm_rot_dict[angle])
            np.save(f'{output_dir}/score_output_rot_{angle}.npy', score_out_rot_dict[angle])
    
    # Save comprehensive statistics and metadata
    # Handle both scalar and per-dimension fvals
    def get_confidence(fvals):
        if fvals.ndim == 1:
            return fvals
        else:
            return fvals.max(axis=1)
    
    stats = {
        'original': {
            'count': len(r_orig),
            'loss': float(loss_orig),
            'accuracy': float(acc_orig),
            'residual_mean': float(r_orig.mean()),
            'residual_std': float(r_orig.std()),
            'residual_median': float(np.median(np.abs(r_orig))),
            'fval_mean': float(f_orig.mean()),
            'fval_std': float(f_orig.std()),
            'corr_abs_residual_confidence': float(np.corrcoef(np.abs(r_orig), (1.0 - get_confidence(f_orig)))[0,1]),
            'score_norm_mean': float(score_norm_orig.mean()) if score_model is not None else None,
            'score_norm_std': float(score_norm_orig.std()) if score_model is not None else None
        },
        'translated': {
            'translation_sizes': translation_sizes,
            'per_size': {
                str(t_size): {
                    'count': len(r_trans_dict[t_size]),
                    'loss': float(translation_losses[t_size]),
                    'accuracy': float(translation_accuracies[t_size]),
                    'residual_mean': float(r_trans_dict[t_size].mean()),
                    'residual_std': float(r_trans_dict[t_size].std()),
                    'residual_median': float(np.median(np.abs(r_trans_dict[t_size]))),
                    'fval_mean': float(f_trans_dict[t_size].mean()),
                    'fval_std': float(f_trans_dict[t_size].std()),
                    'corr_abs_residual_confidence': float(np.corrcoef(np.abs(r_trans_dict[t_size]), (1.0 - get_confidence(f_trans_dict[t_size])))[0,1]),
                    'frac_above_orig_95th_pct': float((np.abs(r_trans_dict[t_size]) > np.percentile(np.abs(r_orig), 95)).mean()),
                    'score_norm_mean': float(score_norm_trans_dict[t_size].mean()) if score_model is not None else None,
                    'score_norm_std': float(score_norm_trans_dict[t_size].std()) if score_model is not None else None
                } for t_size in translation_sizes
            }
        },
        'rotated': {
            'rotation_angles': rotation_angles,
            'per_angle': {
                str(angle): {
                    'count': len(r_rot_dict[angle]),
                    'loss': float(rotation_losses[angle]),
                    'accuracy': float(rotation_accuracies[angle]),
                    'residual_mean': float(r_rot_dict[angle].mean()),
                    'residual_std': float(r_rot_dict[angle].std()),
                    'residual_median': float(np.median(np.abs(r_rot_dict[angle]))),
                    'fval_mean': float(f_rot_dict[angle].mean()),
                    'fval_std': float(f_rot_dict[angle].std()),
                    'corr_abs_residual_confidence': float(np.corrcoef(np.abs(r_rot_dict[angle]), (1.0 - get_confidence(f_rot_dict[angle])))[0,1]),
                    'frac_above_orig_95th_pct': float((np.abs(r_rot_dict[angle]) > np.percentile(np.abs(r_orig), 95)).mean()),
                    'score_norm_mean': float(score_norm_rot_dict[angle].mean()) if score_model is not None else None,
                    'score_norm_std': float(score_norm_rot_dict[angle].std()) if score_model is not None else None
                } for angle in rotation_angles
            }
        },
        'parameters': {
            'hutchinson_probes': args.hutchinson_probes,
            'translation_sizes': translation_sizes,
            'rotation_angles': rotation_angles,
            'batch_size': args.batch_size,
            'classifier_ckpt': classifier_ckpt_path,
            'score_ckpt': score_ckpt_path if score_model is not None else None,
            'device': str(device),
            'use_per_dim_stein': args.use_per_dim_stein,
            'stein_aggregation': args.stein_aggregation if args.use_per_dim_stein else None
        }
    }
    
    with open(f'{output_dir}/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f'\nSaved all Stein results to {output_dir}/')
    print(f'  - Residuals: resid_*.npy (for original, translated, and rotated)')
    print(f'  - Function values: fvals_*.npy')
    print(f'  - Labels: labels_*.npy')
    if score_model is not None:
        print(f'  - Score outputs: score_norm_*.npy and score_output_*.npy (||s(x)||^2 and full vectors)')
    print(f'  - Statistics: statistics.json')
    print(f'  - Histograms: stein_residuals_hist.png')
    print(f'  - Top images: top_images/')

# ---------------------------
# Argument parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Stein shift detection pipeline for MNIST')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training and evaluation (default: 128)')
    parser.add_argument('--classifier-epochs', type=int, default=5,
                       help='Number of epochs to train classifier (default: 5)')
    parser.add_argument('--classifier-ckpt', type=str, default=None,
                       help='Path to classifier checkpoint file. Default: checkpoints/classifier.pth')
    parser.add_argument('--cached-classifier', action='store_true',
                       help='Load classifier from checkpoint. Raises error if checkpoint does not exist.')
    parser.add_argument('--score-model-type', type=str, default='unet', choices=['small', 'unet'],
                       help='Type of score model to use: "small" for SmallScoreNet (simple DSM), "unet" for UNetScore (annealed DSM). Default: unet')
    parser.add_argument('--score-epochs', type=int, default=5,
                       help='Number of epochs to train score model (default: 5)')
    parser.add_argument('--score-ckpt', type=str, default=None,
                       help='Path to score model checkpoint file. Default: checkpoints/score_unet.pth or checkpoints/score.pth based on model type')
    parser.add_argument('--cached-score', action='store_true',
                       help='Load score model from checkpoint. Raises error if checkpoint does not exist.')
    parser.add_argument('--hutchinson-probes', type=int, default=1,
                       help='Number of Hutchinson probes for Laplacian estimate (1-4 typical, default: 1)')
    parser.add_argument('--translate-pixels', type=int, default=4,
                       help='Maximum pixel translation for translated test set (default: 4)')
    parser.add_argument('--use-per-dim-stein', action='store_true',
                       help='Use per-dimension Stein residual computation (computes residuals for each output class, then aggregates)')
    parser.add_argument('--stein-aggregation', type=str, default='sum', choices=['sum', 'l2'],
                       help='Aggregation method for per-dimension Stein residuals: "sum" (default) or "l2" (L2 norm). Only used with --use-per-dim-stein')
    parser.add_argument('--stein-operator', type=str, default='laplacian', choices=['laplacian', 'alternative'],
                       help='Stein operator to use: "laplacian" (default) or "alternative" (f(x)*grad log p(x) + grad f(x)). Only used with regular (not per-dim) Stein computation.')
    
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
    if args.translate_pixels < 0:
        parser.error('--translate-pixels must be non-negative')
    
    return args

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == '__main__':
    args = parse_args()
    evaluate_pipeline(args)
