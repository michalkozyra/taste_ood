"""
Adversarial example generation for OOD detection benchmarking.

This module provides utilities for generating adversarial examples using RobustBench
and integrating them into the OOD detection evaluation pipeline.
"""

from .adversarial_dataset import AdversarialDataset, generate_adversarial_dataset
from .robustbench_attacks import create_attack, AttackConfig
from .attack_config import get_attack_preset, parse_adversarial_dataset_name

__all__ = [
    'AdversarialDataset',
    'generate_adversarial_dataset',
    'create_attack',
    'AttackConfig',
    'get_attack_preset',
    'parse_adversarial_dataset_name',
]
