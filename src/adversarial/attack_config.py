"""
Attack configuration and presets for adversarial example generation.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class AttackConfig:
    """Configuration for adversarial attack generation."""
    attack_type: str  # 'autoattack', 'pgd', 'fgsm', 'apgd', 'square'
    threat_model: str  # 'linf', 'l2', 'l0'
    epsilon: float  # Perturbation budget
    # Attack-specific parameters
    steps: Optional[int] = None  # For PGD, APGD
    step_size: Optional[float] = None  # For PGD, APGD, FGSM
    restarts: Optional[int] = None  # For PGD
    n_queries: Optional[int] = None  # For Square attack
    # Additional options
    verbose: bool = True
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.threat_model not in ['linf', 'l2', 'l0']:
            raise ValueError(f"Unknown threat model: {self.threat_model}")
        if self.attack_type not in ['autoattack', 'pgd', 'fgsm', 'apgd', 'square']:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        if self.epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {self.epsilon}")


# Attack presets
ATTACK_PRESETS: Dict[str, AttackConfig] = {
    # AutoAttack presets (L∞)
    'autoattack_linf_8': AttackConfig(
        attack_type='autoattack',
        threat_model='linf',
        epsilon=8/255,
    ),
    'autoattack_linf_4': AttackConfig(
        attack_type='autoattack',
        threat_model='linf',
        epsilon=4/255,
    ),
    'autoattack_linf_2': AttackConfig(
        attack_type='autoattack',
        threat_model='linf',
        epsilon=2/255,
    ),
    'autoattack_linf_16': AttackConfig(
        attack_type='autoattack',
        threat_model='linf',
        epsilon=16/255,
    ),
    
    # PGD presets (L∞)
    'pgd_linf_8': AttackConfig(
        attack_type='pgd',
        threat_model='linf',
        epsilon=8/255,
        steps=50,
        step_size=2/255,
        restarts=1,
    ),
    'pgd_linf_4': AttackConfig(
        attack_type='pgd',
        threat_model='linf',
        epsilon=4/255,
        steps=50,
        step_size=1/255,
        restarts=1,
    ),
    'pgd_linf_2': AttackConfig(
        attack_type='pgd',
        threat_model='linf',
        epsilon=2/255,
        steps=50,
        step_size=0.5/255,
        restarts=1,
    ),
    
    # FGSM presets (L∞)
    'fgsm_linf_8': AttackConfig(
        attack_type='fgsm',
        threat_model='linf',
        epsilon=8/255,
    ),
    'fgsm_linf_4': AttackConfig(
        attack_type='fgsm',
        threat_model='linf',
        epsilon=4/255,
    ),
    
    # PGD presets (L2)
    'pgd_l2_0.5': AttackConfig(
        attack_type='pgd',
        threat_model='l2',
        epsilon=0.5,
        steps=50,
        step_size=0.1,
        restarts=1,
    ),
    'pgd_l2_1.0': AttackConfig(
        attack_type='pgd',
        threat_model='l2',
        epsilon=1.0,
        steps=50,
        step_size=0.2,
        restarts=1,
    ),
}


def get_attack_preset(preset_name: str) -> AttackConfig:
    """Get a predefined attack configuration."""
    if preset_name not in ATTACK_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {list(ATTACK_PRESETS.keys())}"
        )
    return ATTACK_PRESETS[preset_name]


def parse_adversarial_dataset_name(dataset_name: str) -> Optional[AttackConfig]:
    """
    Parse adversarial dataset name to extract attack configuration.
    
    Format: adversarial:{attack_type}:{threat_model}:{epsilon}
    Examples:
        - adversarial:autoattack:linf:8/255
        - adversarial:pgd:linf:4/255
        - adversarial:fgsm:linf:2/255
        - adversarial:pgd:l2:0.5
    
    Extended format (for custom parameters):
        - adversarial:pgd:linf:8/255:steps=50:restarts=1
    
    Args:
        dataset_name: Dataset name string
    
    Returns:
        AttackConfig if name matches adversarial pattern, None otherwise
    """
    if not dataset_name.startswith('adversarial:'):
        return None
    
    # Remove 'adversarial:' prefix
    config_str = dataset_name[len('adversarial:'):]
    
    # Parse basic format: attack:norm:epsilon
    parts = config_str.split(':')
    if len(parts) < 3:
        raise ValueError(
            f"Invalid adversarial dataset name format: {dataset_name}. "
            f"Expected: adversarial:{{attack}}:{{norm}}:{{epsilon}}"
        )
    
    attack_type = parts[0].lower()
    threat_model = parts[1].lower()
    epsilon_str = parts[2]
    
    # Parse epsilon (handle fractions like "8/255")
    if '/' in epsilon_str:
        num, den = epsilon_str.split('/')
        epsilon = float(num) / float(den)
    else:
        epsilon = float(epsilon_str)
    
    # Parse additional parameters if present
    kwargs = {}
    for part in parts[3:]:
        if '=' in part:
            key, value = part.split('=')
            if key == 'steps':
                kwargs['steps'] = int(value)
            elif key == 'step_size':
                kwargs['step_size'] = float(value)
            elif key == 'restarts':
                kwargs['restarts'] = int(value)
            elif key == 'n_queries':
                kwargs['n_queries'] = int(value)
            elif key == 'seed':
                kwargs['seed'] = int(value)
    
    # Set defaults based on attack type
    if attack_type == 'pgd' and 'steps' not in kwargs:
        kwargs['steps'] = 50
    if attack_type == 'pgd' and 'step_size' not in kwargs:
        # Default step size: epsilon / steps (for linf) or epsilon / 10 (for l2)
        if threat_model == 'linf':
            kwargs['step_size'] = epsilon / kwargs.get('steps', 50)
        else:
            kwargs['step_size'] = epsilon / 10
    if attack_type == 'pgd' and 'restarts' not in kwargs:
        kwargs['restarts'] = 1
    
    return AttackConfig(
        attack_type=attack_type,
        threat_model=threat_model,
        epsilon=epsilon,
        **kwargs
    )
