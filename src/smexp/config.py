from pathlib import Path
import random, os
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg):
    for k in ['manifest_csv', 'feature_dir', 'checkpoint_dir', 'output_dir']:
        p = Path(cfg['paths'][k])
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
