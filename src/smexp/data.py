from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SLTFeatureDataset(Dataset):
    def __init__(self, manifest_csv, feature_dir, split):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        if self.df.empty:
            raise RuntimeError(f'Empty dataset for split={split}')
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = self.feature_dir / f"{r['video_stem']}.npz"
        d = np.load(p, allow_pickle=True)
        item = {
            'face': torch.tensor(d['face'], dtype=torch.float32),
            'left_hand': torch.tensor(d['left_hand'], dtype=torch.float32),
            'right_hand': torch.tensor(d['right_hand'], dtype=torch.float32),
            'pose14': torch.tensor(d['pose14'], dtype=torch.float32),
            'confidence': torch.tensor(d['confidence'], dtype=torch.float32),
            'text': str(r['text']),
            'video_stem': str(r['video_stem']),
        }
        return item

def collate_features(batch):
    max_t = max(x['face'].shape[0] for x in batch)
    def pad(key):
        dim = batch[0][key].shape[-1]
        out = torch.zeros(len(batch), max_t, dim, dtype=torch.float32)
        for i, b in enumerate(batch):
            t = b[key].shape[0]
            out[i, :t] = b[key]
        return out
    mask = torch.ones(len(batch), max_t, dtype=torch.bool)
    for i, b in enumerate(batch):
        mask[i, :b['face'].shape[0]] = False
    return {
        'face': pad('face'),
        'left_hand': pad('left_hand'),
        'right_hand': pad('right_hand'),
        'pose14': pad('pose14'),
        'confidence': pad('confidence'),
        'src_key_padding_mask': mask,
        'text': [b['text'] for b in batch],
        'video_stem': [b['video_stem'] for b in batch],
    }
