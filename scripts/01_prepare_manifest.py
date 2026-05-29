import argparse
from pathlib import Path
import pandas as pd
from smexp.config import load_config, ensure_dirs, set_seed
from smexp.manifest import build_split

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
args = parser.parse_args()
cfg = load_config(args.config)
ensure_dirs(cfg)
set_seed(cfg['project']['seed'])
frac = cfg['prepare']['fraction_per_split']
min_n = cfg['prepare'].get('min_per_split', 1)
max_n = cfg['prepare'].get('max_per_split', None)
exts = cfg['prepare']['video_extensions']
all_df = []
for split in ['train','val','test']:
    df = build_split(
        cfg['data'][f'{split}_video_dir'],
        cfg['data'][f'{split}_csv'],
        split,
        exts,
        fraction=frac,
        min_n=min_n,
        max_n=max_n,
        seed=cfg['project']['seed'] + {'train':0,'val':1,'test':2}[split]
    )
    all_df.append(df)
    print(split, 'sampled:', len(df))
out = pd.concat(all_df, ignore_index=True)
out_path = Path(cfg['paths']['manifest_csv'])
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print('\nSaved manifest:', out_path)
print(out['split'].value_counts())
print(out.head())
