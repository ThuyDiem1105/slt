import argparse
from pathlib import Path
from collections import Counter
from smexp.config import load_config
from smexp.video import list_videos

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
args = parser.parse_args()
cfg = load_config(args.config)
exts = cfg['prepare']['video_extensions']
for split in ['train','val','test']:
    vdir = Path(cfg['data'][f'{split}_video_dir'])
    csv = Path(cfg['data'][f'{split}_csv'])
    print(f'\n[{split}]')
    print('video_dir:', vdir.resolve())
    print('csv:', csv.resolve())
    print('video_dir exists:', vdir.exists())
    print('csv exists:', csv.exists())
    if vdir.exists():
        files = list_videos(vdir, exts)
        print('num videos:', len(files))
        print('extensions:', Counter(p.suffix.lower() for p in files))
        for p in files[:5]:
            print(' -', p)
