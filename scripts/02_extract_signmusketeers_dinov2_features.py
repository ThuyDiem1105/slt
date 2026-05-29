import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from smexp.config import load_config, ensure_dirs, device
from smexp.video import read_sampled_frames
from smexp.features import (
    _lm_array, square_bbox, crop_or_black, pose14_from_mediapipe,
    fallback_face_bbox_from_pose, fallback_hand_bbox_from_pose, encode_pil_batch
)

try:
    import mediapipe as mp
    try:
        mp_holistic = mp.solutions.holistic
    except AttributeError:
        import mediapipe.python.solutions.holistic as mp_holistic
except Exception as e:
    raise RuntimeError(
        "Không import được MediaPipe Holistic. "
        "Bạn nên dùng Python 3.10 hoặc 3.11 và cài lại mediapipe."
    ) from e

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
cfg = load_config(args.config)
ensure_dirs(cfg)
dev = device()
manifest = pd.read_csv(cfg['paths']['manifest_csv'])
out_dir = Path(cfg['paths']['feature_dir'])
out_dir.mkdir(parents=True, exist_ok=True)

model_name = cfg['features']['dinov2_model']
print('Loading pretrained DINOv2:', model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
dino = AutoModel.from_pretrained(model_name).to(dev).eval()

mp_holistic = mp.solutions.holistic
crop_size = cfg['features']['crop_size']
scale = cfg['features']['bbox_scale']
bs = cfg['features']['dinov2_batch_size']

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False) as holistic:
    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        stem = row['video_stem']
        out_path = out_dir / f'{stem}.npz'
        if out_path.exists() and not args.overwrite:
            continue
        try:
            frames = read_sampled_frames(row['video_path'], cfg['features']['max_frames'], cfg['features']['frame_stride'])
            face_imgs, lh_imgs, rh_imgs = [], [], []
            pose_vecs, confs = [], []
            prev_pose = None
            last_face_box = last_lh_box = last_rh_box = None
            for fr in frames:
                h, w = fr.shape[:2]
                res = holistic.process(fr)
                face_pts = _lm_array(res.face_landmarks, w, h)
                lh_pts = _lm_array(res.left_hand_landmarks, w, h)
                rh_pts = _lm_array(res.right_hand_landmarks, w, h)

                face_box = square_bbox(face_pts[:, :2] if face_pts is not None else None, w, h, scale=scale)
                if face_box is None:
                    face_box = fallback_face_bbox_from_pose(res.pose_landmarks, w, h, scale=scale) or last_face_box
                if face_box is not None:
                    last_face_box = face_box
                face_side = (face_box[2]-face_box[0]) if face_box else 80

                lh_box = square_bbox(lh_pts[:, :2] if lh_pts is not None else None, w, h, scale=scale)
                if lh_box is None:
                    lh_box = fallback_hand_bbox_from_pose(res.pose_landmarks, w, h, side_ref=face_side, hand='left') or last_lh_box
                if lh_box is not None:
                    last_lh_box = lh_box

                rh_box = square_bbox(rh_pts[:, :2] if rh_pts is not None else None, w, h, scale=scale)
                if rh_box is None:
                    rh_box = fallback_hand_bbox_from_pose(res.pose_landmarks, w, h, side_ref=face_side, hand='right') or last_rh_box
                if rh_box is not None:
                    last_rh_box = rh_box

                pose14, c_pose = pose14_from_mediapipe(res.pose_landmarks, w, h, prev_pose)
                if c_pose > 0:
                    prev_pose = pose14

                face_imgs.append(crop_or_black(fr, face_box, crop_size))
                lh_imgs.append(crop_or_black(fr, lh_box, crop_size))
                rh_imgs.append(crop_or_black(fr, rh_box, crop_size))
                pose_vecs.append(pose14)
                confs.append([
                    1.0 if lh_pts is not None else 0.0,
                    1.0 if rh_pts is not None else 0.0,
                    1.0 if face_pts is not None else 0.0,
                    float(c_pose),
                ])

            # DINOv2 pretrained cho face/left/right crops.
            face_feat = encode_pil_batch(face_imgs, processor, dino, dev, bs)
            lh_feat = encode_pil_batch(lh_imgs, processor, dino, dev, bs)
            rh_feat = encode_pil_batch(rh_imgs, processor, dino, dev, bs)
            np.savez_compressed(
                out_path,
                face=face_feat.astype(np.float32),
                left_hand=lh_feat.astype(np.float32),
                right_hand=rh_feat.astype(np.float32),
                pose14=np.stack(pose_vecs).astype(np.float32),
                confidence=np.asarray(confs, dtype=np.float32),
                video_path=str(row['video_path']),
                text=str(row['text'])
            )
        except Exception as e:
            print('ERROR', stem, e)
