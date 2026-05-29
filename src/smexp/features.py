import numpy as np
from PIL import Image
import torch


def _lm_array(lms, image_w, image_h):
    if lms is None:
        return None
    arr = np.array([[lm.x * image_w, lm.y * image_h, lm.z] for lm in lms.landmark], dtype=np.float32)
    return arr


def square_bbox(points, w, h, scale=1.2, fallback=None):
    if points is None or len(points) == 0 or not np.isfinite(points[:, :2]).all():
        return fallback
    x1, y1 = points[:, 0].min(), points[:, 1].min()
    x2, y2 = points[:, 0].max(), points[:, 1].max()
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = max(x2-x1, y2-y1) * scale
    side = max(side, 20)
    x1, y1 = cx - side/2, cy - side/2
    x2, y2 = cx + side/2, cy + side/2
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return fallback
    return (x1, y1, x2, y2)


def crop_or_black(frame, bbox, size=224):
    if bbox is None:
        return Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
    return Image.fromarray(crop).resize((size, size), Image.BICUBIC)


def pose14_from_mediapipe(pose_lms, w, h, prev=None):
    # Bài báo dùng 7 điểm: nose, shoulders, elbows, wrists => 14 chiều x,y normalized.
    idx = [0, 11, 12, 13, 14, 15, 16]
    arr = _lm_array(pose_lms, w, h)
    if arr is None or arr.shape[0] <= max(idx):
        if prev is not None:
            return prev, 0.0
        return np.full((14,), -1.0, dtype=np.float32), 0.0
    pts = arr[idx, :2].astype(np.float32)
    if not np.isfinite(pts).all():
        if prev is not None:
            return prev, 0.0
        return np.full((14,), -1.0, dtype=np.float32), 0.0
    left_sh, right_sh = pts[1], pts[2]
    shoulder_dist = np.linalg.norm(left_sh - right_sh)
    if shoulder_dist < 1e-3:
        shoulder_dist = max(w, h) / 6
    head_unit = shoulder_dist / 2.0
    box_w = 6.0 * head_unit
    box_h = 7.0 * head_unit
    center = (left_sh + right_sh) / 2.0
    top_left = np.array([center[0] - box_w/2, center[1] - box_h*0.35], dtype=np.float32)
    norm = np.empty_like(pts)
    norm[:, 0] = (pts[:, 0] - top_left[0]) / max(box_w, 1e-6)
    norm[:, 1] = (pts[:, 1] - top_left[1]) / max(box_h, 1e-6)
    flat = norm.reshape(-1).astype(np.float32)
    return flat, 1.0


def fallback_face_bbox_from_pose(pose_lms, w, h, scale=1.2):
    arr = _lm_array(pose_lms, w, h)
    if arr is None or arr.shape[0] < 11:
        return None
    return square_bbox(arr[0:11, :2], w, h, scale=scale)


def fallback_hand_bbox_from_pose(pose_lms, w, h, side_ref=80, hand='left'):
    arr = _lm_array(pose_lms, w, h)
    if arr is None or arr.shape[0] < 23:
        return None
    ids = [17,19,21] if hand == 'left' else [18,20,22]
    pts = arr[ids, :2]
    if not np.isfinite(pts).all():
        return None
    cx, cy = pts.mean(axis=0)
    side = max(side_ref, 40)
    x1, y1 = max(0, int(cx-side/2)), max(0, int(cy-side/2))
    x2, y2 = min(w, int(cx+side/2)), min(h, int(cy+side/2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


@torch.no_grad()
def encode_pil_batch(images, processor, model, device, batch_size=16):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors='pt').to(device)
        out = model(**inputs)
        feats.append(out.last_hidden_state[:, 0].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)
