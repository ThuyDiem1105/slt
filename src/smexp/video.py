import cv2
import numpy as np
from pathlib import Path


def list_videos(root, extensions):
    root = Path(root)
    exts = {e.lower() for e in extensions}
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in exts])


def read_sampled_frames(video_path, max_frames=64, stride=2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f'No frames read: {video_path}')
    return frames
