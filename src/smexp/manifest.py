from pathlib import Path
import re
import pandas as pd


VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


def read_csv_robust(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    attempts = [
        dict(sep=",", engine="python", encoding="utf-8"),
        dict(sep="\t", engine="python", encoding="utf-8"),
        dict(sep=";", engine="python", encoding="utf-8"),
        dict(sep=None, engine="python", encoding="utf-8"),
        dict(sep=",", engine="python", encoding="utf-8-sig"),
        dict(sep="\t", engine="python", encoding="utf-8-sig"),
        dict(sep=None, engine="python", encoding="utf-8-sig"),
    ]

    last_err = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e

    # fallback: bỏ dòng lỗi
    try:
        df = pd.read_csv(
            csv_path,
            sep=None,
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip"
        )
        if df.shape[1] >= 2:
            return df
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Cannot read CSV: {csv_path}\nLast error: {last_err}")


def normalize_stem(x):
    x = str(x).strip()

    for ext in VIDEO_EXTS:
        if x.lower().endswith(ext):
            x = x[: -len(ext)]

    return x


def make_video_keys_from_stem(stem):
    """
    Video của bạn:
        _fZbAxSSbX4_0-5-rgb_front

    Tạo ra nhiều key để match với CSV:
        _fZbAxSSbX4_0-5-rgb_front
        _fZbAxSSbX4_0-5
        _fZbAxSSbX4_0
        _fZbAxSSbX4
    """
    stem = normalize_stem(stem)
    keys = set()

    keys.add(stem)

    # Bỏ hậu tố camera
    no_cam = (
        stem.replace("-rgb_front", "")
            .replace("_rgb_front", "")
            .replace("-rgb", "")
            .replace("_rgb", "")
    )
    keys.add(no_cam)

    # Match dạng: _fZbAxSSbX4_0-5-rgb_front
    m = re.match(r"(.+?)_(\d+)-(\d+)(?:-|_)?rgb_front$", stem)
    if m:
        vid, start, end = m.groups()
        keys.add(vid)
        keys.add(f"{vid}_{start}")
        keys.add(f"{vid}_{start}-{end}")
        keys.add(f"{vid}-{start}-{end}")

    # Match dạng sau khi bỏ camera: _fZbAxSSbX4_0-5
    m = re.match(r"(.+?)_(\d+)-(\d+)$", no_cam)
    if m:
        vid, start, end = m.groups()
        keys.add(vid)
        keys.add(f"{vid}_{start}")
        keys.add(f"{vid}_{start}-{end}")
        keys.add(f"{vid}-{start}-{end}")

    return {k for k in keys if k and k != "nan"}


def load_annotations(csv_path):
    df = read_csv_robust(csv_path)

    df.columns = [str(c).strip() for c in df.columns]

    print(f"\n[CSV] {csv_path}")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    video_candidates = [
        # How2Sign clip-level name, phải ưu tiên cột này
        "SENTENCE_NAME", "sentence_name", "Sentence_Name",

        # fallback
        "VIDEO_NAME", "video_name", "Video_Name",
        "SENTENCE_ID", "sentence_id", "Sentence_ID",
        "VIDEO_ID", "video_id", "Video_ID",
        "name", "filename", "file", "video"
    ]

    text_candidates = [
        "SENTENCE", "sentence", "Sentence",
        "TEXT", "text", "Text",
        "translation", "Translation",
        "caption", "Caption"
    ]

    video_col = None
    text_col = None

    for c in video_candidates:
        if c in df.columns:
            video_col = c
            break

    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break

    if video_col is None:
        video_col = df.columns[0]

    if text_col is None:
        text_col = df.columns[-1]

    print("Using video column:", video_col)
    print("Using text column :", text_col)

    out = df[[video_col, text_col]].copy()
    out.columns = ["video_key", "text"]

    out["video_key"] = out["video_key"].astype(str).map(normalize_stem)
    out["text"] = out["text"].astype(str).str.strip()

    out = out[out["video_key"].notna()]
    out = out[out["text"].notna()]
    out = out[out["video_key"].str.len() > 0]
    out = out[out["text"].str.len() > 0]

    # tạo annotation map nhiều key
    rows = []

    for _, r in out.iterrows():
        key = r["video_key"]
        text = r["text"]

        keys = set()
        keys.add(key)

        no_cam = (
            key.replace("-rgb_front", "")
               .replace("_rgb_front", "")
               .replace("-rgb", "")
               .replace("_rgb", "")
        )
        keys.add(no_cam)

        # Nếu CSV có VIDEO_NAME dạng _fZbAxSSbX4_0-5-rgb_front
        m = re.match(r"(.+?)_(\d+)-(\d+)(?:-|_)?rgb_front$", key)
        if m:
            vid, start, end = m.groups()
            keys.add(vid)
            keys.add(f"{vid}_{start}")
            keys.add(f"{vid}_{start}-{end}")
            keys.add(f"{vid}-{start}-{end}")

        # Nếu CSV có VIDEO_NAME dạng _fZbAxSSbX4_0-5
        m = re.match(r"(.+?)_(\d+)-(\d+)$", no_cam)
        if m:
            vid, start, end = m.groups()
            keys.add(vid)
            keys.add(f"{vid}_{start}")
            keys.add(f"{vid}_{start}-{end}")
            keys.add(f"{vid}-{start}-{end}")

        for k in keys:
            rows.append({"match_key": k, "text": text})

    ann = pd.DataFrame(rows).drop_duplicates(subset=["match_key"])
    print("Annotation match keys:", len(ann))

    return ann


def scan_videos(video_dir):
    video_dir = Path(video_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video dir not found: {video_dir}")

    files = []
    for ext in VIDEO_EXTS:
        files.extend(video_dir.rglob(f"*{ext}"))

    files = sorted(files)

    rows = []
    for p in files:
        stem = p.stem
        rows.append({
            "video_path": str(p),
            "video_stem": stem,
            "candidate_keys": list(make_video_keys_from_stem(stem))
        })

    return pd.DataFrame(rows)


def build_split(video_dir, csv_path, split_name=None, *args, **kwargs):
    """
    Bản flexible để tương thích với nhiều cách gọi khác nhau từ 01_prepare_manifest.py.

    Hỗ trợ các kiểu gọi:
        build_split(video_dir, csv_path, split, fraction, min_items, seed)
        build_split(video_dir, csv_path, split, fraction=..., min_items=..., seed=...)
        build_split(video_dir, csv_path, split='train', fraction=..., min_items=..., seed=...)
    """

    # Lấy split
    split = kwargs.pop("split", None)
    if split is None:
        split = kwargs.pop("split_name", None)
    if split is None:
        split = split_name
    if split is None:
        split = "unknown"

    # Lấy fraction
    fraction = kwargs.pop("fraction", None)
    if fraction is None:
        fraction = kwargs.pop("fraction_per_split", None)
    if fraction is None and len(args) >= 1:
        fraction = args[0]
    if fraction is None:
        fraction = 1.0

    # Lấy min_items
    min_items = kwargs.pop("min_items", None)
    if min_items is None:
        min_items = kwargs.pop("min_items_per_split", None)
    if min_items is None and len(args) >= 2:
        min_items = args[1]
    if min_items is None:
        min_items = 1

    # Lấy seed
    seed = kwargs.pop("seed", None)
    if seed is None and len(args) >= 3:
        seed = args[2]
    if seed is None:
        seed = 42

    fraction = float(fraction)
    min_items = int(min_items)
    seed = int(seed)

    videos = scan_videos(video_dir)
    ann = load_annotations(csv_path)

    if videos.empty:
        raise RuntimeError(f"No videos found in {video_dir}")

    ann_map = dict(zip(ann["match_key"], ann["text"]))

    matched_rows = []

    for _, r in videos.iterrows():
        text = None

        for k in r["candidate_keys"]:
            if k in ann_map:
                text = ann_map[k]
                break

        if text is not None:
            matched_rows.append({
                "video_path": r["video_path"],
                "video_stem": r["video_stem"],
                "text": text,
                "split": split
            })

    df = pd.DataFrame(matched_rows)

    print(f"\n[{split}]")
    print("Video dir:", video_dir)
    print("Total videos:", len(videos))
    print("Matched videos:", len(df))

    if len(df) == 0:
        print("\nKhông match được video nào.")
        print("Ví dụ video stem:")
        print(videos["video_stem"].head(10).tolist())
        print("Ví dụ annotation keys:")
        print(ann["match_key"].head(20).tolist())
        raise RuntimeError(f"No matched videos for split={split}")

    # Sample fraction mỗi split
    if fraction is not None and fraction < 1.0:
        n = max(int(len(df) * fraction), min_items)
        n = min(n, len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        print(f"Sampled {fraction * 100:.2f}% -> {len(df)} videos")
    else:
        df = df.reset_index(drop=True)
        print(f"Using full split -> {len(df)} videos")

    return df