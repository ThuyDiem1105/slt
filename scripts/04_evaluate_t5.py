import argparse
import inspect
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from smexp.config import load_config, device
from smexp.data import SLTFeatureDataset, collate_features
from smexp.metrics import compute_metrics

# Import linh hoạt để tương thích với cả code cũ và code mới
try:
    from smexp.model import CA_CSAT5
except ImportError:
    CA_CSAT5 = None

try:
    from smexp.model import PaperLikeSignMusketeersT5
except ImportError:
    PaperLikeSignMusketeersT5 = None

try:
    from smexp.model import SignMusketeersT5
except ImportError:
    SignMusketeersT5 = None


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid}


def batch_to_device(batch, dev):
    out = dict(batch)
    for key in ["face", "left_hand", "right_hand", "pose14", "confidence", "src_key_padding_mask"]:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(dev)

    out["attention_mask"] = (~out["src_key_padding_mask"]).long()
    return out


def build_model(cfg, mode, t5_name):
    dropout = cfg["model"].get("dropout", 0.1)

    # =========================
    # 1. Paper-like baseline
    # =========================
    if mode == "paper":
        if PaperLikeSignMusketeersT5 is not None:
            kwargs = {
                "t5_name": t5_name,
                "face_proj_dim": cfg["model"].get("face_proj_dim", 256),
                "hand_proj_dim": cfg["model"].get("hand_proj_dim", 256),
                "pose_proj_dim": cfg["model"].get("pose_proj_dim", 128),
                "dropout": dropout,
            }
            kwargs = filter_kwargs(PaperLikeSignMusketeersT5, kwargs)
            return PaperLikeSignMusketeersT5(**kwargs)

        if SignMusketeersT5 is not None:
            return SignMusketeersT5(
                t5_name,
                cfg["model"].get("face_proj_dim", 256),
                cfg["model"].get("hand_proj_dim", 256),
                cfg["model"].get("pose_proj_dim", 128),
                dropout,
                use_confidence=False,
            )

        raise RuntimeError("Không tìm thấy PaperLikeSignMusketeersT5 hoặc SignMusketeersT5 trong src/smexp/model.py")

    # =========================
    # 2. CA-CSA variants
    # =========================
    if CA_CSAT5 is None:
        raise RuntimeError(
            "Checkpoint là mode CA-CSA nhưng src/smexp/model.py chưa có class CA_CSAT5."
        )

    ca_cfg = cfg.get("ca_csa", {})

    use_learned_conf = True
    use_temporal_smoothing = True
    use_confidence_bias = True

    if mode == "confidence":
        mode = "ca_csa_full"

    if mode == "ca_csa_full":
        use_learned_conf = True
        use_temporal_smoothing = True
        use_confidence_bias = True

    elif mode == "ca_csa_learned":
        use_learned_conf = True
        use_temporal_smoothing = False
        use_confidence_bias = True

    elif mode == "ca_csa_mp":
        use_learned_conf = False
        use_temporal_smoothing = False
        use_confidence_bias = True

    elif mode == "csa_only":
        use_learned_conf = False
        use_temporal_smoothing = False
        use_confidence_bias = False

    else:
        raise ValueError(f"Unknown mode: {mode}")

    kwargs = {
        "t5_name": t5_name,
        "stream_dim": ca_cfg.get("stream_dim", 256),
        "num_heads": ca_cfg.get("num_heads", 4),
        "num_layers": ca_cfg.get("num_layers", 2),
        "dropout": ca_cfg.get("dropout", dropout),
        "alpha": ca_cfg.get("alpha", 0.5),
        "use_learned_conf": use_learned_conf,
        "use_temporal_smoothing": use_temporal_smoothing,
        "use_confidence_bias": use_confidence_bias,
    }

    kwargs = filter_kwargs(CA_CSAT5, kwargs)
    return CA_CSAT5(**kwargs)


@torch.no_grad()
def generate_model(model, batch, cfg):
    max_length = cfg["model"].get("max_target_len", 128)
    num_beams = cfg["model"].get("num_beams", 5)

    try:
        return model.generate(
            batch,
            max_length=max_length,
            num_beams=num_beams,
        )
    except TypeError:
        return model.generate(
            batch["face"],
            batch["left_hand"],
            batch["right_hand"],
            batch["pose14"],
            batch["confidence"],
            batch["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_csv", default="outputs/predictions.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev = device()

    print("=" * 60)
    print("Evaluate SignMusketeers / CA-CSA")
    print("=" * 60)
    print("Device:", dev)
    print("Checkpoint:", args.checkpoint)
    print("Split:", args.split)
    print("=" * 60)

    ckpt = torch.load(args.checkpoint, map_location=dev)

    mode = ckpt.get("mode", "paper")
    t5_name = ckpt.get("t5_model", cfg["model"]["t5_model"])

    print("Checkpoint mode:", mode)
    print("T5:", t5_name)

    tok_path = Path(args.checkpoint).parent / "tokenizer"

    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(t5_name)

    model = build_model(cfg, mode, t5_name).to(dev)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = SLTFeatureDataset(
        cfg["paths"]["manifest_csv"],
        cfg["paths"]["feature_dir"],
        args.split,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_features,
    )

    preds, refs, stems = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {args.split}"):
            batch = batch_to_device(batch, dev)

            out = generate_model(model, batch, cfg)

            pred_texts = tokenizer.batch_decode(out, skip_special_tokens=True)

            preds.extend(pred_texts)
            refs.extend(batch["text"])
            stems.extend(batch["video_stem"])

    metrics = compute_metrics(preds, refs)

    print("\nMetrics:")
    print(metrics)

    out_df = pd.DataFrame(
        {
            "video_stem": stems,
            "reference": refs,
            "prediction": preds,
        }
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nSaved predictions:", out_path)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()