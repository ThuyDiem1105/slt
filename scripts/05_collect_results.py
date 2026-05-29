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

    for key in [
        "face",
        "left_hand",
        "right_hand",
        "pose14",
        "confidence",
        "src_key_padding_mask",
    ]:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(dev)

    out["attention_mask"] = (~out["src_key_padding_mask"]).long()
    return out


def build_model(cfg, mode, t5_name):
    dropout = cfg["model"].get("dropout", 0.1)

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

        raise RuntimeError(
            "Không tìm thấy PaperLikeSignMusketeersT5 hoặc SignMusketeersT5 trong src/smexp/model.py"
        )

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


def evaluate_one_split(model, tokenizer, cfg, split, dev, out_pred_csv):
    ds = SLTFeatureDataset(
        cfg["paths"]["manifest_csv"],
        cfg["paths"]["feature_dir"],
        split,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_features,
    )

    preds, refs, stems = [], [], []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {split}", leave=False):
            batch = batch_to_device(batch, dev)
            out = generate_model(model, batch, cfg)
            pred_texts = tokenizer.batch_decode(out, skip_special_tokens=True)

            preds.extend(pred_texts)
            refs.extend(batch["text"])
            stems.extend(batch["video_stem"])

    metrics = compute_metrics(preds, refs)

    out_df = pd.DataFrame(
        {
            "video_stem": stems,
            "reference": refs,
            "prediction": preds,
        }
    )

    out_pred_csv = Path(out_pred_csv)
    out_pred_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")

    metrics_path = out_pred_csv.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics, out_pred_csv, metrics_path, len(out_df)


def find_checkpoints(checkpoint_dir, requested_runs=None):
    checkpoint_dir = Path(checkpoint_dir)

    if requested_runs:
        ckpts = []
        for run in requested_runs:
            ckpt = checkpoint_dir / run / "best.pt"
            if ckpt.exists():
                ckpts.append(ckpt)
            else:
                print(f"[WARNING] Không tìm thấy checkpoint: {ckpt}")
        return ckpts

    return sorted(checkpoint_dir.glob("*/best.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Tên các run muốn gom kết quả. Ví dụ: --runs paper_like_pretrained_dinov2 ca_csa_full",
    )
    parser.add_argument(
        "--summary_csv",
        default="outputs/summary_all_splits.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev = device()

    checkpoint_dir = args.checkpoint_dir or cfg["paths"]["checkpoint_dir"]
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ckpts = find_checkpoints(checkpoint_dir, args.runs)

    if not ckpts:
        raise SystemExit(
            f"Không tìm thấy checkpoint nào trong {checkpoint_dir}. "
            f"Hãy train model trước hoặc truyền --runs đúng tên."
        )

    print("=" * 70)
    print("Collect Results: train / val / test")
    print("=" * 70)
    print("Device:", dev)
    print("Checkpoint dir:", checkpoint_dir)
    print("Splits:", args.splits)
    print("Found checkpoints:")
    for ckpt in ckpts:
        print(" -", ckpt)
    print("=" * 70)

    summary_rows = []

    for ckpt_path in ckpts:
        run_name = ckpt_path.parent.name

        print("\n" + "=" * 70)
        print(f"Loading checkpoint: {run_name}")
        print("=" * 70)

        ckpt = torch.load(ckpt_path, map_location=dev)

        mode = ckpt.get("mode", "paper")
        t5_name = ckpt.get("t5_model", cfg["model"]["t5_model"])
        best_bleu = ckpt.get("best_bleu", None)
        epoch = ckpt.get("epoch", None)

        print("Run:", run_name)
        print("Mode:", mode)
        print("T5:", t5_name)
        print("Best val BLEU in ckpt:", best_bleu)
        print("Epoch:", epoch)

        tok_path = ckpt_path.parent / "tokenizer"
        if tok_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(t5_name)

        model = build_model(cfg, mode, t5_name).to(dev)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        for split in args.splits:
            out_pred_csv = outputs_dir / f"pred_{run_name}_{split}.csv"

            print(f"\nEvaluating {run_name} on {split}...")
            metrics, pred_csv, metrics_json, n_samples = evaluate_one_split(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                split=split,
                dev=dev,
                out_pred_csv=out_pred_csv,
            )

            print(f"{run_name} | {split} | samples={n_samples} | metrics={metrics}")

            row = {
                "run_name": run_name,
                "mode": mode,
                "split": split,
                "num_samples": n_samples,
                "checkpoint": str(ckpt_path),
                "prediction_csv": str(pred_csv),
                "metrics_json": str(metrics_json),
                "t5_model": t5_name,
                "best_val_bleu_in_checkpoint": best_bleu,
                "best_epoch": epoch,
            }

            for k, v in metrics.items():
                row[k] = v

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("DONE")
    print("Saved summary:", summary_csv)
    print("=" * 70)
    print(summary_df)


if __name__ == "__main__":
    main()