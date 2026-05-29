import argparse
import inspect
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from smexp.config import load_config, ensure_dirs, set_seed, device
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
    """
    Chỉ truyền các tham số mà class __init__ hỗ trợ.
    Giúp tránh lỗi nếu CA_CSAT5 của bạn chưa có vài tham số ablation.
    """
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid}


def batch_to_device(batch, dev):
    """
    Đưa các tensor trong batch lên GPU/CPU.
    Dataset hiện tại trả về:
        face, left_hand, right_hand, pose14, confidence, src_key_padding_mask
    """
    out = dict(batch)
    for key in ["face", "left_hand", "right_hand", "pose14", "confidence", "src_key_padding_mask"]:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(dev)

    # attention_mask: 1 = token thật, 0 = padding
    out["attention_mask"] = (~out["src_key_padding_mask"]).long()
    return out


def build_model(cfg, mode):
    """
    mode:
        paper          : baseline giống SignMusketeers paper-like, concat + linear
        confidence     : full CA-CSA
        ca_csa_full    : full CA-CSA
        ca_csa_learned : CA-CSA + learned confidence, không temporal smoothing
        ca_csa_mp      : CA-CSA + MediaPipe confidence
        csa_only       : cross-stream attention không confidence bias nếu model.py hỗ trợ
    """
    t5_name = cfg["model"]["t5_model"]
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

        # Fallback cho code cũ của bạn
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
            "Bạn đang chạy mode CA-CSA nhưng src/smexp/model.py chưa có class CA_CSAT5.\n"
            "Hãy thêm CA_CSAT5 vào model.py trước."
        )

    ca_cfg = cfg.get("ca_csa", {})

    # Default: full CA-CSA
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
        "use_learned_conf": ca_cfg.get("use_learned_conf", use_learned_conf),
        "use_temporal_smoothing": ca_cfg.get("use_temporal_smoothing", use_temporal_smoothing),
        "use_confidence_bias": ca_cfg.get("use_confidence_bias", use_confidence_bias),
    }

    # Nếu mode yêu cầu override thì ưu tiên mode
    kwargs["use_learned_conf"] = use_learned_conf
    kwargs["use_temporal_smoothing"] = use_temporal_smoothing
    kwargs["use_confidence_bias"] = use_confidence_bias

    kwargs = filter_kwargs(CA_CSAT5, kwargs)
    return CA_CSAT5(**kwargs)


def forward_model(model, batch, labels):
    """
    Tương thích với 2 kiểu model:
    1. Model mới: forward(batch, labels=labels)
    2. Model cũ: forward(face, left_hand, right_hand, pose14, confidence, attention_mask, labels)
    """
    try:
        return model(batch, labels=labels)
    except TypeError:
        return model(
            batch["face"],
            batch["left_hand"],
            batch["right_hand"],
            batch["pose14"],
            batch["confidence"],
            batch["attention_mask"],
            labels=labels,
        )


@torch.no_grad()
def generate_model(model, batch, cfg):
    """
    Tương thích với 2 kiểu generate:
    1. Model mới: generate(batch, ...)
    2. Model cũ: generate(face, left_hand, right_hand, pose14, confidence, attention_mask, ...)
    """
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


@torch.no_grad()
def run_eval(model, loader, tokenizer, cfg, dev):
    model.eval()
    preds, refs = [], []

    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = batch_to_device(batch, dev)

        out = generate_model(model, batch, cfg)

        pred_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(pred_texts)
        refs.extend(batch["text"])

    metrics = compute_metrics(preds, refs)
    return metrics, preds, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "paper",
            "confidence",
            "csa_only",
            "ca_csa_mp",
            "ca_csa_learned",
            "ca_csa_full",
        ],
        default="paper",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg["project"]["seed"])

    dev = device()

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Train SignMusketeers / CA-CSA")
    print("=" * 60)
    print("Device:", dev)
    print("Mode:", args.mode)
    print("Run name:", args.run_name)
    print("T5:", cfg["model"]["t5_model"])
    print("Manifest:", cfg["paths"]["manifest_csv"])
    print("Feature dir:", cfg["paths"]["feature_dir"])
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["t5_model"])

    model = build_model(cfg, args.mode).to(dev)

    train_ds = SLTFeatureDataset(
        cfg["paths"]["manifest_csv"],
        cfg["paths"]["feature_dir"],
        "train",
    )
    val_ds = SLTFeatureDataset(
        cfg["paths"]["manifest_csv"],
        cfg["paths"]["feature_dir"],
        "val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_features,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_features,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 0.1),
    )

    grad_accum_steps = cfg["train"].get("grad_accum_steps", 1)
    epochs = cfg["train"].get("epochs", 5)

    steps_per_epoch = max(1, len(train_loader) // max(1, grad_accum_steps))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, total_steps // 10)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_fp16 = bool(cfg["train"].get("fp16", False)) and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_bleu = -1.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for step, batch in enumerate(pbar, start=1):
            batch = batch_to_device(batch, dev)

            encoded = tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=cfg["model"].get("max_target_len", 128),
                return_tensors="pt",
            ).to(dev)

            labels = encoded.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = forward_model(model, batch, labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            total_loss += float(loss.item()) * grad_accum_steps

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["train"].get("grad_clip", 1.0),
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            avg_loss = total_loss / max(1, step)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", step=global_step)

        metrics, _, _ = run_eval(model, val_loader, tokenizer, cfg, dev)

        print(f"\nEpoch {epoch} validation metrics:")
        print(metrics)

        bleu = float(metrics.get("BLEU", 0.0))

        if bleu > best_bleu:
            best_bleu = bleu

            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg,
                "mode": args.mode,
                "run_name": args.run_name,
                "t5_model": cfg["model"]["t5_model"],
                "best_bleu": best_bleu,
                "epoch": epoch,
            }

            torch.save(ckpt, ckpt_dir / "best.pt")
            tokenizer.save_pretrained(ckpt_dir / "tokenizer")

            with open(ckpt_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            print("Saved best checkpoint:", ckpt_dir / "best.pt")

    print("\nTraining finished.")
    print("Best BLEU:", best_bleu)
    print("Checkpoint dir:", ckpt_dir)


if __name__ == "__main__":
    main()