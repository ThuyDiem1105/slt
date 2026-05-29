import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
import math 

class SignMusketeersT5(nn.Module):
    def __init__(self, t5_name, face_proj_dim=256, hand_proj_dim=256, pose_proj_dim=128, dropout=0.1, use_confidence=False):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        d_model = self.t5.config.d_model
        self.use_confidence = use_confidence
        self.face_proj = nn.Sequential(nn.Linear(384, face_proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lh_proj = nn.Sequential(nn.Linear(384, hand_proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.rh_proj = nn.Sequential(nn.Linear(384, hand_proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.pose_proj = nn.Sequential(nn.Linear(14, pose_proj_dim), nn.ReLU(), nn.Dropout(dropout))
        concat_dim = face_proj_dim + hand_proj_dim + hand_proj_dim + pose_proj_dim
        self.to_t5 = nn.Sequential(nn.Linear(concat_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        if use_confidence:
            self.conf_gate = nn.Sequential(
                nn.Linear(4, 4),
                nn.Sigmoid()
            )

    def make_inputs_embeds(self, face, left_hand, right_hand, pose14, confidence):
        f = self.face_proj(face)
        l = self.lh_proj(left_hand)
        r = self.rh_proj(right_hand)
        p = self.pose_proj(pose14)
        if self.use_confidence:
            # confidence: [B,T,4] = left, right, face, pose.
            # Gating học được + nhân với confidence gốc để giảm stream mất tín hiệu.
            g = self.conf_gate(confidence[:, :, :4]) * confidence[:, :, :4]
            l = l * g[:, :, 0:1]
            r = r * g[:, :, 1:2]
            f = f * g[:, :, 2:3]
            p = p * g[:, :, 3:4]
        x = torch.cat([f, l, r, p], dim=-1)
        return self.to_t5(x)

    def forward(self, face, left_hand, right_hand, pose14, confidence, attention_mask, labels=None):
        embeds = self.make_inputs_embeds(face, left_hand, right_hand, pose14, confidence)
        return self.t5(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, face, left_hand, right_hand, pose14, confidence, attention_mask, max_length=128, num_beams=5):
        embeds = self.make_inputs_embeds(face, left_hand, right_hand, pose14, confidence)
        return self.t5.generate(inputs_embeds=embeds, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams)

class LearnedConfidence(nn.Module):
    """
    Learned confidence module:
    c = alpha * learned_conf + (1 - alpha) * mediapipe_conf
    """

    def __init__(self, d_model=256, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, stream_tokens, mp_conf):
        """
        stream_tokens: (B, T, 4, D)
        mp_conf:       (B, T, 4)
        """
        learned_conf = self.scorer(stream_tokens).squeeze(-1)  # (B, T, 4)
        conf = self.alpha * learned_conf + (1.0 - self.alpha) * mp_conf
        return conf


class ConfidenceTemporalSmoother(nn.Module):
    """
    Optional temporal smoothing for confidence scores.
    Input : (B, T, 4)
    Output: (B, T, 4)
    """

    def __init__(self, num_streams=4, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_streams,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, num_streams),
            nn.Sigmoid()
        )

    def forward(self, conf):
        h, _ = self.gru(conf)
        return self.out(h)


class ConfidenceBiasedCrossStreamBlock(nn.Module):
    """
    CA-CSA block.

    Attention over 4 stream tokens:
        face, left hand, right hand, pose

    Confidence bias:
        B[i, j] = log(c_j + eps)

    This penalizes attending to unreliable streams.
    """

    def __init__(self, d_model=256, num_heads=4, dropout=0.1, eps=1e-6, use_confidence_bias=True):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps
        self.use_confidence_bias = use_confidence_bias

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, conf, return_attn=False):
        """
        x:    (B, T, 4, D)
        conf: (B, T, 4)
        """
        B, T, S, D = x.shape
        H = self.num_heads

        residual = x

        q = self.q_proj(x).view(B, T, S, H, self.head_dim).transpose(2, 3)
        k = self.k_proj(x).view(B, T, S, H, self.head_dim).transpose(2, 3)
        v = self.v_proj(x).view(B, T, S, H, self.head_dim).transpose(2, 3)

        # q, k, v: (B, T, H, 4, head_dim)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # logits: (B, T, H, 4, 4)

        if self.use_confidence_bias:
            conf_clipped = conf.clamp(0.05, 0.95)
            bias = torch.log(conf_clipped + self.eps)  # (B, T, 4)
            bias = bias[:, :, None, None, :]           # (B, T, 1, 1, 4)
            logits = logits + bias

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, T, H, 4, head_dim)
        out = out.transpose(2, 3).contiguous().view(B, T, S, D)
        out = self.o_proj(out)

        x = self.norm1(residual + out)
        x = self.norm2(x + self.ffn(x))

        if return_attn:
            return x, attn

        return x


class CA_CSAT5(nn.Module):
    """
    Confidence-Aware Cross-Stream Attention + T5.

    Input streams:
        face       : (B, T, 384)
        left_hand  : (B, T, 384)
        right_hand : (B, T, 384)
        pose14     : (B, T, 14)
        confidence : (B, T, 4) or (B, T, 5)

    Output:
        T5 conditional generation loss / generated text.
    """

    def __init__(
        self,
        t5_name="t5-small",
        stream_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        alpha=0.5,
        use_learned_conf=True,
        use_temporal_smoothing=True,
        use_confidence_bias=True,
    ):
        super().__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.t5_dim = self.t5.config.d_model

        self.stream_dim = stream_dim
        self.use_learned_conf = use_learned_conf
        self.use_temporal_smoothing = use_temporal_smoothing
        self.use_confidence_bias = use_confidence_bias

        # Stream tokenization
        self.face_proj = nn.Linear(384, stream_dim)
        self.lhand_proj = nn.Linear(384, stream_dim)
        self.rhand_proj = nn.Linear(384, stream_dim)
        self.pose_proj = nn.Linear(14, stream_dim)

        # Stream identity embedding: [face, left hand, right hand, pose]
        self.stream_embed = nn.Parameter(torch.randn(4, stream_dim) * 0.02)

        # Confidence estimator
        if use_learned_conf:
            self.confidence_module = LearnedConfidence(
                d_model=stream_dim,
                alpha=alpha
            )
        else:
            self.confidence_module = None

        # Temporal smoothing
        if use_temporal_smoothing:
            self.conf_smoother = ConfidenceTemporalSmoother(
                num_streams=4,
                hidden_dim=16
            )
        else:
            self.conf_smoother = None

        # CA-CSA layers
        self.blocks = nn.ModuleList([
            ConfidenceBiasedCrossStreamBlock(
                d_model=stream_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_confidence_bias=use_confidence_bias
            )
            for _ in range(num_layers)
        ])

        # Final projection to T5 embedding dimension
        self.to_t5 = nn.Linear(stream_dim, self.t5_dim)

    def _get_batch_value(self, batch, names):
        for name in names:
            if name in batch:
                return batch[name]
        raise KeyError(f"Cannot find any of keys {names} in batch. Available keys: {list(batch.keys())}")

    def build_stream_tokens(self, batch):
        """
        Returns:
            streams: (B, T, 4, D)
        """
        face = self._get_batch_value(batch, ["face"])
        left_hand = self._get_batch_value(batch, ["left_hand", "lhand"])
        right_hand = self._get_batch_value(batch, ["right_hand", "rhand"])
        pose14 = self._get_batch_value(batch, ["pose14", "pose"])

        face = self.face_proj(face)
        left_hand = self.lhand_proj(left_hand)
        right_hand = self.rhand_proj(right_hand)
        pose = self.pose_proj(pose14)

        streams = torch.stack(
            [face, left_hand, right_hand, pose],
            dim=2
        )  # (B, T, 4, D)

        streams = streams + self.stream_embed.view(1, 1, 4, self.stream_dim)

        return streams

    def prepare_confidence(self, batch, streams):
        """
        Convert confidence to order:
            [face, left_hand, right_hand, pose]

        Some feature files save confidence as:
            [left_hand, right_hand, face, pose, rgb]
        """
        if "confidence" in batch:
            mp_conf = batch["confidence"]
        else:
            B, T, S, _ = streams.shape
            mp_conf = torch.ones(B, T, S, device=streams.device)

        # If confidence has 5 streams: [left, right, face, pose, rgb]
        if mp_conf.size(-1) == 5:
            mp_conf = torch.stack(
                [
                    mp_conf[..., 2],  # face
                    mp_conf[..., 0],  # left hand
                    mp_conf[..., 1],  # right hand
                    mp_conf[..., 3],  # pose
                ],
                dim=-1
            )

        # If confidence has 4 streams but old order may be [left, right, face, pose]
        # In this project, feature extraction usually follows [face, left, right, pose]
        # after SignMusketeers extraction. If your feature file is old, adjust here.
        elif mp_conf.size(-1) == 4:
            mp_conf = mp_conf[..., :4]

        else:
            B, T, S, _ = streams.shape
            mp_conf = torch.ones(B, T, S, device=streams.device)

        mp_conf = mp_conf.to(streams.device).float()

        if self.confidence_module is not None:
            conf = self.confidence_module(streams, mp_conf)
        else:
            conf = mp_conf

        if self.conf_smoother is not None:
            conf = self.conf_smoother(conf)

        return conf

    def fuse(self, batch, return_attn=False):
        streams = self.build_stream_tokens(batch)
        conf = self.prepare_confidence(batch, streams)

        attn_maps = []

        for block in self.blocks:
            if return_attn:
                streams, attn = block(streams, conf, return_attn=True)
                attn_maps.append(attn)
            else:
                streams = block(streams, conf, return_attn=False)

        # Confidence-weighted pooling
        weights = torch.softmax(conf, dim=-1)  # (B, T, 4)
        fused = (streams * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)

        inputs_embeds = self.to_t5(fused)  # (B, T, t5_dim)

        if return_attn:
            return inputs_embeds, conf, attn_maps

        return inputs_embeds

    def forward(self, batch, labels=None):
        inputs_embeds = self.fuse(batch)

        attention_mask = batch.get("attention_mask", None)

        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

    @torch.no_grad()
    def generate(self, batch, **gen_kwargs):
        inputs_embeds = self.fuse(batch)
        attention_mask = batch.get("attention_mask", None)

        return self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    @torch.no_grad()
    def get_attention_analysis(self, batch):
        inputs_embeds, conf, attn_maps = self.fuse(batch, return_attn=True)

        return {
            "confidence": conf.detach().cpu(),
            "attention": [a.detach().cpu() for a in attn_maps]
        }