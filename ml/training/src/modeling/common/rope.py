import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"rotate_half expected even last dim, got {x.shape[-1]}")

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    x_rotated = torch.stack((-x_odd, x_even), dim=-1)
    return x_rotated.flatten(-2)


class ROPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
        has_cls_token: bool = False,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.has_cls_token = has_cls_token

        idx = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        theta = 1.0 / (base ** idx)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32)

        if has_cls_token:
            if max_seq_len == 1:
                position_ids = torch.zeros(1, dtype=torch.float32)
            else:
                tail = torch.arange(max_seq_len - 1, dtype=torch.float32)
                position_ids = torch.cat([torch.zeros(1), tail], dim=0)

        phase = position_ids[:, None] * theta[None, :]
        phase = torch.repeat_interleave(phase, repeats=2, dim=-1)

        sin = torch.sin(phase)[None, None, :, :]
        cos = torch.cos(phase)[None, None, :, :]

        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device | str | None = None,
        has_cls_token: bool | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} > max_seq_len={self.max_seq_len}"
            )

        sin = self.sin_cached[:, :, :seq_len, :]
        cos = self.cos_cached[:, :, :seq_len, :]

        has_cls_token_resolved = self.has_cls_token if has_cls_token is None else bool(has_cls_token)
        if has_cls_token_resolved != self.has_cls_token:
            idx = torch.arange(0, self.dim, 2, dtype=torch.float32, device=sin.device) / self.dim
            theta = 1.0 / (self.base ** idx)
            if has_cls_token_resolved:
                if seq_len == 1:
                    position_ids = torch.zeros(1, dtype=torch.float32, device=sin.device)
                else:
                    tail = torch.arange(seq_len - 1, dtype=torch.float32, device=sin.device)
                    position_ids = torch.cat([torch.zeros(1, dtype=torch.float32, device=sin.device), tail], dim=0)
            else:
                position_ids = torch.arange(seq_len, dtype=torch.float32, device=sin.device)
            phase = position_ids[:, None] * theta[None, :]
            phase = torch.repeat_interleave(phase, repeats=2, dim=-1)
            sin = torch.sin(phase)[None, None, :, :]
            cos = torch.cos(phase)[None, None, :, :]

        if device is not None:
            sin = sin.to(device=device)
            cos = cos.to(device=device)

        if dtype is not None:
            sin = sin.to(dtype=dtype)
            cos = cos.to(dtype=dtype)

        return sin.squeeze(0).squeeze(0), cos.squeeze(0).squeeze(0)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rotary_pos_emb_trt(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        rotary_dim=rotary_dim,
    )

def apply_rotary_pos_emb_trt(
    q: torch.Tensor,  # [B, H, N, D]
    k: torch.Tensor,  # [B, H, N, D]
    cos: torch.Tensor,  # [1, 1, N, D]
    sin: torch.Tensor,  # [1, 1, N, D]
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)

    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]

    cos = cos[..., :rotary_dim].to(dtype=q.dtype)
    sin = sin[..., :rotary_dim].to(dtype=q.dtype)

    q1 = q1 * cos + rotate_half(q1) * sin
    k1 = k1 * cos + rotate_half(k1) * sin

    q = torch.cat([q1, q2], dim=-1)
    k = torch.cat([k1, k2], dim=-1)

    return q, k
