import torch
import torch.nn as nn

from src.domain.ports.common.common_component_ports import PositionalEncodingPort


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., dim) -> (..., dim)
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"rotate_half: expected even last dimension, got {x.shape[-1]}"
        )
    x_even = x[..., ::2]  # (..., dim/2)
    x_odd = x[..., 1::2]  # (..., dim/2)
    x_rotated = torch.stack((-x_odd, x_even), dim=-1)  # (..., dim/2, 2)
    return x_rotated.flatten(-2)  # (..., dim)


class ROPE(nn.Module, PositionalEncodingPort):
    def __init__(
        self,
        dim: int,
        base: int = 10000,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"ROPE.__init__: dim must be > 0, got {dim}")
        if dim % 2 != 0:
            raise ValueError(f"ROPE.__init__: dim must be even, got {dim}")
        if base <= 1:
            raise ValueError(f"ROPE.__init__: base must be > 1, got {base}")
        self.dim = dim
        self.base = base

        idx = torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
        theta = 1.0 / (self.base ** idx)  # (dim/2,)
        self.register_buffer("theta", theta)

    def forward(
        self,
        seq_len: int,
        device: str | torch.device | None = None,
        has_cls_token: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError(f"ROPE.forward: seq_len must be > 0, got {seq_len}")

        target_device = device if device is not None else self.theta.device
        position_ids = torch.arange(
            seq_len, device=target_device, dtype=self.theta.dtype
        )  # (seq_len,)
        if has_cls_token:
            if seq_len == 1:
                position_ids = torch.zeros(1, device=target_device, dtype=self.theta.dtype)
            else:
                tail = torch.arange(
                    seq_len - 1, device=target_device, dtype=self.theta.dtype
                )
                position_ids = torch.cat(
                    [torch.zeros(1, device=target_device, dtype=self.theta.dtype), tail]
                )

        theta = self.theta.to(target_device)  # (dim/2,)
        phase = position_ids[:, None] * theta[None, :]  # (seq_len, dim/2)
        phase = torch.repeat_interleave(phase, repeats=2, dim=-1)  # (seq_len, dim)

        sin = torch.sin(phase)
        cos = torch.cos(phase)
        return sin, cos

    @property
    def get_dim(self) -> int:
        return self.dim

def apply_rotary_pos_emb(
    q: torch.Tensor,  # [B,H,N,D]
    k: torch.Tensor,  # [B,H,N,D]
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            f"q/k must be rank-4 [B,H,N,D], got q.ndim={q.ndim}, k.ndim={k.ndim}"
        )
    if q.shape[:3] != k.shape[:3]:
        raise ValueError(
            f"q/k must have same [B,H,N], got q={q.shape[:3]}, k={k.shape[:3]}"
        )
    if cos.ndim < 2 or sin.ndim < 2:
        raise ValueError(
            f"cos/sin must have at least 2 dims [..., N, D], got cos.ndim={cos.ndim}, sin.ndim={sin.ndim}"
        )

    if rotary_dim is None:
        rotary_dim = q.size(-1)
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if rotary_dim > q.size(-1) or rotary_dim > k.size(-1):
        raise ValueError(
            f"rotary_dim ({rotary_dim}) must be <= q/k last dim ({q.size(-1)}, {k.size(-1)})"
        )
    if cos.size(-1) < rotary_dim or sin.size(-1) < rotary_dim:
        raise ValueError(
            f"cos/sin last dim must be >= rotary_dim ({rotary_dim}), got ({cos.size(-1)}, {sin.size(-1)})"
        )
    if cos.size(-2) != q.size(-2) or sin.size(-2) != k.size(-2):
        raise ValueError(
            f"cos/sin seq len must match q/k seq len, got cos={cos.size(-2)}, sin={sin.size(-2)}, q={q.size(-2)}, k={k.size(-2)}"
        )

    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., :rotary_dim].to(device=q.device, dtype=q.dtype)
    sin_q = sin[..., :rotary_dim].to(device=q.device, dtype=q.dtype)
    cos_k = cos_q.to(device=k.device, dtype=k.dtype)
    sin_k = sin_q.to(device=k.device, dtype=k.dtype)

    q1_rot = q1 * cos_q + rotate_half(q1) * sin_q
    k1_rot = k1 * cos_k + rotate_half(k1) * sin_k

    q_rot = torch.cat([q1_rot, q2], dim=-1)
    k_rot = torch.cat([k1_rot, k2], dim=-1)
    return q_rot, k_rot
