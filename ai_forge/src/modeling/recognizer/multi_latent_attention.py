import torch
import torch.nn as nn

from src.domain.ports.recognizer.recognizer_component_ports import AttentionPort
from src.modeling.common.rope import apply_rotary_pos_emb

class MultiLatentAttention(nn.Module, AttentionPort):
    def __init__(
        self,
        device: str | torch.device,
        num_heads: int,
        head_dim: int,
        out_feature: int,
        d_c: int,
        d_c1: int,
        rope: nn.Module | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"MultiLatentAttention.__init__: num_heads must be > 0, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"MultiLatentAttention.__init__: head_dim must be > 0, got {head_dim}")
        if out_feature <= 0:
            raise ValueError(f"MultiLatentAttention.__init__: out_feature must be > 0, got {out_feature}")
        if rope is not None and head_dim % 2 != 0:
            raise ValueError(
                f"MultiLatentAttention.__init__: head_dim must be even when rope is enabled, got {head_dim}"
            )

        self._device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_feature = out_feature
        self.d_model = num_heads * head_dim
        self.d_c = d_c
        self.d_c1 = d_c1
        self.rope = rope

        self.w_dkv = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_c,
            bias=False,
            device=device,
        )
        
        self.w_ukv = nn.Linear(
            in_features=self.d_c,
            out_features=self.d_model * 2,
            bias=False,
            device=device,
        )

        self.w_dq = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_c1,
            bias=False,
            device=device,
        )

        self.w_uq = nn.Linear(
            in_features=self.d_c1,
            out_features=self.d_model,
            bias=False,
            device=device,
        )

        self.w_out = nn.Linear(
            in_features=self.d_model,
            out_features=self.out_feature,
            bias=False,
            device=device,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _reshape_heads(self, X: torch.Tensor) -> torch.Tensor:
        #X [Bsz, Patch, d_model]
        bsz, patch, _ = X.shape
        X = X.reshape(bsz, patch, self.num_heads, self.head_dim)
        return X.permute(0, 2, 1, 3).contiguous()
    
    def _merge_heads(self, X: torch.Tensor) -> torch.Tensor:
        #X [bsz, num_head, patch, d_head]
        bsz, _, patch, _ = X.shape
        X = X.permute(0, 2, 1, 3).contiguous()
        return X.reshape(bsz, patch, self.d_model)
    
    def _validate_attn_mask(self, attn_mask: torch.Tensor, bsz: int, seq_len: int) -> None:
        if attn_mask.dim() < 2 or attn_mask.dim() > 4:
            raise ValueError(
                f"attn_mask must have 2 to 4 dims, got shape={tuple(attn_mask.shape)}"
            )
        if attn_mask.size(-2) != seq_len or attn_mask.size(-1) != seq_len:
            raise ValueError(
                f"attn_mask last two dims must be ({seq_len}, {seq_len}), got {tuple(attn_mask.shape)}"
            )
        if attn_mask.dim() == 3 and attn_mask.size(0) not in (1, bsz):
            raise ValueError(
                f"attn_mask batch dim must be 1 or {bsz}, got shape={tuple(attn_mask.shape)}"
            )
        if attn_mask.dim() == 4:
            if attn_mask.size(0) not in (1, bsz):
                raise ValueError(
                    f"attn_mask batch dim must be 1 or {bsz}, got shape={tuple(attn_mask.shape)}"
                )
            if attn_mask.size(1) not in (1, self.num_heads):
                raise ValueError(
                    f"attn_mask head dim must be 1 or {self.num_heads}, got shape={tuple(attn_mask.shape)}"
                )
            
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        has_cls_token: bool = False,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be 3D [batch, patch, d_model], got shape={tuple(x.shape)}")
        if x.size(-1) != self.d_model:
            raise ValueError(f"x last dim must be {self.d_model}, got {x.size(-1)}")

        bsz, patch, _ = x.shape

        dkv = self.w_dkv(x) # [B, P, d_c]
        ukv = self.w_ukv(dkv) # [B, P, d_model * 2]
        uk, uv = ukv.chunk(2, dim=-1) # [B, P, d_model], [B, P, d_model]
        dq = self.w_dq(x) # [B, P, d_c1]
        uq = self.w_uq(dq) # [B, P, d_model]

        q = self._reshape_heads(uq) # [B, num_head, P, head_dim]
        k = self._reshape_heads(uk) # [B, num_head, P, head_dim]
        v = self._reshape_heads(uv) # [B, num_head, P, head_dim]

        if self.rope is not None:
            sin, cos = self.rope(patch, device=x.device, has_cls_token=has_cls_token)
            q, k = apply_rotary_pos_emb(
                q=q,
                k=k,
                cos=cos,
                sin=sin,
                rotary_dim=self.head_dim,
            )

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B, num_head, P, P]
        effective_mask = attn_mask if attn_mask is not None else mask
        if mask is not None and attn_mask is not None:
            raise ValueError("Provide only one of mask or attn_mask, not both.")

        if effective_mask is not None:
            self._validate_attn_mask(effective_mask, bsz, patch)
            if effective_mask.dim() == 2:
                effective_mask = effective_mask.unsqueeze(0).unsqueeze(0)
            elif effective_mask.dim() == 3:
                effective_mask = effective_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(effective_mask == 0, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1) # [B, num_head, P, P]
        attn_probs = self.attn_drop(attn_probs)
        attn_output = torch.matmul(attn_probs, v) # [B, num_head, P, head_dim]
        attn_output = self._merge_heads(attn_output) # [B, P, d_model]
        attn_output = self.w_out(attn_output) # [B, P, out_feature]
        attn_output = self.proj_drop(attn_output)
        return attn_output
