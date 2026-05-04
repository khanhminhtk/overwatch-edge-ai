from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DomainConfig:
    data: Mapping[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in keys:
            if not isinstance(node, Mapping) or key not in node:
                return default
            node = node[key]
        return node

    def get_required(self, *keys: str) -> Any:
        value = self.get(*keys, default=None)
        if value is None:
            path = ".".join(keys)
            raise ValueError(f"Missing required config key: {path}")
        return value


@dataclass(frozen=True)
class RecognizerConfig:
    d_model: int
    num_heads: int
    d_c: int
    d_c1: int
    dim_rope: int
    n_shared_experts: int
    n_routed_experts: int
    top_k: int
    expert_hidden_dim: int
    attn_drop: float
    proj_drop: float
    moe_dropout: float
    config_training: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RecognizerConfig":
        required = [
            "d_model",
            "num_heads",
            "d_c",
            "d_c1",
            "dim_rope",
            "n_shared_experts",
            "n_routed_experts",
            "top_k",
            "expert_hidden_dim",
            "attn_drop",
            "proj_drop",
            "moe_dropout",
            "config_training",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing recognizer config keys: {missing}")

        return cls(
            d_model=int(data["d_model"]),
            num_heads=int(data["num_heads"]),
            d_c=int(data["d_c"]),
            d_c1=int(data["d_c1"]),
            dim_rope=int(data["dim_rope"]),
            n_shared_experts=int(data["n_shared_experts"]),
            n_routed_experts=int(data["n_routed_experts"]),
            top_k=int(data["top_k"]),
            expert_hidden_dim=int(data["expert_hidden_dim"]),
            attn_drop=float(data["attn_drop"]),
            proj_drop=float(data["proj_drop"]),
            moe_dropout=float(data["moe_dropout"]),
            config_training=data["config_training"],
        )
