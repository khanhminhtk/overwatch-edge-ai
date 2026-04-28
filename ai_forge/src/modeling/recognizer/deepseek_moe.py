from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.domain.ports.recognizer.recognizer_component_ports import MoEPort
from src.modeling.common.activations import SwiGLU

class DeepSeekMOE(nn.Module, MoEPort):
    def __init__(
        self,
        device: str | torch.device | None,
        n_shared_experts: int,
        n_routed_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        d_model: int,
        dropout: float
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if expert_hidden_dim <= 0:
            raise ValueError(f"expert_hidden_dim must be > 0, got {expert_hidden_dim}")
        if n_routed_experts <= 0:
            raise ValueError(f"n_routed_experts must be > 0, got {n_routed_experts}")
        if n_shared_experts < 0:
            raise ValueError(f"n_shared_experts must be >= 0, got {n_shared_experts}")
        if top_k <= 0 or top_k > n_routed_experts:
            raise ValueError(
                f"top_k must be in [1, n_routed_experts], got top_k={top_k}, "
                f"n_routed_experts={n_routed_experts}"
            )
        
        self.d_model = d_model
        self._device = device
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim

        self.input_proj = nn.Linear(
            in_features=self.d_model,
            out_features=self.expert_hidden_dim,
            bias=False,
            device=self._device
        )

        self.shared_experts = nn.ModuleList(
            (
                SwiGLU(
                    out_feature=self.expert_hidden_dim,
                    device=self._device
                )
                for _ in range(self.n_shared_experts)
            )
        )

        self.routed_experts = nn.ModuleList(
            (
                SwiGLU(
                    out_feature=self.expert_hidden_dim,
                    device=self._device
                )
                for _ in range(self.n_routed_experts)
            )
        )

        self.act_input_proj = nn.LeakyReLU()

        self.router = nn.Linear(
            in_features=self.expert_hidden_dim,
            out_features=self.n_routed_experts,
            bias=False,
            device=self._device
        )

        self.out_proj = nn.Linear(
            in_features=self.expert_hidden_dim,
            out_features=self.d_model,
            bias=False,
            device=self._device
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.last_aux_loss: torch.Tensor | None = None


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if x.dim() != 3:
            raise ValueError(f"x must be 3D [batch, patch, d_model], got shape={tuple(x.shape)}")

        bsz, patch, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"last dim of x must be {self.d_model}, got {d_model}")
        
        out_input_proj = self.act_input_proj(self.input_proj(x)) #(bsz, patch, expert_hidden_dim)
        out_input_proj_flat = out_input_proj.reshape(-1, self.expert_hidden_dim) #(bsz*patch, expert_hidden_dim)
        shared_out = torch.zeros_like(out_input_proj_flat)

        for shared_expert in self.shared_experts:
            shared_out += shared_expert(out_input_proj_flat)

        if self.n_shared_experts > 1:
            shared_out = shared_out / self.n_shared_experts

        router_logits = self.router(out_input_proj_flat)
        routing_weights = F.softmax(router_logits.float(), dim=-1).to(dtype=out_input_proj_flat.dtype)

        top_k_weights, top_k_indices = torch.topk(routing_weights, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        num_token = out_input_proj_flat.size(0)
        token_indices = torch.arange(num_token, device=out_input_proj_flat.device).unsqueeze(-1)
        token_indices = token_indices.expand(num_token, self.top_k).reshape(-1)
        expert_indices = top_k_indices.reshape(-1)
        dispatch_weights = top_k_weights.reshape(-1, 1)
        expert_prob_mean = routing_weights.mean(dim=0)
        expert_density = torch.bincount(
            expert_indices,
            minlength=self.n_routed_experts
        ).to(dtype=expert_prob_mean.dtype) / (num_token * self.top_k)
        self.last_aux_loss = self.n_routed_experts * torch.sum(expert_prob_mean * expert_density)

        routed_out = torch.zeros_like(out_input_proj_flat)
        for expert_idx_tensor in torch.unique(expert_indices):
            expert_idx = int(expert_idx_tensor)
            expert = self.routed_experts[expert_idx]
            selected_positions = torch.nonzero(expert_indices == expert_idx_tensor, as_tuple=False).squeeze(-1)
            selected_tokens = token_indices.index_select(0, selected_positions)
            expert_input = out_input_proj_flat.index_select(0, selected_tokens)
            expert_output = expert(expert_input)
            selected_weights = dispatch_weights.index_select(0, selected_positions)
            weighted_output = expert_output * selected_weights
            routed_out.index_add_(0, selected_tokens, weighted_output)

        final_hidden = shared_out + routed_out
        final_hidden = final_hidden.reshape(bsz, patch, self.expert_hidden_dim)
        final_hidden = self.dropout_layer(final_hidden)
        return self.out_proj(final_hidden), self.last_aux_loss
