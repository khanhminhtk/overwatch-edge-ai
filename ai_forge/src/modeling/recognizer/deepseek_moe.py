import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.common.activations import SwiGLU


class DeepSeekMOE(nn.Module):
    def __init__(
        self,
        device: str | torch.device | None,
        n_shared_experts: int,
        n_routed_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        d_model: int,
        dropout: float,
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
                f"top_k must be in [1, n_routed_experts], got top_k={top_k}, n_routed_experts={n_routed_experts}"
            )

        self.d_model = d_model
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim

        self.input_proj = nn.Linear(
            d_model,
            expert_hidden_dim,
            bias=False,
            device=device,
        )

        self.shared_experts = nn.ModuleList(
            [
                SwiGLU(
                    out_feature=expert_hidden_dim,
                    device=device,
                )
                for _ in range(n_shared_experts)
            ]
        )

        self.routed_experts = nn.ModuleList(
            [
                SwiGLU(
                    out_feature=expert_hidden_dim,
                    device=device,
                )
                for _ in range(n_routed_experts)
            ]
        )

        self.act_input_proj = nn.LeakyReLU()

        self.router = nn.Linear(
            expert_hidden_dim,
            n_routed_experts,
            bias=False,
            device=device,
        )

        self.out_proj = nn.Linear(
            expert_hidden_dim,
            d_model,
            bias=False,
            device=device,
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.last_aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True,
    ):
        # x: [B, N, D]
        bsz, seq_len, _ = x.shape

        h = self.act_input_proj(self.input_proj(x))
        h_flat = h.reshape(-1, self.expert_hidden_dim)

        if self.n_shared_experts > 0:
            shared_out = torch.zeros_like(h_flat)

            for expert in self.shared_experts:
                shared_out = shared_out + expert(h_flat)

            if self.n_shared_experts > 1:
                shared_out = shared_out / self.n_shared_experts
        else:
            shared_out = torch.zeros_like(h_flat)

        router_logits = self.router(h_flat)

        routing_weights = F.softmax(
            router_logits.float(),
            dim=-1,
        ).to(dtype=h_flat.dtype)

        top_k_weights, top_k_indices = torch.topk(
            routing_weights,
            k=self.top_k,
            dim=-1,
        )

        top_k_weights = top_k_weights / top_k_weights.sum(
            dim=-1,
            keepdim=True,
        )

        expert_outputs = []

        for expert in self.routed_experts:
            expert_outputs.append(expert(h_flat))

        # [tokens, n_experts, hidden]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        gather_index = top_k_indices.unsqueeze(-1).expand(
            -1,
            -1,
            self.expert_hidden_dim,
        )

        # [tokens, top_k, hidden]
        selected_outputs = torch.gather(
            expert_outputs,
            dim=1,
            index=gather_index,
        )

        routed_out = (
            selected_outputs * top_k_weights.unsqueeze(-1)
        ).sum(dim=1)

        final_hidden = shared_out + routed_out
        final_hidden = final_hidden.reshape(
            bsz,
            seq_len,
            self.expert_hidden_dim,
        )

        final_hidden = self.dropout_layer(final_hidden)
        out = self.out_proj(final_hidden)

        if not return_aux_loss:
            self.last_aux_loss = None
            return out, None

        aux_loss = self._router_aux_loss(
            routing_weights=routing_weights,
            top_k_indices=top_k_indices,
        )
        self.last_aux_loss = aux_loss

        return out, aux_loss

    def _router_aux_loss(
        self,
        routing_weights: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = routing_weights.size(0)

        expert_prob_mean = routing_weights.mean(dim=0)

        expert_indices = top_k_indices.reshape(-1)

        expert_density = torch.bincount(
            expert_indices,
            minlength=self.n_routed_experts,
        ).to(dtype=expert_prob_mean.dtype)

        expert_density = expert_density / (num_tokens * self.top_k)

        aux_loss = self.n_routed_experts * torch.sum(
            expert_prob_mean * expert_density
        )

        return aux_loss
