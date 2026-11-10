from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictorConfig:
    d_model: int = 1024
    dropout: float = 0.3
    context_layers: int = 1
    use_condition_emb: bool = True


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class ModalityExpert(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(output_dim, dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ModalityExperts(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.audio = ModalityExpert(512, dimension, dropout)
        self.vision = ModalityExpert(1024, dimension, dropout)
        self.text = ModalityExpert(1024, dimension, dropout)
        self.modality_emb = nn.Parameter(torch.zeros(3, dimension))
        nn.init.normal_(self.modality_emb, std=0.02)

    def forward(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        values = (self.audio(audio), self.vision(vision), self.text(text))
        return torch.stack(values, dim=2) + self.modality_emb.view(1, 1, 3, -1)


class MeanFusion(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        value = (tokens * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        return self.norm(value)


class TemporalContext(nn.Module):
    def __init__(self, dimension: int, layers: int, dropout: float):
        super().__init__()
        if dimension % 2:
            raise ValueError("the bidirectional GRU requires an even state dimension")
        self.encoder = nn.GRU(
            input_size=dimension,
            hidden_size=dimension // 2,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.long().sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            value,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            batch_first=True,
            total_length=value.size(1),
        )
        return self.drop(output) * mask.to(dtype=value.dtype).unsqueeze(-1)


def condition_index(mask: torch.Tensor) -> torch.Tensor:
    weights = mask.new_tensor([4, 2, 1])
    index = (mask.long() * weights).sum(dim=-1)
    if bool(index.eq(0).any().item()):
        raise ValueError("a condition must observe at least one modality")
    return index


class Predictor(nn.Module):
    """Mean modality fusion followed by video-level BiGRU context."""

    def __init__(self, config: PredictorConfig | None = None):
        super().__init__()
        self.config = config or PredictorConfig()
        dimension = self.config.d_model
        self.experts = ModalityExperts(dimension, self.config.dropout)
        self.condition_emb = nn.Embedding(8, dimension)
        nn.init.normal_(self.condition_emb.weight, std=0.02)
        self.fusion = MeanFusion(dimension)
        self.context = TemporalContext(
            dimension,
            self.config.context_layers,
            self.config.dropout,
        )
        self.state_norm = nn.LayerNorm(dimension)
        self.predictor = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dimension, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text: torch.Tensor,
        condition_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = self.experts(audio, vision, text)
        condition = condition_mask.to(device=tokens.device, dtype=torch.long)
        if condition.dim() == 1:
            condition = condition.view(1, 1, 3).expand(*tokens.shape[:2], 3)
        elif condition.dim() == 2:
            condition = condition.view(1, 1, 3).expand(*tokens.shape[:2], 3)
        elif condition.shape != tokens.shape[:2] + (3,):
            raise ValueError("condition mask must be [3] or [B,L,3]")
        condition_embedding = self.condition_emb(condition_index(condition))
        fused = self.fusion(tokens, condition)
        context = self.context(fused, sequence_mask.bool())
        state = self.state_norm(fused + context)
        if self.config.use_condition_emb:
            state = self.state_norm(state + condition_embedding)
        prediction = self.predictor(state).squeeze(-1)
        return {"state": state, "pred": prediction}

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return self.predictor(state).squeeze(-1)

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass
class ProjectorConfig:
    text_dim: int
    d_model: int = 1024
    num_anchors: int = 5
    num_missing: int = 3
    meta_dim: int = 128
    bottleneck: int = 256
    dropout: float = 0.1
    init_alpha: float = 0.1
    max_alpha: float = 0.7
    init_anchor_gate: float = 0.9


class SemanticProjector(nn.Module):
    """Action-only bottleneck with state-dependent scalar gating."""

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        if not 0.0 < config.init_alpha < config.max_alpha:
            raise ValueError("init_alpha must lie in (0, max_alpha)")
        self.missing_emb = nn.Embedding(config.num_missing, config.meta_dim)
        self.anchor_emb = nn.Embedding(config.num_anchors, config.meta_dim)
        input_dim = config.text_dim + 2 * config.meta_dim
        self.action_projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.bottleneck),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.bottleneck),
        )
        self.residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.bottleneck, config.bottleneck),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.bottleneck, config.d_model),
                    nn.LayerNorm(config.d_model),
                )
                for _ in range(config.num_missing)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.bottleneck + config.d_model),
                    nn.Linear(config.bottleneck + config.d_model, config.bottleneck),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.bottleneck, 1),
                )
                for _ in range(config.num_missing)
            ]
        )
        ratio = config.init_alpha / config.max_alpha
        alpha_logit = math.log(ratio / (1.0 - ratio))
        self.alpha_logit = nn.Parameter(
            torch.full((config.num_missing,), alpha_logit, dtype=torch.float32)
        )
        gate_logit = math.log(config.init_anchor_gate / (1.0 - config.init_anchor_gate))
        self.anchor_gate_logit = nn.Parameter(
            torch.full(
                (config.num_missing, config.num_anchors),
                gate_logit,
                dtype=torch.float32,
            )
        )
        nn.init.normal_(self.missing_emb.weight, std=0.02)
        nn.init.normal_(self.anchor_emb.weight, std=0.02)

    def alpha(self) -> torch.Tensor:
        return self.config.max_alpha * torch.sigmoid(self.alpha_logit)

    def anchor_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.anchor_gate_logit)

    def encode(
        self,
        action_embedding: torch.Tensor,
        missing_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
    ) -> torch.Tensor:
        missing = missing_idx.long().clamp(0, self.config.num_missing - 1)
        anchor = anchor_idx.long().clamp(0, self.config.num_anchors - 1)
        value = torch.cat(
            [action_embedding, self.missing_emb(missing), self.anchor_emb(anchor)],
            dim=-1,
        )
        return self.action_projector(value)

    def forward(
        self,
        action_embedding: torch.Tensor,
        missing_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
        state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(action_embedding, missing_idx, anchor_idx)
        missing = missing_idx.long().clamp(0, self.config.num_missing - 1)
        anchor = anchor_idx.long().clamp(0, self.config.num_anchors - 1)
        raw = action_embedding.new_zeros(action_embedding.size(0), self.config.d_model)
        gate = action_embedding.new_zeros(action_embedding.size(0), 1)
        for missing_value in range(self.config.num_missing):
            active = missing.eq(missing_value)
            if not bool(active.any().item()):
                continue
            raw[active] = self.residual_heads[missing_value](latent[active])
            state_gate = torch.sigmoid(
                self.gates[missing_value](
                    torch.cat([latent[active], state[active].detach()], dim=-1)
                )
            )
            prior = self.anchor_gates()[missing_value, anchor[active]].unsqueeze(-1)
            gate[active] = state_gate * prior
        scale = self.alpha()[missing].unsqueeze(-1) * gate
        delta = scale * raw
        direction = F.normalize(raw, dim=-1, eps=1e-6)
        return {
            "delta": delta,
            "direction": direction,
            "latent": latent,
            "gate": gate.squeeze(-1),
        }

    def all_candidates(
        self,
        action_embedding: torch.Tensor,
        missing_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, dimension = action_embedding.shape
        flat_output = self(
            action_embedding.reshape(batch * candidates, dimension),
            missing_idx[:, None].expand(batch, candidates).reshape(-1),
            anchor_idx.reshape(-1),
            state[:, None, :].expand(batch, candidates, -1).reshape(
                batch * candidates, -1
            ),
        )
        delta = flat_output["delta"].view(batch, candidates, -1)
        latent = flat_output["latent"].view(batch, candidates, -1)
        return delta, latent

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)


@dataclass
class SelectorConfig:
    d_model: int = 1024
    action_dim: int = 256
    num_candidates: int = 5
    max_steps: int = 5
    num_missing: int = 3
    meta_dim: int = 128
    hidden: int = 512
    dropout: float = 0.1


class ActionSelector(nn.Module):
    """Missing-specific Q network over candidate actions and STOP."""

    def __init__(self, config: SelectorConfig):
        super().__init__()
        self.config = config
        self.missing_emb = nn.Embedding(config.num_missing, config.meta_dim)
        self.step_emb = nn.Embedding(config.max_steps + 2, config.meta_dim)
        action_input = config.d_model + 2 * config.action_dim + 2 * config.meta_dim
        stop_input = config.d_model + config.action_dim + 2 * config.meta_dim
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_input),
            nn.Linear(action_input, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.action_heads = nn.ModuleList(
            [nn.Linear(config.hidden, 1) for _ in range(config.num_missing)]
        )
        self.stop_encoder = nn.Sequential(
            nn.LayerNorm(stop_input),
            nn.Linear(stop_input, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.stop_heads = nn.ModuleList(
            [nn.Linear(config.hidden, 1) for _ in range(config.num_missing)]
        )
        for embedding in (self.missing_emb, self.step_emb):
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(
        self,
        *,
        state: torch.Tensor,
        action_latent: torch.Tensor,
        missing_idx: torch.Tensor,
        used_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        batch, candidates, _ = action_latent.shape
        if candidates != self.config.num_candidates:
            raise ValueError(
                f"selector expects {self.config.num_candidates} candidates, got {candidates}"
            )
        missing = missing_idx.long().clamp(0, self.config.num_missing - 1)
        missing_embedding = self.missing_emb(missing)
        step_ids = torch.full(
            (batch,),
            min(step, self.config.max_steps + 1),
            device=state.device,
            dtype=torch.long,
        )
        step_embedding = self.step_emb(step_ids)
        remaining = candidate_mask.bool() & ~used_mask.bool()
        denominator = remaining.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (
            action_latent * remaining.unsqueeze(-1).to(action_latent.dtype)
        ).sum(dim=1) / denominator
        action_features = torch.cat(
            [
                state[:, None, :].expand(-1, candidates, -1),
                action_latent,
                pooled[:, None, :].expand(-1, candidates, -1),
                missing_embedding[:, None, :].expand(-1, candidates, -1),
                step_embedding[:, None, :].expand(-1, candidates, -1),
            ],
            dim=-1,
        )
        encoded_action = self.action_encoder(action_features)
        action_by_missing = torch.stack(
            [head(encoded_action).squeeze(-1) for head in self.action_heads],
            dim=-1,
        )
        action_q = action_by_missing.gather(
            -1,
            missing[:, None, None].expand(batch, candidates, 1),
        ).squeeze(-1)
        stop_features = torch.cat(
            [state, pooled, missing_embedding, step_embedding],
            dim=-1,
        )
        encoded_stop = self.stop_encoder(stop_features)
        stop_by_missing = torch.stack(
            [head(encoded_stop).squeeze(-1) for head in self.stop_heads],
            dim=-1,
        )
        stop_q = stop_by_missing.gather(-1, missing[:, None]).squeeze(-1)

        return torch.cat([action_q, stop_q.unsqueeze(-1)], dim=-1)

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)
