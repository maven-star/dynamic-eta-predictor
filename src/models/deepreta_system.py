"""Route ETA model with fixed feature scaling and ordered quantile outputs.

The model predicts *multipliers* of a routing engine's base duration, rather
than an absolute duration. This keeps the learned component focused on delay
and uncertainty while OSRM supplies the road-network travel-time baseline.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustDeeprETAEncoder(nn.Module):
    """Encode a single route of arbitrary non-empty length."""

    # elevation m, grade %, temperature C, relative humidity %, wind km/h,
    # precipitation mm/h. Fixed physical ranges avoid route-local scaling.
    FEATURE_RANGES = (
        (-500.0, 9000.0),
        (-30.0, 30.0),
        (-50.0, 60.0),
        (0.0, 100.0),
        (0.0, 200.0),
        (0.0, 200.0),
    )

    def __init__(
        self,
        vocab_size: int = 5000,
        h3_dim: int = 32,
        num_bins: int = 20,
        continuous_dim: int = 16,
        max_seq_len: int = 8,
        embed_dim: int = 128,
        linformer_k: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.num_continuous_features = 6
        self.max_seq_len = max_seq_len
        self.h3_embedding = nn.Embedding(vocab_size, h3_dim)
        self.continuous_embeddings = nn.ModuleList(
            [nn.Embedding(num_bins, continuous_dim) for _ in range(self.num_continuous_features)]
        )
        self.driver_projection = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))
        self.sequence_projection = nn.Linear(h3_dim + self.num_continuous_features * continuous_dim, embed_dim)
        self.linformer_k = linformer_k
        self.E_proj = nn.Parameter(torch.randn(self.linformer_k, max_seq_len))
        self.W_proj = nn.Parameter(torch.randn(self.linformer_k, max_seq_len))
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.final_fusion = nn.Linear(embed_dim + 32, embed_dim)

    def _hash_h3_cells(self, h3_strings: Sequence[str], device: torch.device) -> torch.Tensor:
        indices = [
            int(hashlib.md5(str(cell).encode("utf-8")).hexdigest(), 16) % self.vocab_size
            for cell in h3_strings
        ]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _discretize_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.num_continuous_features:
            raise ValueError("continuous_features must have shape [sequence_length, 6]")
        bounds = torch.tensor(self.FEATURE_RANGES, dtype=features.dtype, device=features.device)
        lower, upper = bounds[:, 0], bounds[:, 1]
        normalized = (features - lower) / (upper - lower)
        return torch.clamp((normalized * self.num_bins).long(), 0, self.num_bins - 1)

    @staticmethod
    def _projection_for_length(projection: torch.Tensor, length: int) -> torch.Tensor:
        if length == projection.shape[1]:
            return projection
        return F.interpolate(projection.unsqueeze(0), size=length, mode="linear", align_corners=True).squeeze(0)

    def forward(
        self,
        h3_strings: Sequence[str],
        continuous_features: torch.Tensor,
        driver_profile: torch.Tensor,
    ) -> torch.Tensor:
        if not h3_strings:
            raise ValueError("a route must contain at least one sampled point")
        if len(h3_strings) != continuous_features.shape[0]:
            raise ValueError("h3_strings and continuous_features must have the same sequence length")
        if driver_profile.numel() != 3:
            raise ValueError("driver_profile must contain three values")

        device = self.h3_embedding.weight.device
        features = continuous_features.to(device=device, dtype=torch.float32)
        driver = driver_profile.to(device=device, dtype=torch.float32).reshape(3)
        spatial = self.h3_embedding(self._hash_h3_cells(h3_strings, device))
        bins = self._discretize_features(features)
        continuous = [embedding(bins[:, index]) for index, embedding in enumerate(self.continuous_embeddings)]
        seq_x = self.sequence_projection(torch.cat([spatial, *continuous], dim=-1)).unsqueeze(0)
        driver_embedding = self.driver_projection(driver.unsqueeze(0))

        q, k, v = self.q_linear(seq_x), self.k_linear(seq_x), self.v_linear(seq_x)
        e_proj = self._projection_for_length(self.E_proj, seq_x.shape[1])
        w_proj = self._projection_for_length(self.W_proj, seq_x.shape[1])
        k_projected = torch.matmul(e_proj, k.squeeze(0)).unsqueeze(0)
        v_projected = torch.matmul(w_proj, v.squeeze(0)).unsqueeze(0)
        scores = torch.matmul(q, k_projected.transpose(1, 2)) / (seq_x.shape[-1] ** 0.5)
        context = torch.matmul(F.softmax(scores, dim=-1), v_projected)
        sequence = self.layer_norm(seq_x + self.out_projection(context)).squeeze(0)
        return self.final_fusion(torch.cat([sequence, driver_embedding.expand(sequence.shape[0], -1)], dim=-1))


class DeeprETAPredictionHead(nn.Module):
    """Predict strictly ordered p10, p50, p90 ETA multipliers."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.temporal_pooler = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.quantile_head = nn.Linear(hidden_dim, 3)

    def forward(self, encoded_route_tokens: torch.Tensor) -> torch.Tensor:
        pooled = self.temporal_pooler(encoded_route_tokens.unsqueeze(0).transpose(1, 2)).squeeze(-1)
        raw_lower, raw_median, raw_upper = self.quantile_head(self.mlp(pooled)).squeeze(0)
        median = F.softplus(raw_median) + 1e-4
        upper_gap = F.softplus(raw_upper) + 1e-4
        # A fractional lower head and an additive upper gap make crossings
        # mathematically impossible for every parameter value.
        p10 = median * torch.sigmoid(raw_lower)
        return torch.stack((p10, median, median + upper_gap))


class DeeprETAEndToEndSystem(nn.Module):
    """Unified ETA multiplier model."""

    def __init__(self, **encoder_kwargs: int) -> None:
        super().__init__()
        self.encoder = RobustDeeprETAEncoder(**encoder_kwargs)
        self.predictor = DeeprETAPredictionHead()

    def predict_eta(
        self,
        h3_strings: Sequence[str],
        continuous_features: torch.Tensor,
        driver_profile: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(self.encoder(h3_strings, continuous_features, driver_profile))
