from __future__ import annotations

import torch
from torch import nn


class ForecastModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_groups: int,
        embedding_dim: int = 128,
        numeric_dim: int = 5,
        time_dim: int = 4,
        id_embedding_dim: int = 32,
        hidden_dim: int = 160,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.token_dropout = nn.Dropout(dropout)

        # Use mean+max concatenation for richer representation
        self.token_projection = nn.Linear(embedding_dim * 2, hidden_dim)
        self.numeric_projection = nn.Sequential(
            nn.LayerNorm(numeric_dim),
            nn.Linear(numeric_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
        )

        self.id_embedding = nn.Embedding(max(num_groups, 1), id_embedding_dim)
        self.id_projection = nn.Linear(id_embedding_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.token_embedding.weight)
        for module in [self.token_projection, self.numeric_projection[1], self.time_projection[0], self.id_projection]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for seq in [self.classifier_head, self.regression_head]:
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = batch["tokens"]  # (B, S, T)
        numeric = batch["numeric"]  # (B, S, F)
        time_feat = batch["time"]  # (B, S, 4)
        group_id = batch["group_id"]  # (B,)

        batch_size, seq_len, token_len = tokens.shape

        # Improved text feature extraction: use both mean and max pooling
        token_emb = self.token_embedding(tokens.view(batch_size * seq_len, token_len))
        mask = (tokens.view(batch_size * seq_len, token_len) != 0).float().unsqueeze(-1)
        
        # Mean pooling
        token_emb_sum = (token_emb * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1.0)
        token_mean = token_emb_sum / token_counts
        
        # Max pooling (better for capturing important tokens)
        token_emb_masked = token_emb * mask + (1 - mask) * (-1e9)
        token_max = token_emb_masked.max(dim=1)[0]
        
        # Concatenate mean and max, then project
        token_combined = torch.cat([token_mean, token_max], dim=-1)
        token_combined = token_combined.view(batch_size, seq_len, -1)
        token_combined = self.token_dropout(token_combined)
        token_proj = self.token_projection(token_combined)

        numeric_proj = self.numeric_projection(numeric)
        time_proj = self.time_projection(time_feat)

        id_emb = self.id_embedding(group_id)
        id_proj = self.id_projection(id_emb).unsqueeze(1)
        id_proj = id_proj.expand(-1, seq_len, -1)

        seq_repr = token_proj + numeric_proj + time_proj + id_proj
        encoded = self.encoder(seq_repr)
        encoded = self.norm(encoded)
        
        # Use both last timestep and mean pooling for better representation
        context_last = encoded[:, -1, :]
        context_mean = encoded.mean(dim=1)
        context = context_last + 0.5 * context_mean  # Combine both

        logits = self.classifier_head(context).squeeze(-1)
        rtf = self.regression_head(context).squeeze(-1)

        return {"logits": logits, "rtf": rtf}
